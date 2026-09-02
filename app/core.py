from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


ALLOWED_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


class InputCleaner:
    def __init__(self, max_length: int = 100_000) -> None:
        self.max_length = max_length

    def clean(self, value: str) -> str:
        text = (value or "")[: self.max_length]
        text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", " ", text)
        text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return re.sub(r"\s+", " ", text).strip()


class ModelGateway(Protocol):
    async def generate_json(self, model: str, system: str, user: str) -> str: ...


class AuditRepository(Protocol):
    def save(self, record: "AuditRecord") -> None: ...


@dataclass(frozen=True)
class AuditRecord:
    model: str
    status: str
    input_sha256: str
    input_length: int
    duration_ms: int
    attempts: int
    error_type: str | None = None


@dataclass(frozen=True)
class TaskRequest:
    instruction: str
    input_data: str
    response_schema: Mapping[str, str]
    model: str = "llama3"


@dataclass(frozen=True)
class TaskResult:
    data: Mapping[str, Any]
    model: str
    duration_ms: int
    attempts: int


class AiTaskService:
    def __init__(
        self,
        gateway: ModelGateway,
        audit: AuditRepository,
        allowed_models: set[str],
        cleaner: InputCleaner | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.gateway = gateway
        self.audit = audit
        self.allowed_models = set(allowed_models)
        self.cleaner = cleaner or InputCleaner()
        self.max_attempts = max_attempts

    async def execute(self, request: TaskRequest) -> TaskResult:
        self._validate_request(request)
        cleaned = self.cleaner.clean(request.input_data)
        if not cleaned:
            raise ValueError("input_data has no usable text")

        started = time.perf_counter()
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = await self.gateway.generate_json(
                    request.model, self._build_system_message(request), cleaned
                )
                data = json.loads(raw)
                self._validate_response(data, request.response_schema)
                duration = int((time.perf_counter() - started) * 1000)
                self.audit.save(AuditRecord(
                    request.model, "SUCCESS", digest, len(cleaned), duration, attempt
                ))
                return TaskResult(data, request.model, duration, attempt)
            except (json.JSONDecodeError, ResponseSchemaError) as exc:
                last_error = exc

        duration = int((time.perf_counter() - started) * 1000)
        self.audit.save(AuditRecord(
            request.model, "ERROR", digest, len(cleaned), duration,
            self.max_attempts, type(last_error).__name__ if last_error else "UnknownError"
        ))
        raise ModelResponseError("model did not return the requested JSON") from last_error

    def _validate_request(self, request: TaskRequest) -> None:
        if request.model not in self.allowed_models:
            raise ValueError("model is not allowed")
        if not request.instruction.strip():
            raise ValueError("instruction is required")
        if not request.response_schema or len(request.response_schema) > 30:
            raise ValueError("response_schema must contain 1 to 30 keys")
        unknown = set(request.response_schema.values()) - set(ALLOWED_TYPES)
        if unknown:
            raise ValueError(f"unsupported schema type: {sorted(unknown)}")

    @staticmethod
    def _build_system_message(request: TaskRequest) -> str:
        schema = json.dumps(request.response_schema, ensure_ascii=False, sort_keys=True)
        return (
            f"{request.instruction.strip()}\n"
            "Return one JSON object only. Do not add markdown or explanation.\n"
            f"Required key and type map: {schema}"
        )

    @staticmethod
    def _validate_response(data: Any, schema: Mapping[str, str]) -> None:
        if not isinstance(data, dict):
            raise ResponseSchemaError("response is not an object")
        if set(data) != set(schema):
            raise ResponseSchemaError("response keys do not match")
        for key, type_name in schema.items():
            expected = ALLOWED_TYPES[type_name]
            if type_name in {"integer", "number"} and isinstance(data[key], bool):
                raise ResponseSchemaError(f"{key} has invalid type")
            if not isinstance(data[key], expected):
                raise ResponseSchemaError(f"{key} has invalid type")


class ResponseSchemaError(ValueError):
    pass


class ModelResponseError(RuntimeError):
    pass

