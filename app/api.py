from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.audit import SqliteAuditRepository
from app.core import AiTaskService, ModelResponseError, TaskRequest
from app.ollama_gateway import OllamaGateway


class ApiTaskRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=10_000)
    inputData: str = Field(min_length=1)
    responseSchema: dict[str, str] = Field(min_length=1, max_length=30)
    model: str = "llama3"


class ApiTaskResponse(BaseModel):
    result: str
    data: dict[str, Any]
    usageInfo: dict[str, Any]


allowed_models = {
    item.strip() for item in os.getenv("ALLOWED_MODELS", "llama3").split(",") if item.strip()
}
service = AiTaskService(
    OllamaGateway(),
    SqliteAuditRepository(Path(os.getenv("AUDIT_DB", "ai_audit.db"))),
    allowed_models,
)
app = FastAPI(title="Generic Local LLM JSON API", version="1.0.0")


@app.post("/api/v1/tasks", response_model=ApiTaskResponse)
async def execute_task(body: ApiTaskRequest) -> ApiTaskResponse:
    try:
        result = await service.execute(TaskRequest(
            instruction=body.instruction,
            input_data=body.inputData,
            response_schema=body.responseSchema,
            model=body.model,
        ))
        return ApiTaskResponse(
            result="success",
            data=dict(result.data),
            usageInfo={
                "model": result.model,
                "durationMs": result.duration_ms,
                "attempts": result.attempts,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

