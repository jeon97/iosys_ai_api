from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class BatchRecord:
    record_id: str
    text: str


@dataclass(frozen=True)
class BatchFailure:
    record_id: str
    failure_type: str


@dataclass(frozen=True)
class BatchSummary:
    succeeded: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[BatchFailure, ...]


class CheckpointStore(Protocol):
    def is_complete(self, run_id: str, record_id: str) -> bool: ...
    def save_result(self, run_id: str, record_id: str, result: dict) -> None: ...
    def save_failure(self, run_id: str, record_id: str, failure_type: str) -> None: ...


class BatchProcessor:
    def __init__(self, evaluate: Callable[[str], dict], checkpoints: CheckpointStore):
        self._evaluate = evaluate
        self._checkpoints = checkpoints

    def run(self, run_id: str, records: Iterable[BatchRecord]) -> BatchSummary:
        if not run_id.strip():
            raise ValueError("run_id is required")

        values = list(records)
        ids = [record.record_id for record in values]
        if any(not value.strip() for value in ids):
            raise ValueError("record_id is required")
        if len(ids) != len(set(ids)):
            raise ValueError("record_id must be unique")

        succeeded: list[str] = []
        skipped: list[str] = []
        failed: list[BatchFailure] = []
        for record in values:
            if self._checkpoints.is_complete(run_id, record.record_id):
                skipped.append(record.record_id)
                continue
            try:
                if not record.text.strip():
                    raise ValueError("input is empty")
                result = self._evaluate(record.text)
                if not isinstance(result, dict):
                    raise TypeError("evaluation result must be an object")
                self._checkpoints.save_result(run_id, record.record_id, result)
                succeeded.append(record.record_id)
            except Exception as error:  # 건별 실패를 격리해 다음 레코드를 계속 처리한다.
                failure_type = type(error).__name__
                self._checkpoints.save_failure(run_id, record.record_id, failure_type)
                failed.append(BatchFailure(record.record_id, failure_type))

        return BatchSummary(tuple(succeeded), tuple(skipped), tuple(failed))
