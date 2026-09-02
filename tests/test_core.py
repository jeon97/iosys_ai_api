import asyncio
import json
import unittest

from app.core import AiTaskService, InputCleaner, ModelResponseError, TaskRequest


class RecordingAudit:
    def __init__(self):
        self.records = []

    def save(self, record):
        self.records.append(record)


class SequenceGateway:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate_json(self, model, system, user):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class InputCleanerTest(unittest.TestCase):
    def test_removes_html_base64_and_control_characters(self):
        cleaner = InputCleaner()
        value = '<script>alert(1)</script><b>장애</b> data:image/png;base64,AAAA\x00  발생'
        self.assertEqual("장애 발생", cleaner.clean(value))


class AiTaskServiceTest(unittest.TestCase):
    def test_retries_invalid_json_and_validates_schema(self):
        gateway = SequenceGateway("not-json", json.dumps({"score": 91, "reason": "제출 장애"}))
        audit = RecordingAudit()
        service = AiTaskService(gateway, audit, {"llama3"})

        result = asyncio.run(service.execute(TaskRequest(
            instruction="위험도를 평가하세요.",
            input_data="시험 제출이 되지 않습니다.",
            response_schema={"score": "integer", "reason": "string"},
        )))

        self.assertEqual(91, result.data["score"])
        self.assertEqual(2, result.attempts)
        self.assertEqual("SUCCESS", audit.records[0].status)
        self.assertEqual(64, len(audit.records[0].input_sha256))

    def test_rejects_unapproved_model_before_gateway_call(self):
        gateway = SequenceGateway('{}')
        service = AiTaskService(gateway, RecordingAudit(), {"llama3"})

        with self.assertRaisesRegex(ValueError, "not allowed"):
            asyncio.run(service.execute(TaskRequest(
                instruction="분류하세요.", input_data="내용",
                response_schema={"result": "string"}, model="unknown",
            )))
        self.assertEqual(0, gateway.calls)

    def test_fails_after_bounded_schema_retries(self):
        gateway = SequenceGateway('{"wrong": 1}', '{"wrong": 2}', '{"wrong": 3}')
        audit = RecordingAudit()
        service = AiTaskService(gateway, audit, {"llama3"})

        with self.assertRaises(ModelResponseError):
            asyncio.run(service.execute(TaskRequest(
                instruction="분류하세요.", input_data="내용",
                response_schema={"result": "string"},
            )))

        self.assertEqual(3, gateway.calls)
        self.assertEqual("ERROR", audit.records[0].status)


if __name__ == "__main__":
    unittest.main()

