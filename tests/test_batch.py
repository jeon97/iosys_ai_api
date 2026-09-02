import unittest

from app.batch import BatchProcessor, BatchRecord


class MemoryCheckpoints:
    def __init__(self):
        self.completed = set()
        self.results = {}
        self.failures = {}

    def is_complete(self, run_id, record_id):
        return (run_id, record_id) in self.completed

    def save_result(self, run_id, record_id, result):
        self.completed.add((run_id, record_id))
        self.results[(run_id, record_id)] = result

    def save_failure(self, run_id, record_id, failure_type):
        self.failures[(run_id, record_id)] = failure_type


class BatchProcessorTest(unittest.TestCase):
    def test_continues_after_individual_failure(self):
        store = MemoryCheckpoints()

        def evaluate(text):
            if text == "fail":
                raise TimeoutError("model timeout")
            return {"score": len(text)}

        result = BatchProcessor(evaluate, store).run("run-1", [
            BatchRecord("1", "normal"),
            BatchRecord("2", "fail"),
            BatchRecord("3", "next"),
        ])

        self.assertEqual(("1", "3"), result.succeeded)
        self.assertEqual("TimeoutError", result.failed[0].failure_type)

    def test_skips_completed_checkpoint_on_retry(self):
        store = MemoryCheckpoints()
        store.completed.add(("run-1", "1"))
        calls = []
        processor = BatchProcessor(lambda text: calls.append(text) or {"ok": True}, store)

        result = processor.run("run-1", [BatchRecord("1", "old"), BatchRecord("2", "new")])

        self.assertEqual(("1",), result.skipped)
        self.assertEqual(["new"], calls)

    def test_rejects_duplicate_record_ids(self):
        with self.assertRaises(ValueError):
            BatchProcessor(lambda text: {}, MemoryCheckpoints()).run(
                "run-1", [BatchRecord("1", "a"), BatchRecord("1", "b")]
            )


if __name__ == "__main__":
    unittest.main()
