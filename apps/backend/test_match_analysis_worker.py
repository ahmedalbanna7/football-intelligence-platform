import unittest
from types import SimpleNamespace

from app.workers.match_analysis_worker import (
    run_is_already_complete,
    settle_message,
)


class FakeMessage:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.acked = False
        self.nacked = False

    async def ack(self) -> None:
        if self.fail:
            raise RuntimeError("channel closed")
        self.acked = True

    async def nack(self, requeue: bool = False) -> None:
        if self.fail:
            raise RuntimeError("channel closed")
        self.nacked = not requeue


class MatchAnalysisWorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_completed_run_is_idempotent(self) -> None:
        completed = SimpleNamespace(
            status="processed",
            summary_json={"status": "ok"},
            output_object="matches/12/output.mp4",
        )
        incomplete = SimpleNamespace(
            status="processing",
            summary_json=None,
            output_object=None,
        )

        self.assertTrue(run_is_already_complete(completed))
        self.assertFalse(run_is_already_complete(incomplete))
        self.assertFalse(run_is_already_complete(None))

    async def test_message_can_be_acked_or_rejected(self) -> None:
        acknowledged = FakeMessage()
        rejected = FakeMessage()

        self.assertTrue(await settle_message(acknowledged, "ack"))
        self.assertTrue(acknowledged.acked)
        self.assertTrue(await settle_message(rejected, "nack"))
        self.assertTrue(rejected.nacked)

    async def test_closed_channel_is_handled_without_worker_crash(self) -> None:
        self.assertFalse(await settle_message(FakeMessage(fail=True), "ack"))


if __name__ == "__main__":
    unittest.main()
