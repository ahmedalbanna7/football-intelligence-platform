import asyncio
import traceback

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.match import Match
from app.pipeline.video_pipeline import VideoPipeline
from app.queues.consumer import consume_video_uploaded_events


def update_match_status(db: Session, match_id: int, status: str) -> None:
    match = db.get(Match, match_id)
    if match is None:
        return

    match.status = status
    db.commit()


async def run_worker() -> None:
    pipeline = VideoPipeline()

    async for message, event in consume_video_uploaded_events():
        db = SessionLocal()
        try:
            update_match_status(db, event.match_id, "processing")
            pipeline.run(event.bucket, event.object_name)
            update_match_status(db, event.match_id, "processed")
            await message.ack()
        except Exception:
            update_match_status(db, event.match_id, "failed")
            traceback.print_exc()
            await message.nack(requeue=False)
        finally:
            db.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
