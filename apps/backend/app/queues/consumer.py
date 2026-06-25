import json
from collections.abc import AsyncIterator

from aio_pika.abc import AbstractIncomingMessage

from app.queues.events import VideoUploadedEvent
from app.queues.routing_keys import VIDEO_PROCESSING_QUEUE
from app.services.rabbitmq import rabbitmq_channel, setup_video_topology


async def consume_video_uploaded_events() -> AsyncIterator[
    tuple[AbstractIncomingMessage, VideoUploadedEvent]
]:
    async with rabbitmq_channel() as channel:
        await setup_video_topology(channel)
        queue = await channel.get_queue(VIDEO_PROCESSING_QUEUE)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                payload = json.loads(message.body.decode("utf-8"))
                yield message, VideoUploadedEvent.model_validate(payload)
