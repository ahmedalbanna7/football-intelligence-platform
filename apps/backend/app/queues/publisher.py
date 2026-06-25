import json

import aio_pika

from app.queues.events import VideoUploadedEvent
from app.queues.routing_keys import VIDEO_EXCHANGE, VIDEO_UPLOADED_ROUTING_KEY
from app.services.rabbitmq import rabbitmq_channel, setup_video_topology


async def publish_video_uploaded(event: VideoUploadedEvent) -> None:
    async with rabbitmq_channel() as channel:
        await setup_video_topology(channel)
        exchange = await channel.get_exchange(VIDEO_EXCHANGE)
        payload = event.model_dump(mode="json")

        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=VIDEO_UPLOADED_ROUTING_KEY,
        )
