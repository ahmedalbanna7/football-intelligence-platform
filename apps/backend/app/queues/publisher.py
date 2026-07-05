import json

import aio_pika

from app.queues.events import MatchAnalysisRequestedEvent
from app.queues.events import VideoUploadedEvent
from app.queues.routing_keys import MATCH_ANALYSIS_EXCHANGE
from app.queues.routing_keys import MATCH_ANALYSIS_REQUESTED_ROUTING_KEY
from app.queues.routing_keys import VIDEO_EXCHANGE
from app.queues.routing_keys import VIDEO_UPLOADED_ROUTING_KEY
from app.services.rabbitmq import rabbitmq_channel
from app.services.rabbitmq import setup_match_analysis_topology
from app.services.rabbitmq import setup_video_topology


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


async def publish_match_analysis_requested(event: MatchAnalysisRequestedEvent) -> None:
    async with rabbitmq_channel() as channel:
        await setup_match_analysis_topology(channel)
        exchange = await channel.get_exchange(MATCH_ANALYSIS_EXCHANGE)
        payload = event.model_dump(mode="json")

        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=MATCH_ANALYSIS_REQUESTED_ROUTING_KEY,
        )
