import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractRobustConnection

from app.core.config import settings
from app.queues.routing_keys import (
    VIDEO_EXCHANGE,
    VIDEO_PROCESSING_QUEUE,
    VIDEO_UPLOADED_ROUTING_KEY,
)

CONNECT_RETRIES = 30
CONNECT_RETRY_DELAY_SECONDS = 2


async def connect() -> AbstractRobustConnection:
    last_error: Exception | None = None

    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            return await aio_pika.connect_robust(settings.RABBITMQ_URL)
        except Exception as exc:
            last_error = exc
            print(
                "RabbitMQ is not ready yet "
                f"(attempt {attempt}/{CONNECT_RETRIES}): {exc}"
            )
            await asyncio.sleep(CONNECT_RETRY_DELAY_SECONDS)

    raise RuntimeError("RabbitMQ connection retries exhausted") from last_error


@asynccontextmanager
async def rabbitmq_channel() -> AsyncIterator[AbstractChannel]:
    connection = await connect()
    try:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        yield channel
    finally:
        await connection.close()


async def setup_video_topology(channel: AbstractChannel) -> None:
    exchange = await channel.declare_exchange(
        VIDEO_EXCHANGE,
        aio_pika.ExchangeType.TOPIC,
        durable=True,
    )
    queue = await channel.declare_queue(
        VIDEO_PROCESSING_QUEUE,
        durable=True,
    )
    await queue.bind(exchange, routing_key=VIDEO_UPLOADED_ROUTING_KEY)
