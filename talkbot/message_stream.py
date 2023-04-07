"""MessageStream component."""

from typing import NoReturn

import zmq
import zmq.asyncio

from .utilities.config import Config
from .utilities.socket import get_context


async def message_stream(config: Config = Config()) -> NoReturn:
    """MessageStream component."""
    logger = config.get_logger("MessageStream")

    logger.info("Initializing")

    context = get_context()
    with context.socket(zmq.SUB) as sub_socket, context.socket(zmq.PUB) as pub_socket:
        sub_socket.bind(config.get("message_stream.write", required=True))
        sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        pub_socket.bind(config.get("message_stream.read", required=True))

        logger.info("Started")

        while True:
            message = await sub_socket.recv_json()  # type: ignore
            logger.debug("Message: %s", message)
            await pub_socket.send_json(message)  # type: ignore
