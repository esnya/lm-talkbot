"""Bridge between two talkbots."""

import re
import time

import zmq.asyncio

from .utilities.config import Config
from .utilities.message import AssistantMessage, UserMessage, is_assistant_message
from .utilities.socket import Socket, get_sockets


async def bridge(config1: Config = Config(), config2: Config = Config(), sleep: float = 10.0):
    """Bridge between two talkbots."""
    logger = config1.get_logger("Bridge")
    logger.info("Initializing")

    async def _bridge(
        read: Socket,
        write: Socket,
        remove_pattern: str,
        log_format: str,
        sleep: float,
    ):
        """Bridge from socket1 to socket2."""
        start_time = time.time()
        messages: list[UserMessage] = []

        def append_message(message: AssistantMessage):
            messages.append(
                UserMessage(
                    role="user",
                    content=re.sub(
                        remove_pattern,
                        "",
                        message["content"],
                    ),
                )
            )
            logger.info(log_format.format(message["content"]))

        while time.time() - start_time < sleep:
            try:
                message = await read.recv_json()
                if is_assistant_message(message):
                    append_message(message)
            except zmq.error.Again:
                pass

        for msg in messages or [{"role": "user", "content": " "}]:
            await write.send_json(msg)

    with get_sockets(config1, sleep) as (write_socket1, read_socket1), get_sockets(config2, sleep) as (
        write_socket2,
        read_socket2,
    ):
        logger.info("Initialized")
        logger.info("Started")
        while not (write_socket1.closed or write_socket2.closed or read_socket1.closed or read_socket2.closed):
            logger.info("1 >>> 2")
            await _bridge(
                read_socket1,
                write_socket2,
                config1.get("message_to_speak.remove_pattern", ""),
                "「{}」",
                sleep,
            )
            logger.info("1 <<< 2")
            await _bridge(
                read_socket2,
                write_socket1,
                config2.get("message_to_speak.remove_pattern", ""),
                "『{}』",
                sleep,
            )

    logger.info("Terminated")
