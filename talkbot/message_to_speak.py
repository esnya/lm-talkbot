"""Message to speak"""

import asyncio
import os
import re
from io import BytesIO
from wave import Wave_read

from talkbot.utilities.message import is_assistant_message

from .utilities.audio import get_device_by_name, get_pa, open_stream
from .utilities.config import Config
from .utilities.constants import ComponentState
from .utilities.socket import get_sockets, send_state
from .utilities.voicevox import audio_query, synthesis, version


async def message_to_speak(config: Config = Config()) -> None:
    """Message to speak component"""

    logger = config.get_logger("MessageToSpeak")
    logger.info("Initializing")

    english_kana_dict: dict[str, str] = {}

    logger.info("VOICEVOX version: %s", await version())

    english_kana_dict_filename = config.get("message_to_speak.english_kana_dict")
    if english_kana_dict_filename and os.path.exists(english_kana_dict_filename):
        logger.info("Loading English to Kana dictionary")
        with open(english_kana_dict_filename, "r", encoding="utf-8") as file:
            next(file)
            english_kana_dict = {
                col[0].strip().lower(): col[1].strip()
                for col in (line.split(",") for line in file.readlines() if line and not line.startswith("#"))
                if len(col) >= 2
            }
            logger.info("Loaded English to Kana dictionary (%d entries)", len(english_kana_dict))

    logger.info("Initialized")

    def _play(data: bytes):
        device_name = config.get("message_to_speak.output_device.name")
        output_device_index = 0
        if device_name is not None:
            output_device_index = get_device_by_name(device_name, min_output_channels=1)
        with Wave_read(BytesIO(data)) as wav, open_stream(
            output_device_index=output_device_index,
            format=get_pa().get_format_from_width(wav.getsampwidth()),
            channels=wav.getnchannels(),
            rate=wav.getframerate(),
            output=True,
        ) as stream:
            logger.debug(
                "Playing on: %s",
                get_pa().get_device_info_by_index(output_device_index),
            )
            stream.write(data[44:])
            stream.stop_stream()

    async def play(data: bytes):
        await asyncio.to_thread(_play, data)

    with get_sockets(config) as (write_socket, read_socket):
        logger.info("Started")
        await send_state(write_socket, "MessageToSpeak", ComponentState.READY)

        while not write_socket.closed and not read_socket.closed:
            message = await read_socket.recv_json()
            if not is_assistant_message(message):
                continue

            match = re.search(
                config.get("message_to_speak.talk_pattern", r"(.+)"),
                message["content"],
                re.MULTILINE,
            )

            if not match:
                continue

            text = match.group(1)

            remove_pattern = config.get("message_to_speak.remove_pattern")
            if remove_pattern:
                text = re.sub(remove_pattern, "", text)

            if english_kana_dict:
                text = re.sub(
                    "[a-zA-Z]+",
                    lambda m: english_kana_dict.get(m.group(0).lower(), m.group(0)),
                    text,
                )

            if not text:
                continue

            speaker = config.get("message_to_speak.voicevox.speaker", 1)
            logger.info("Query: %s", text)

            query = await audio_query(text, speaker)
            logger.info("Synthesis: %d", speaker)

            data = await synthesis(query, speaker)

            await send_state(write_socket, "MessageToSpeak", ComponentState.BUSY)
            await play(data)
            await send_state(write_socket, "MessageToSpeak", ComponentState.READY)

        logger.info("Terminated")
