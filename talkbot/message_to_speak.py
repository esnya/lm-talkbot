"""Message to speak"""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

import numpy as np
import pyaudio
import zmq

from talkbot.utilities.message import is_assistant_message

from .utilities.audio import get_device_by_name, get_pa
from .utilities.config import Config
from .utilities.constants import ComponentState
from .utilities.socket import Socket, get_context, send_state
from .utilities.voicevox import audio_query, synthesis, version


class MessageToSpeak(AbstractContextManager, ABC):
    def __init__(self, config: Config = Config()):
        self.config = config
        logger = self.logger = config.get_logger("MessageToSpeak")

        """Message to speak component"""

        logger.info("Initializing")

        self.english_kana_dict: dict[str, str] = {}

        english_kana_dict_filename = config.get("message_to_speak.english_kana_dict")
        if english_kana_dict_filename and os.path.exists(english_kana_dict_filename):
            logger.info("Loading English to Kana dictionary")
            with open(english_kana_dict_filename, "r", encoding="utf-8") as file:
                next(file)
                self.english_kana_dict = {
                    col[0].strip().lower(): col[1].strip()
                    for col in (line.split(",") for line in file.readlines() if line and not line.startswith("#"))
                    if len(col) >= 2
                }
                logger.info("Loaded English to Kana dictionary (%d entries)", len(self.english_kana_dict))

        logger.info("Initialized")

    def _stream_format(self):
        return {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 24000,
        }

    def __enter__(self):
        self.write_socket: Socket = get_context().socket(zmq.PUB)  # type: ignore
        self.write_socket.connect(self.config.get("message_stream.write", ""))

        self.read_socket: Socket = get_context().socket(zmq.SUB)  # type: ignore
        self.read_socket.connect(self.config.get("message_stream.read", ""))
        self.read_socket.setsockopt(zmq.SUBSCRIBE, b"")

        device_name = self.config.get("message_to_speak.output_device.name")
        output_device_index = 0 if device_name is None else get_device_by_name(device_name, min_output_channels=1)
        self.logger.debug(
            "Playing on: %s",
            get_pa().get_device_info_by_index(output_device_index),
        )
        self.stream = get_pa().open(
            output_device_index=output_device_index,
            output=True,
            **self._stream_format(),
        )

        return self

    def __exit__(self, *args):
        self.write_socket.close()
        self.read_socket.close()
        self.stream.close()

    async def play(self, data: bytes):
        await asyncio.to_thread(self.stream.write, data)

    async def run(self):
        logger = self.logger
        config = self.config
        write_socket = self.write_socket
        read_socket = self.read_socket

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

            if self.english_kana_dict:
                text = re.sub(
                    "[a-zA-Z]+",
                    lambda m: self.english_kana_dict.get(m.group(0).lower(), m.group(0)),
                    text,
                )

            if not text:
                continue

            await self.play_text(text)

        logger.info("Terminated")

    @abstractmethod
    async def play_text(self, text: str):
        raise NotImplementedError()


class MessageToVoicevox(MessageToSpeak):
    async def run(self):
        self.logger.info("VOICEVOX version: %s", await version())
        await super().run()

    async def play_text(self, text: str):
        config = self.config
        logger = self.logger

        speaker = config.get("message_to_speak.voicevox.speaker", 1)
        segments = re.split(config.get("message_to_speak.split_pattern", r"。"), text)

        prev_task: asyncio.Task | None = None
        for segment in segments:
            if not segment:
                continue

            logger.info("Query: %s", segment)

            query = await audio_query(segment, speaker)
            logger.info("Synthesis: %d", speaker)

            data = await synthesis(query, speaker)

            if prev_task:
                await prev_task

            await send_state(self.write_socket, "MessageToSpeak", ComponentState.BUSY)
            prev_task = asyncio.create_task(self.play(data[44:]))

        if prev_task:
            await prev_task
        await send_state(self.write_socket, "MessageToSpeak", ComponentState.READY)


class MessageToRVC(MessageToVoicevox):
    def __init__(self, config: Config = Config()):
        import sys

        import torch
        from rvc_eval.model import load_hubert, load_net_g

        sys.path.append(os.path.join(os.path.dirname(__file__), "../rvc"))

        super().__init__(config)
        super().__init__(config)

        device = self.device = torch.device(
            self.config.get("message_to_speak.rvc.device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.is_half = self.config.get("message_to_speak.rvc.is_half", True)

        self.logger.info("Loading hubert model")
        self.hubert = load_hubert(config.get("message_to_speak.rvc.hubert", True), self.is_half, device)

        self.logger.info("Loading RVC model")
        self.net_g, self.sample_rate = load_net_g(config.get("message_to_speak.rvc.model", True), self.is_half, device)

    def _stream_format(self):
        return {
            "format": pyaudio.paFloat32,
            "channels": 1,
            "rate": 48000,
        }

    async def play_text(self, text: str):
        from rvc_eval.vc_infer_pipeline import VC

        quality = self.config.get("message_to_speak.rvc.quality", 1)
        self.vc = VC(self.sample_rate, self.device, self.is_half, (3 if self.is_half else 1) * quality)
        await super().play_text(text)

    async def play(self, data: bytes):
        import resampy

        resampled = resampy.resample(np.frombuffer(data, np.int16), 24000, 16000, axis=0)
        rvc_output = await asyncio.to_thread(
            self.vc.pipeline,
            self.hubert,
            self.net_g,
            0,
            resampled,
            int(self.config.get("message_to_speak.rvc.f0_up_key", 0)),
            self.config.get("message_to_speak.rvc.f0_method", "pm"),
        )

        await super().play(
            (rvc_output.cpu().float().numpy() * self.config.get("message_to_speak.rvc.volume", 1.0)).tobytes()
        )


def get_class_by_name(name: str) -> type[MessageToSpeak]:
    match name:
        case "rvc":
            return MessageToRVC
        case "voicevox":
            return MessageToVoicevox
        case _:
            raise ValueError(f"Unknown engine: {name}")


async def message_to_speak(config: Config = Config()) -> None:
    Impl = get_class_by_name(config.get("message_to_speak.engine", "voicevox"))

    with Impl(config) as message_to_speak:
        await message_to_speak.run()
