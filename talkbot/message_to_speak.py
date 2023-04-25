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
        from hf_rvc import RVCFeatureExtractor, RVCModel

        super().__init__(config)

        self.logger.info("Loading RVC model")

        device = "cpu"

        rvc_model_path = config.get("message_to_speak.rvc.model", True)
        rvc_model = RVCModel.from_pretrained(rvc_model_path)
        assert isinstance(rvc_model, RVCModel)
        if self.config.get("message_to_speak.rvc.fp16", device != "cpu"):
            rvc_model = rvc_model.half()
            self.fp16 = True
        else:
            self.fp16 = False

        rvc_model = rvc_model.to(device)
        rvc_model.vits = rvc_model.vits.to(device)
        self.rvc_model = rvc_model

        feature_extractor = RVCFeatureExtractor.from_pretrained(rvc_model_path)
        assert isinstance(feature_extractor, RVCFeatureExtractor)
        self.rvc_feature_extractor = feature_extractor

    def _stream_format(self):
        return {
            "format": pyaudio.paFloat32,
            "channels": 1,
            "rate": 48000,
        }

    async def play_text(self, text: str):
        import torch

        await super().play_text(text)
        torch.cuda.empty_cache()

    async def play(self, data: bytes):
        import resampy
        import torch

        if len(data) == 0:
            return
        with torch.no_grad():
            resampled = resampy.resample(np.frombuffer(data, np.int16), 24000, 16000, axis=0)
            self.rvc_feature_extractor.set_f0_method(self.config.get("message_to_speak.rvc.f0_method", "pm"))
            features = self.rvc_feature_extractor(
                resampled,
                f0_up_key=int(self.config.get("message_to_speak.rvc.f0_up_key", 0)),
                sampling_rate=16000,
                return_tensors="pt",
            )
            dtype = torch.float16 if self.fp16 else torch.float32
            rvc_output = await asyncio.to_thread(
                self.rvc_model,
                input_values=features.input_values.to(self.rvc_model.device, dtype=dtype),
                f0=features.f0.to(self.rvc_model.device, dtype=dtype),
                f0_coarse=features.f0_coarse.to(self.rvc_model.device),
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
