"""Audio To Message component."""

import asyncio
import math
from functools import lru_cache
from typing import Any, TypedDict, TypeGuard

import numpy as np
import pyaudio
import torch
import whisper
import zmq
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core.annotation import Annotation
from whisper.audio import N_SAMPLES, SAMPLE_RATE

from .utilities.audio import get_device_by_name, get_pa, open_stream
from .utilities.config import Config
from .utilities.constants import ComponentState
from .utilities.message import UserMessage, is_busy, update_busy_time
from .utilities.socket import get_sockets, send_state


class DecodingResultSegment(TypedDict):
    """Decoding result segment."""

    text: str
    compression_ratio: float
    avg_logprob: float
    no_speech_prob: float
    temperature: float


def is_whisper_result_segment(value) -> TypeGuard[DecodingResultSegment]:
    return isinstance(value, dict) and "text" in value


async def async_read_mic_input(stream: pyaudio.Stream, chunk: int) -> np.ndarray:
    """Read a chunk of audio from the microphone input stream."""
    data = await asyncio.to_thread(stream.read, chunk)
    return np.frombuffer(data, dtype=np.float32)


async def audio_to_message(config: Config = Config()) -> None:
    """Convert audio to text messages."""
    logger = config.get_logger("AudioToMessage")
    logger.info("Initializing...")

    @lru_cache(maxsize=1)
    def load_model(model_name: str | None = None, device: str | None = None):
        logger.info("Loading model %s (%s) ...", model_name, device)

        if device == "cuda" and not torch.cuda.is_initialized():
            logger.info("Initializing CUDA...")
            torch.cuda.init()

        return whisper.load_model(model_name or "base", device=device)

    load_model(
        config.get("audio_to_message.whisper.model", "base"),
        config.get(
            "audio_to_message.whisper.device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    )

    logger.info("Loading VAD pipeline")
    vad_pipeline = VoiceActivityDetection(**config.get("audio_to_message.vad.pipeline", {}))
    vad_pipeline.instantiate(config.get("audio_to_message.vad.hyper_parameters", {}))

    def filter_segment(
        segment: Any,
    ) -> TypeGuard[DecodingResultSegment]:
        if not is_whisper_result_segment(segment):
            return False

        text: str = segment["text"]
        compression_ratio: float = segment["compression_ratio"]
        avg_logprob: float = segment["avg_logprob"]
        no_speech_prob: float = segment["no_speech_prob"]
        temperature: float = segment["temperature"]
        logger.info(
            "Seg: comp=%.3f logprob=%.3f no_speech=%.3f temp=%.3f %s",
            compression_ratio,
            avg_logprob,
            no_speech_prob,
            temperature,
            text,
        )
        return (
            bool(text)
            and text not in config.get("audio_to_message.whisper.blacklist", [])
            and math.isfinite(compression_ratio * avg_logprob * no_speech_prob)
            and temperature <= config.get("audio_to_message.whisper.max_temperature", 1.0)
        )

    _initial_prompt: str | None = None

    def transcribe(data: np.ndarray) -> str:
        nonlocal _initial_prompt

        model = load_model(
            config.get("audio_to_message.whisper.model", "base"),
            config.get(
                "audio_to_message.whisper.device",
                "cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
        logger.info("Transcribing: frames=%d", data.size)
        result = whisper.transcribe(
            model, data, initial_prompt=_initial_prompt, **config.get("audio_to_message.whisper.decode_config", {})
        )

        text = "".join([segment.get("text") for segment in result["segments"] if filter_segment(segment)])
        if text:
            _initial_prompt = (_initial_prompt or "") + text

        prompt_length: int = config.get("audio_to_message.whisper.prompt_length") or 0
        if prompt_length is None or prompt_length == 0:
            _initial_prompt = None
        else:
            _initial_prompt = ((_initial_prompt or "") + "\n")[-prompt_length:]

        return text

    buffer = np.empty((0,), np.float32)

    device_name: str | None = config.get("audio_to_message.input_device.name")
    device_index = 0
    if device_name is not None:
        device_index = get_device_by_name(device_name, min_input_channels=1)

    def stream_callback(in_data: bytes, frame_count: int, time_info, status):
        nonlocal buffer

        audio_data = np.frombuffer(in_data, dtype=np.float32)
        buffer = np.concatenate((buffer, audio_data))

        return (audio_data, pyaudio.paContinue)

    _stream_frames_per_buffer = config.get("audio_to_message.input_device.frames_per_buffer", 1024)
    with get_sockets(config, 0.5) as (write_socket, read_socket), open_stream(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=_stream_frames_per_buffer,
        stream_callback=stream_callback,
    ) as stream:
        last_busy_time = 0.0
        logger.info("Initialized")
        stream.start_stream()
        logger.info("Started: recording device i %s", get_pa().get_device_info_by_index(device_index))
        await send_state(write_socket, "AudioToMessage", ComponentState.READY)
        while not write_socket.closed and not read_socket.closed:
            try:
                message = await read_socket.recv_json()
                last_busy_time = update_busy_time(message, "ChatEngine", last_busy_time)
            except zmq.error.Again:
                pass

            if buffer.size == 0:
                continue

            silence_duration = config.get("message_to_message.silence_duration", 1.0)

            max_buffer_size: int = config.get("audio_to_message.max_buffer_size", N_SAMPLES)

            if is_busy(last_busy_time, config.get("global.busy_timeout", 30.0)):
                await asyncio.sleep(1)
                continue

            vad: Annotation = vad_pipeline(
                {"waveform": torch.from_numpy(buffer.reshape((1, -1))), "sample_rate": SAMPLE_RATE}
            )
            timeline = vad.get_timeline(False)
            segments: list[tuple[(int, int)]] = [
                (max(int((seg.start - 0.5) * SAMPLE_RATE), 0), int(math.ceil((seg.end + 0.5) * SAMPLE_RATE)))
                for seg in timeline  # type: ignore
            ]

            if len(segments) == 0:
                continue

            left = segments[0][0]
            right = segments[-2][1] if len(segments) >= 2 else segments[-1][1]
            if (
                len(segments) >= 2
                or right - left >= max_buffer_size
                or buffer.size - right >= silence_duration * SAMPLE_RATE
            ):
                await send_state(write_socket, "AudioToMessage", ComponentState.BUSY)
                text = await asyncio.to_thread(transcribe, buffer[left:right])
                buffer = buffer[right:]
                if text:
                    await write_socket.send_json(
                        UserMessage(
                            role="user",
                            content=text,
                        )
                    )
                await send_state(write_socket, "AudioToMessage", ComponentState.READY)
            elif left > 0:
                buffer = buffer[left:]
    logger.info("Terminated")
