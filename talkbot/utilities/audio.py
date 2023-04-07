"""Utilities for working with audio."""

import atexit
import signal
import threading
from contextlib import contextmanager
from typing import List

import numpy as np
import pyaudio
from numpy.typing import ArrayLike


def get_device_by_name(device_name: str, min_input_channels: int = 0, min_output_channels: int = 0) -> int:
    """Get the index of the first matched device."""
    for i, info in enumerate(list_device_info()):
        if (
            device_name in str(info["name"])
            and int(info["maxOutputChannels"]) >= min_output_channels
            and int(info["maxInputChannels"]) >= min_input_channels
        ):
            return i

    raise ValueError(f"No output device found with name containing '{device_name}'")


def list_device_info() -> List[dict]:
    """Get a list of device info dictionaries."""
    return [get_pa().get_device_info_by_index(i) for i in range(get_pa().get_device_count())]  # type: ignore


def list_host_api_info() -> List[dict]:
    """Get a list of host API info dictionaries."""
    return [get_pa().get_host_api_info_by_index(i) for i in range(get_pa().get_host_api_count())]  # type: ignore


def get_volume_range(audio_data: ArrayLike) -> float:
    """Get the range of the audio data."""
    return np.max(audio_data) - np.min(audio_data)


_pa_local = threading.local()


def get_pa() -> pyaudio.PyAudio:
    """Get a thread-local instance of pyaudio.PyAudio."""
    if not hasattr(_pa_local, "pa"):
        _pa_local.pa = pyaudio.PyAudio()
        atexit.register(_pa_local.pa.terminate)
    return _pa_local.pa


@contextmanager
def open_stream(
    *args,
    **kwargs,
):
    """Open a stream."""
    stream = get_pa().open(*args, **kwargs)
    yield stream
    stream.stop_stream()


def _on_sigint(*_):
    """Handle SIGINT."""
    if hasattr("_pa_local", "pa"):
        _pa_local.pa.terminate()


signal.signal(signal.SIGINT, _on_sigint)
