"""Talkbot main entry point."""

import asyncio
import signal
from typing import Any, Callable, Coroutine

from .audio_to_message import audio_to_message
from .bridge import bridge
from .chat_engine import chat_engine
from .console import console
from .message_stream import message_stream
from .message_to_speak import message_to_speak
from .neos_connector import neos_connector
from .prune import prune
from .train import train
from .utilities.asyncargh import AsyncArghParser
from .utilities.config import Config

COMPONENTS: list[Callable[..., Coroutine[Any, Any, Any]]] = [
    neos_connector,
    audio_to_message,
    message_to_speak,
    chat_engine,
    message_stream,
]


async def run(config: Config = Config()):
    """Talkbot main entry point."""
    tasks = [asyncio.create_task(component(config=config)) for component in COMPONENTS]

    if not tasks:
        raise ValueError("No components to run")

    await asyncio.gather(*tasks)


async def run_bridge(
    config1: Config = Config(),
    config2: Config = Config(),
    sleep: float = 10.0,
    no_whisper: bool = False,
):
    """Run a bridge between two talkbots."""
    tasks = [
        bridge(config1=config1, config2=config2, sleep=sleep),
        *[(component(config=config1)) for component in COMPONENTS if not no_whisper or component != audio_to_message],
        *[(component(config=config2)) for component in COMPONENTS if component != audio_to_message],
    ]

    await asyncio.gather(*[asyncio.create_task(task) for task in tasks])


def _on_sigint(*_):
    """Handle SIGINT."""
    for task in asyncio.all_tasks():
        task.cancel()


signal.signal(signal.SIGINT, _on_sigint)

parser = AsyncArghParser()
parser.add_async_commands(COMPONENTS)
parser.add_async_commands([run])
parser.add_async_commands([console])
parser.add_async_commands([train])
parser.add_async_commands([prune])
parser.add_async_commands([run_bridge])
parser.set_async_default_command(run)
parser.dispatch()
