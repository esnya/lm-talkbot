"""Provide utilities for working with ZMQ sockets."""

import asyncio
import atexit
import signal
import sys
import threading
from abc import ABC
from contextlib import contextmanager
from typing import Any, Awaitable, Iterator, TypeVar

import zmq
import zmq.asyncio

from .config import Config
from .constants import ComponentState
from .message import StateMessage

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_local = threading.local()

T = TypeVar("T")
JsonValue = list[Any] | str | int | float | dict[Any, Any]


class Socket(ABC, zmq.asyncio.Socket):
    """A socket that can send and receive JSON messages."""

    def send_json(self, obj: Any, flags: int | None = None, **kwargs: Any) -> Awaitable[None]:  # type: ignore
        """Send a JSON-serializable object as a message."""
        raise NotImplementedError()

    def recv_json(self, flags: int | None = None, **kwargs: Any) -> Awaitable[JsonValue]:  # type: ignore
        """Receive a JSON-serializable object as a message."""
        raise NotImplementedError()


def get_context() -> zmq.asyncio.Context:
    """Return a context for creating sockets."""
    if not hasattr(_local, "context"):
        _local.context = zmq.asyncio.Context()
        atexit.register(_local.context.term)
    return _local.context


def _on_sigint(*_):
    """Handle SIGINT."""
    if hasattr(_local, "context"):
        _local.context.term()


signal.signal(signal.SIGINT, _on_sigint)


@contextmanager
def get_write_socket(config: Config) -> Iterator[Socket]:
    """Return a socket that sends messages to the message stream."""
    context = get_context()
    with context.socket(zmq.PUB) as socket:
        socket.connect(config.get("message_stream.write", ""))
        yield socket  # type: ignore


@contextmanager
def get_read_socket(config: Config, timeout: float | None = None) -> Iterator[Socket]:
    """Return a socket that receives messages from the message stream."""
    context = get_context()
    with context.socket(zmq.SUB) as socket:
        socket.connect(config.get("message_stream.read", ""))
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        if timeout is not None:
            socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        yield socket  # type: ignore


@contextmanager
def get_sockets(
    config: Config,
    timeout: float | None = None,
) -> Iterator[tuple[Socket, Socket]]:
    """Return a tuple of (write_socket, read_socket)."""
    with get_write_socket(config) as write_socket, get_read_socket(config, timeout) as read_socket:
        yield write_socket, read_socket


async def send_state(write_socket: Socket, component: str, state: ComponentState):
    """Send a state message to the message stream."""
    await write_socket.send_json(
        StateMessage(
            role="state",
            component=component,
            state=state.value,
        )
    )
