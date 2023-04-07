"""Message types."""

import time
from typing import Any, Literal, TypedDict, TypeGuard

from .constants import ComponentState


class UserMessage(TypedDict):
    """A message from the user."""

    role: Literal["user"]
    content: str


class AssistantMessage(TypedDict):
    """A message from the assistant."""

    role: Literal["assistant"]
    content: str


class SystemMessage(TypedDict):
    """A message from the system."""

    role: Literal["system"]
    content: str


class StateMessage(TypedDict):
    """Message for local status."""

    role: Literal["state"]
    component: str
    state: int


Message = UserMessage | AssistantMessage | SystemMessage | StateMessage
"""A message from the user, assistant, or system."""

TextMessage = UserMessage | AssistantMessage | SystemMessage
"""A message from the user, assistant, or system that contains text."""


def is_message(message: Any) -> TypeGuard[Message]:
    """Check if a message is a message."""
    return isinstance(message, dict) and "role" in message


def is_text_message(message: Any) -> TypeGuard[TextMessage]:
    """Check if a message is a text message."""
    return is_message(message) and message["role"] in ("user", "assistant", "system")


def is_user_message(message: Any) -> TypeGuard[UserMessage]:
    """Check if a message is a user message."""
    return is_text_message(message) and message["role"] == "user"


def is_assistant_message(message: Any) -> TypeGuard[AssistantMessage]:
    """Check if a message is an assistant message."""
    return is_text_message(message) and message["role"] == "assistant"


def is_state_message(message: Any) -> TypeGuard[StateMessage]:
    """Check if a message is a state message."""
    return is_message(message) and message["role"] == "state"


def update_busy_time(message: Any, component: str, current_value: float) -> float:
    """Get the busy time of a message."""
    if not is_message(message):
        return current_value

    if message["role"] != "state":
        return current_value

    if message["component"] != component:
        return current_value

    return time.time() if ComponentState(message["state"]) == ComponentState.BUSY else 0


def is_busy(busy_time: float, timeout: float) -> bool:
    """Check if a component is busy."""
    return time.time() - busy_time < timeout
