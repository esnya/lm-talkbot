"""A module that allows async functions to be used as commands with argh."""

import asyncio
from typing import Any, Callable, Coroutine, List, TypeVar

from argh import ArghParser
from argh.assembling import ATTR_NAME, _extract_command_meta_from_func

T = TypeVar("T")


def wrap_async(async_func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Wrap an async function so that it can be used as a command."""

    def wrapper(*args, **kwargs: Any) -> T:
        return asyncio.run(async_func(*args, **kwargs))

    setattr(wrapper, "__wrapped__", async_func)

    cmd_name, _ = _extract_command_meta_from_func(async_func)
    setattr(wrapper, ATTR_NAME, cmd_name)
    setattr(wrapper, "__doc__", async_func.__doc__)

    return wrapper


class AsyncArghParser(ArghParser):
    """A subclass of ArghParser that allows async functions to be added as commands."""

    def add_async_commands(self, commands: List[Callable[..., Coroutine[Any, Any, Any]]], *args, **kwargs) -> None:
        """Add a list of async functions as commands."""
        self.add_commands([wrap_async(func) for func in commands], *args, **kwargs)

    def set_async_default_command(self, command: Callable[..., Coroutine[Any, Any, Any]]) -> None:
        """Set the default command to an async function."""
        self.set_default_command(wrap_async(command))
