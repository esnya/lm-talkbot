"""Chat Engine component."""

import asyncio
import json
import math
import os
import random
import re
import time
from datetime import date
from functools import lru_cache
from typing import Any

import requests
import torch
import zmq
from transformers import AutoModelForCausalLM, T5Tokenizer

from .utilities.config import Config
from .utilities.constants import ComponentState
from .utilities.message import (
    AssistantMessage,
    TextMessage,
    UserMessage,
    is_text_message,
    is_user_message,
    update_busy_time,
)
from .utilities.socket import get_sockets, send_state


def is_valid_message(message) -> bool:
    """Check if a message is valid."""
    return set(message.keys()) == {"role", "content"}


@lru_cache(maxsize=1)
def load_model(
    model_name: str, tokenizer_name: str, device: str, fp16: bool
) -> tuple[T5Tokenizer, AutoModelForCausalLM]:
    """Load model."""
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(  # type: ignore
        tokenizer_name or model_name,
        use_fast=False,
    )
    tokenizer.do_lower_case = True  # type: ignore

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # type: ignore
    if fp16:
        model = model.half()
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (tokenizer, model)


async def chat_engine(config: Config = Config()):
    """Chat Engine component."""
    logger = config.get_logger("ChatEngine")
    logger.info("Initializing")

    with get_sockets(
        config,
        1,
    ) as (write_socket, read_socket):
        model_name = config.get("chat_engine.model")
        tokenizer_name = config.get("chat_engine.tokenizer", model_name)
        device = config.get("chat_engine.device", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model: %s (%s)", model_name, device)
        load_model(model_name, tokenizer_name, device, config.get("chat_engine.fp16", False))

        def _get_history_filename() -> str:
            """Get the history filename."""
            model_name = config.get("chat_engine.model", "history")
            pattern = r"checkpoint.*|-pruned.*"
            model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name).strip("_")
            model_name = re.sub(pattern, "", model_name)
            default = f"history/{model_name}.jsonl" if model_name else "history.jsonl"
            return config.get("chat_engine.history.file", default)

        def _parse_history_line(line: str) -> TextMessage | None:
            """Parse a line from the history file."""
            try:
                message = json.loads(line)
                if not is_text_message(message):
                    return None
                return message
            except json.JSONDecodeError:
                return None

        _history: list[TextMessage] = []

        def _get_chat_history() -> list[TextMessage]:
            """Get the chat history."""
            nonlocal _history
            if not _history:
                if os.path.exists(_get_history_filename()):
                    with open(_get_history_filename(), "r", encoding="utf-8") as file:
                        _history = [message for message in [_parse_history_line(line) for line in file] if message]
                else:
                    _history = []

            return _history

        def _append_history(message: TextMessage) -> None:
            """Append a message to the history file."""
            nonlocal _history
            if not is_valid_message(message):
                raise TypeError(f"Invalid message: {message}")

            _history.append(message)
            with open(_get_history_filename(), "a+", encoding="utf-8") as file:
                file.write(json.dumps(message, ensure_ascii=False))
                file.write("\n")

        def _format_messages(messages: list[TextMessage]) -> list[str]:
            """Format messages for the model."""
            return [
                config.get(f"chat_engine.message_formats.{message['role']}", "{}").format(message["content"])
                for message in messages
                if message["role"] != "system"
            ]

        @torch.no_grad()
        def _generate(
            received_message: list[TextMessage],
            history: list[TextMessage],
        ) -> str | None:
            """Generate a response from the model."""
            tokenizer_name = config.get("chat_engine.tokenizer", model_name)
            device = config.get(
                "chat_engine.device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
            logger.debug("Loading model: %s (%s)", model_name, device)
            (tokenizer, model) = load_model(model_name, tokenizer_name, device, config.get("chat_engine.fp16", False))

            input_text = config.get("chat_engine.message_separator", "").join(
                _format_messages(history + received_message) + config.get("chat_engine.suffix_messages", [])
            )

            logger.info("Generating from: %s", input_text)

            logger.debug("Encoding")
            input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(  # type: ignore
                model.device  # type: ignore
            )
            logger.debug("Generating: input={%s}", tokenizer.batch_decode(input_ids))
            generation_config: dict[str, Any] = config.get(
                "chat_engine.generation_config",
                dict[str, Any](),
            )
            output_ids = model.generate(  # type: ignore
                input_ids,
                **generation_config,
            )
            loss = model(output_ids, labels=output_ids).loss.item()  # type: ignore
            logger.info("Loss: %s", loss)
            if math.isnan(loss) or loss > config.get("chat_engine.max_loss", 10):
                return None

            output_token_count = len(output_ids[0]) - len(input_ids[0])
            logger.debug("Decoding %s tokens: %s", output_token_count, tokenizer.batch_decode(output_ids))
            output_text = tokenizer.decode(output_ids[0][len(input_ids[0]) :])

            logger.debug("Formatting")
            end_match = re.search(config.get("chat_engine.stop_pattern", "</s>"), output_text)
            logger.debug(end_match)
            content = output_text[: end_match.start()] if end_match else output_text

            max_length = config.get("chat_engine.content_max_length")
            if max_length and len(content) > max_length:
                return None

            return content

        async def _process_command(message: AssistantMessage) -> str | None:
            """Process a command."""
            try:
                if not message["role"] == "assistant":
                    return

                match = re.search(r"(?<=<req>) *[A-Z]+", message["content"])
                if not match:
                    return None
                command = match.group(0).strip()

                timeout = config.get("chat_engine.command.http_timeout")

                match (command):
                    case "DATE":
                        return f"{date.today()}"
                    case "WEATHER":
                        res = requests.get(
                            "https://weather.tsukumijima.net/api/forecast?city=130010",
                            timeout=timeout,
                        ).json()
                        weather = (
                            res["forecasts"][0]["dateLabel"] + "の" + res["title"] + "は" + res["forecasts"][0]["telop"]
                        )
                        await _add_assistant_message(weather)
                        return weather
                    case "NEWS":
                        rss = requests.get(
                            "https://www.nhk.or.jp/rss/news/cat0.xml",
                            timeout=timeout,
                        ).text
                        titles = re.findall(r"(?<=<title>).*(?=</title>)", rss)[1:]
                        random.shuffle(titles)
                        title = titles[0]
                        await _add_assistant_message(title + " （NHKニュース）")
                        return title
                return None
            except requests.HTTPError as err:
                logger.error(err)
                return None

        async def _add_assistant_message(content: str) -> AssistantMessage:
            assistant_message_json = AssistantMessage(
                role="assistant",
                content=content,
            )
            _append_history(assistant_message_json)
            logger.info(assistant_message_json)
            await write_socket.send_json(assistant_message_json)
            return assistant_message_json

        async def _process_messages(messages: list[TextMessage]) -> AssistantMessage | None:
            max_history_count = config.get("chat_engine.history.max_count") or 0
            history = _get_chat_history()[-max_history_count:] if max_history_count else []

            for message in messages:
                logger.info("Message: %s", message)
                _append_history(message)

            for _ in range(0, 10):
                logger.info("Generating")

                content = await asyncio.to_thread(
                    _generate,
                    messages,
                    history,
                )
                if content:
                    return await _add_assistant_message(content)

                logger.warning("Failed to generate")

            return None

        prev_time = time.time()
        message_buffer: list[TextMessage] = []
        last_stt_busy_time = 0
        last_tts_busy_time = 0
        logger.info("Started")
        await send_state(write_socket, "ChatEngine", ComponentState.READY)
        while not write_socket.closed and not read_socket.closed:
            while True:
                try:
                    message = await read_socket.recv_json()
                    if is_user_message(message):
                        message_buffer.append(message)
                    else:
                        last_stt_busy_time = update_busy_time(message, "AudioToMessage", last_stt_busy_time)
                        last_tts_busy_time = update_busy_time(message, "MessageToSpeak", last_tts_busy_time)
                except zmq.error.Again:
                    break

            current_time = time.time()

            if (
                message_buffer
                and current_time - prev_time >= config.get("chat_engine.min_interval", 30)
                and time.time() - last_stt_busy_time > config.get("global.busy_timeout", 30)
                and time.time() - last_tts_busy_time > config.get("global.busy_timeout", 30)
            ):
                await send_state(write_socket, "ChatEngine", ComponentState.BUSY)

                prev_time = current_time

                assistant_message = await _process_messages(message_buffer)
                message_buffer = []

                if not assistant_message:
                    continue

                command_result = await _process_command(assistant_message)
                if command_result:
                    await _process_messages([UserMessage(role="user", content="<res>" + command_result)])

                await send_state(write_socket, "ChatEngine", ComponentState.READY)

                await asyncio.sleep(config.get("chat_engine.sleep_after_completion", 0))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                await asyncio.sleep(1)
