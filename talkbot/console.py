"""Console for TalkBot."""

from transformers import AutoTokenizer

from .utilities.audio import list_device_info, list_host_api_info
from .utilities.config import Config
from .utilities.socket import get_write_socket
from .utilities.voicevox import speakers


async def console(config: Config = Config()):
    """Console component."""
    with get_write_socket(config) as socket:
        role = "user"
        while True:
            match (input(f"{role}> ")):
                case value if value in ["\\quit", "\\q"]:
                    break
                case value if value in ["\\user", "\\u"]:
                    role = "user"
                case value if value in ["\\system", "\\s"]:
                    role = "system"
                case value if value in ["\\api", "\\a"]:
                    print(list_host_api_info(), sep="\n")
                case value if value in ["\\output", "\\o"]:
                    print(
                        [info for info in list_device_info() if int(info["maxOutputChannels"]) > 0],
                        sep="\n",
                    )
                case value if value in ["\\input", "\\i"]:
                    print(
                        [info for info in list_device_info() if int(info["maxInputChannels"]) > 0],
                        sep="\n",
                    )
                case value if value in ["\\voice", "\\v"]:
                    print(await speakers())
                case value if value in ["\\t", "\\tokens"]:
                    tokenizer = AutoTokenizer.from_pretrained(
                        config.get("chat_engine.tokenizer", config.get("chat_engine.model")),
                    )
                    print(tokenizer.special_tokens_map)
                case value if value.startswith("\\e ") or value.startswith("\\encode "):
                    _, text = value.split(" ", 1)
                    print(
                        AutoTokenizer.from_pretrained(
                            config.get("chat_engine.tokenizer", config.get("chat_engine.model"))
                        ).encode(text, add_special_tokens=False)
                    )
                case value:
                    message = {
                        "role": role,
                        "content": value,
                    }
                    await socket.send_json(message)
