"""Export the CPU smoke-test subset with exact hashes from Pipfile.lock.

This deliberately does not install the incomplete legacy speech/model stack.
All selected names are public PyPI packages; no source builds are required.
"""
import json
from pathlib import Path

NAMES = """absl-py aiohappyeyeballs aiohttp aiosignal async-timeout attrs cachetools certifi
charset-normalizer colorama contourpy cycler datasets dill filelock fonttools frozenlist fsspec
gitdb gitpython google-auth google-auth-oauthlib grpcio huggingface-hub idna
kiwisolver mako markdown markupsafe matplotlib msgpack multidict multiprocess
numpy oauthlib packaging pandas pillow propcache protobuf pyarrow pyasn1
pyasn1-modules pyparsing python-dateutil pytz pyyaml requests requests-oauthlib
responses rsa setuptools six smmap tensorboard tensorboard-data-server
tensorboard-plugin-wit tqdm typing-extensions tzdata urllib3 werkzeug wheel
xxhash yarl
""".split()

if __name__ == "__main__":
    lock = json.loads((Path(__file__).resolve().parents[1] / "Pipfile.lock").read_text(encoding="utf-8"))
    packages = {**lock["default"], **lock["develop"]}
    for name in sorted(NAMES):
        entry = packages[name]
        assert entry["version"].startswith("==")
        print(name + entry["version"] + " " + " ".join("--hash=" + digest for digest in entry["hashes"]))
