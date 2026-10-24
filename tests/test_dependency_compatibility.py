"""Exercise changed dependency APIs without models, audio devices or services."""
import asyncio
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


class DependencyCompatibilityTests(unittest.TestCase):
    def test_voicevox_requests(self):
        from talkbot.utilities import voicevox

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'"test-engine"')

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.end_headers()
                if self.path.startswith("/audio_query?"):
                    self.wfile.write(b'{"accent_phrases": [], "speedScale": 1.0}')
                else:
                    assert json.loads(body)["speedScale"] == 1.0
                    self.wfile.write(b"RIFF-test-audio")

            def log_message(self, *_):
                pass

        async def requests():
            import aiohttp

            version = await voicevox.voicevox_get("/version", timeout=5)
            self.assertEqual(version.json(), "test-engine")
            query = await voicevox.voicevox_post("/audio_query", params={"text": "test text", "speaker": 1}, timeout=5)
            audio = await voicevox.voicevox_post("/synthesis", json=query.json(), timeout=5)
            self.assertEqual(audio.content, b"RIFF-test-audio")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(voicevox.BASE_URL + "/version") as response:
                    self.assertEqual(await response.json(content_type=None), "test-engine")

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.object(voicevox, "BASE_URL", f"http://127.0.0.1:{server.server_port}"):
                    asyncio.run(requests())
            finally:
                server.shutdown()
                thread.join()

    def test_training_dataset_and_arrow(self):
        import numpy as np
        import pyarrow as pa
        from datasets import Dataset

        dataset = Dataset.from_dict({"input_ids": np.array([[1, 2], [3, 4]]), "attention_mask": [[1, 1], [1, 1]]})
        self.assertEqual(dataset[1]["input_ids"], [3, 4])
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, dataset.data.schema) as writer:
            writer.write_table(dataset.data.table)
        restored = pa.ipc.open_stream(sink.getvalue()).read_all()
        self.assertEqual(restored.to_pydict()["input_ids"], [[1, 2], [3, 4]])

    def test_pillow_matplotlib_and_fonttools(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from fontTools.ttLib import TTFont
        from PIL import Image

        for format in ("PNG", "JPEG", "TIFF"):
            data = io.BytesIO()
            Image.new("RGB", (8, 8), (10, 20, 30)).save(data, format=format)
            data.seek(0)
            with Image.open(data) as image:
                image.load()
                self.assertEqual(image.size, (8, 8))
        figure, axes = plt.subplots()
        axes.plot([0, 1], [1, 0])
        output = io.BytesIO()
        figure.savefig(output, format="png")
        plt.close(figure)
        self.assertTrue(output.getvalue().startswith(b"\x89PNG"))
        with TTFont(matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf") as font:
            self.assertIn("A", font.getBestCmap().values())

    def test_tensorboard_protobuf_and_grpc(self):
        import grpc
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        event = Event(step=1, summary=Summary(value=[Summary.Value(tag="loss", simple_value=0.5)]))
        self.assertEqual(Event.FromString(event.SerializeToString()).summary.value[0].simple_value, 0.5)
        with grpc.insecure_channel("127.0.0.1:1") as channel:
            self.assertIsNotNone(channel.unary_unary("/test/unused"))

    def test_storage_and_development_helpers(self):
        import fsspec
        import msgpack
        from git import Repo
        from mako.template import Template
        from pyasn1.codec.der import decoder, encoder
        from pyasn1.type.univ import ObjectIdentifier
        from werkzeug.test import Client
        from werkzeug.wrappers import Response
        from wheel.wheelfile import WheelFile

        self.assertEqual(msgpack.unpackb(msgpack.packb({"text": "sample"})), {"text": "sample"})
        self.assertEqual(Template("${value}").render(value="sample"), "sample")
        oid = ObjectIdentifier("1.2.840.113549")
        self.assertEqual(decoder.decode(encoder.encode(oid))[0], oid)
        self.assertEqual(Client(Response("ready")).get("/").text, "ready")
        self.assertIsNotNone(WheelFile)
        with tempfile.TemporaryDirectory() as directory:
            repo = Repo.init(directory)
            self.assertTrue(repo.bare is False)
            with fsspec.open(str(Path(directory) / "sample.txt"), "w") as handle:
                handle.write("sample")
            with fsspec.open(str(Path(directory) / "sample.txt")) as handle:
                self.assertEqual(handle.read(), b"sample")


if __name__ == "__main__":
    unittest.main()
