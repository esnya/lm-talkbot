"""VoiceVox Core API wrapper."""

import asyncio
import requests

BASE_URL = "http://localhost:50021"


async def voicevox_post(
    path: str,
    params: dict | None = None,
    data: bytes | None = None,
    json: dict | None = None,
    timeout: int | None = None,
):
    """POST request to VoiceVox Core."""
    return await asyncio.to_thread(
        requests.post, f"{BASE_URL}{path}", params=params, data=data, json=json, timeout=timeout
    )


async def voicevox_get(path: str, params: dict | None = None, timeout: int | None = None):
    """GET request to VoiceVox Core."""
    return await asyncio.to_thread(requests.get, f"{BASE_URL}{path}", params=params, timeout=timeout)


async def version() -> str:
    """Get VoiceVox Core version."""
    return (await voicevox_get("/version")).json()


async def engine_versions() -> str:
    """Get VoiceVox Core engine versions."""
    return (await voicevox_get("/engine_versions")).json()


async def speakers() -> dict:
    """Get VoiceVox Core speakers."""
    return (await voicevox_get("/speakers")).json()


async def audio_query(text: str, speaker: int = 1) -> dict:
    """Get VoiceVox Core audio query."""
    return (await voicevox_post("/audio_query", params={"text": text, "speaker": speaker})).json()


async def synthesis(query: dict, speaker: int = 1) -> bytes:
    """Get VoiceVox Core synthesis."""
    return (await voicevox_post("/synthesis", params={"speaker": speaker}, json=query)).content
