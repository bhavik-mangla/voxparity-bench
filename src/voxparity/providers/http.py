"""Shared HTTP retry helper for all providers.

Retries transport failures (connect timeouts, resets — e.g. when large model
downloads saturate the link) and rate-limit/server statuses, with exponential
backoff. Raises the last transport error or returns the first definitive
response; 4xx other than 429 is definitive (caller decides what it means).
"""

from __future__ import annotations

import time
from typing import Any

import httpx


def post_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    content: bytes | None = None,
    timeout: float = 120.0,
    tries: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
) -> httpx.Response:
    last_exc: httpx.TransportError | None = None
    for attempt in range(tries):
        try:
            resp = httpx.post(
                url, json=json_body, content=content, headers=headers, timeout=timeout
            )
        except httpx.TransportError as e:
            last_exc = e
            time.sleep(min(base_delay * 2**attempt, max_delay))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(min(base_delay * 2**attempt, max_delay))
            continue
        return resp
    if last_exc is not None:
        raise last_exc
    raise TimeoutError(f"still rate-limited/5xx after {tries} attempts: {url}")
