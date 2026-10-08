"""HTTP helpers with bounded retries and deterministic request metadata."""
from __future__ import annotations

import time
from typing import Any

import httpx


class HTTPClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def get(self, path: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
        return self._request("GET", path, params=params, headers=headers)

    def post(self, path: str, *, json: dict[str, Any], headers: dict[str, str] | None = None) -> httpx.Response:
        return self._request("POST", path, json=json, headers=headers)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.request(method, url, timeout=self.timeout_s, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
        assert last_error is not None
        raise last_error
