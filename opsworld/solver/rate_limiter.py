"""
rate_limiter.py — thread-safe limiter shared across all collector workers so we
never exceed the Azure GPT-5.4 requests-per-minute (and optionally concurrency).

    rl = RateLimiter(rpm=120, max_concurrent=8)
    with rl:
        resp = llm.chat_json(...)
"""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, rpm: int = 120, max_concurrent: int | None = None):
        self.rpm = max(1, rpm)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_concurrent) if max_concurrent else None

    def _throttle(self):
        while True:
            with self._lock:
                now = time.time()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0]) + 0.01
            time.sleep(min(wait, 5.0))

    def __enter__(self):
        if self._sem:
            self._sem.acquire()
        self._throttle()
        return self

    def __exit__(self, *exc):
        if self._sem:
            self._sem.release()
        return False
