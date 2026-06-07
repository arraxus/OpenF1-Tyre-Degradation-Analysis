import time
import threading
from collections import deque

class RateLimiter:
    """
    Pilnuje dwóch limitów:
      - max 3 req/s  (okno 1 s)
      - max 30 req/min (okno 60 s)
    """
    def __init__(self, per_second: int = 3, per_minute: int = 30):
        self.per_second = per_second
        self.per_minute = per_minute
        self._lock      = threading.Lock()
        self._history   = deque()  # timestamps wszystkich requestów

    def wait(self):
        with self._lock:
            now = time.monotonic()

            # Wyczyść wpisy starsze niż 60 s
            while self._history and self._history[0] < now - 60:
                self._history.popleft()

            # Sprawdź limit minutowy
            if len(self._history) >= self.per_minute:
                oldest_in_minute = self._history[0]
                wait_min = (oldest_in_minute + 60) - now
                if wait_min > 0:
                    time.sleep(wait_min)
                    now = time.monotonic()
                    while self._history and self._history[0] < now - 60:
                        self._history.popleft()

            # Sprawdź limit sekundowy (ostatnie 3 wpisy w oknie 1 s)
            recent = [t for t in self._history if t >= now - 1.0]
            if len(recent) >= self.per_second:
                oldest_recent = min(recent)
                wait_sec = (oldest_recent + 1.0) - now
                if wait_sec > 0:
                    time.sleep(wait_sec)
                    now = time.monotonic()

            self._history.append(time.monotonic())