"""
Log Collector
=============
Watches one or more log files for new lines using a polling loop.
Handles file rotation (truncation / recreation) and encoding errors gracefully.
New lines are forwarded to a callback together with a rolling context window.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, List, Optional

logger = logging.getLogger(__name__)


# ── Single-file tracker ───────────────────────────────────────────────────────

class _FileTracker:
    """
    Tracks a single log file.

    - Starts reading from the *end* of the file so only new lines are reported.
    - Detects rotation / truncation by comparing stored vs current file size.
    - Maintains a rolling context window of recent lines.
    """

    def __init__(
        self,
        file_path: str,
        callback: Callable[[str, str, List[str], int], None],
        context_window: int = 5,
        encoding: str = "utf-8",
    ) -> None:
        self.file_path = Path(file_path)
        self.callback = callback
        self.context_window = context_window
        self.encoding = encoding

        # Rolling buffer of recent lines for context
        self._context: Deque[str] = deque(maxlen=context_window)

        # Seek to end of existing file so we only see new entries
        self._pos: int = self._current_size()

        # Count lines already in the file so new lines get accurate 1-based numbers
        self._line_number: int = 0
        if self.file_path.exists() and self._pos > 0:
            try:
                with open(self.file_path, "rb") as _fh:
                    self._line_number = _fh.read(self._pos).count(b"\n")
            except Exception:
                pass

    # ── Public ───────────────────────────────────────────────

    def check(self) -> None:
        """Read any new lines appended since the last check."""
        if not self.file_path.exists():
            return

        try:
            current_size = self._current_size()

            # File was rotated / truncated — reset position
            if current_size < self._pos:
                logger.info("Log file rotated: %s — resetting position", self.file_path)
                self._pos = 0
                self._line_number = 0

            if current_size == self._pos:
                return

            with open(self.file_path, "r", encoding=self.encoding, errors="replace") as fh:
                fh.seek(self._pos)
                chunk = fh.read()
                self._pos = fh.tell()

            for raw_line in chunk.splitlines():
                self._line_number += 1
                line = raw_line.strip()
                if not line:
                    continue
                context_snapshot = list(self._context) + [line]
                self._context.append(line)
                self.callback(str(self.file_path), line, context_snapshot, self._line_number)

        except PermissionError:
            logger.warning("Permission denied reading: %s", self.file_path)
        except Exception:
            logger.exception("Unexpected error reading: %s", self.file_path)

    # ── Private ──────────────────────────────────────────────

    def _current_size(self) -> int:
        try:
            return self.file_path.stat().st_size
        except OSError:
            return 0


# ── Multi-file collector ──────────────────────────────────────────────────────

class LogCollector:
    """
    Monitors multiple log files in a background thread using a simple poll loop.

    Args:
        config:   Full application config dict.
        callback: Called for every new log line:
                  callback(file_path: str, line: str, context: List[str])
    """

    def __init__(
        self,
        config: dict,
        callback: Callable[[str, str, List[str], int], None],
    ) -> None:
        self.callback = callback
        self._poll_interval: float = config.get("monitoring", {}).get("poll_interval", 2)
        self._context_lines: int  = config.get("monitoring", {}).get("context_lines", 5)

        self._trackers: List[_FileTracker] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        paths: List[str] = config.get("log_paths", [])
        for path in paths:
            self._trackers.append(
                _FileTracker(
                    file_path=path,
                    callback=callback,
                    context_window=self._context_lines,
                )
            )
            logger.info("Registered log file: %s", path)

        if not self._trackers:
            logger.warning("No log files registered — collector will idle")

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="LogCollector",
            daemon=True,
        )
        self._thread.start()
        logger.info("Log collector started (poll interval: %ss)", self._poll_interval)

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for the thread to exit."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval + 2)
        logger.info("Log collector stopped")

    # ── Internal poll loop ────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            for tracker in self._trackers:
                tracker.check()
            self._stop.wait(timeout=self._poll_interval)
