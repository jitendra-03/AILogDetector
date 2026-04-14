"""
Screenshot Capture
==================
Before each screenshot, opens the monitored log file in Notepad++ and
navigates to the exact error line so the screen shows the relevant context.
Then takes a full-screen screenshot and saves it as a PNG/JPEG.

Requires Notepad++ to be installed.  The executable path is read from
config (screenshot.notepadpp_path) and defaults to the standard Windows
install location; a fallback PATH search is also attempted.

Uses pyautogui for capture.  Gracefully skipped on headless machines.
"""

import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Candidate install locations tried when no explicit path is configured
_NPP_SEARCH_PATHS = [
    r"C:\Program Files\Notepad++\notepad++.exe",
    r"C:\Program Files (x86)\Notepad++\notepad++.exe",
]


class ScreenshotCapture:
    """
    Captures and persists screenshots.

    Config keys consumed (under ``screenshot``):
        enabled   – bool, default True
        save_path – directory to save images (default: ./screenshots)
        format    – "PNG" or "JPEG" (default: PNG)
    """

    def __init__(self, config: dict) -> None:
        sc = config.get("screenshot", {})
        self.enabled:   bool = sc.get("enabled", True)
        self.save_path: Path = Path(sc.get("save_path", "./screenshots"))
        self.fmt:       str  = sc.get("format", "PNG").upper()
        # Seconds to wait after launching Notepad++ before taking the screenshot
        self.notepadpp_focus_delay: float = float(sc.get("notepadpp_focus_delay", 2.0))
        # Resolve Notepad++ executable path once at startup
        self.notepadpp_path: str = self._resolve_notepadpp(sc.get("notepadpp_path", ""))

        if self.enabled:
            self.save_path.mkdir(parents=True, exist_ok=True)
            logger.info("Screenshot capture enabled → %s", self.save_path)
            if self.notepadpp_path:
                logger.info("Notepad++ resolved at: %s", self.notepadpp_path)
            else:
                logger.warning(
                    "Notepad++ executable not found — log file will not be opened "
                    "before screenshots.  Set screenshot.notepadpp_path in config.yaml."
                )
        else:
            logger.info("Screenshot capture disabled")

    # ── Public API ────────────────────────────────────────────

    def capture(
        self,
        file_path: str = "",
        line_number: int = 0,
    ) -> Optional[str]:
        """
        Open *file_path* at *line_number* in Notepad++ (if available), wait
        for it to gain focus, then take a full-screen screenshot.

        Returns the absolute path of the saved file, or None on failure.
        """
        if not self.enabled:
            return None

        # Open the log file in Notepad++ at the error line before shooting
        if file_path and line_number and self.notepadpp_path:
            self._open_in_notepadpp(file_path, line_number)

        try:
            import pyautogui
        except ImportError:
            logger.warning("pyautogui not installed — screenshot skipped")
            return None

        try:
            img = pyautogui.screenshot()
            path = self._unique_path()
            img.save(str(path), format=self.fmt)
            logger.info("Screenshot saved: %s", path)
            return str(path)

        except Exception as exc:
            # Common in headless / remote-desktop scenarios
            logger.warning("Screenshot capture failed: %s", exc)
            return None

    # ── Helpers ───────────────────────────────────────────────

    def _open_in_notepadpp(self, file_path: str, line_number: int) -> None:
        """
        Launch (or reuse) Notepad++ with *file_path* scrolled to *line_number*,
        then sleep long enough for the window to appear before the screenshot.

        Notepad++ CLI:  notepad++.exe "<path>" -n<line>
        If N++ is already open it brings the existing instance to the front
        and opens the file in a new tab.
        """
        try:
            subprocess.Popen(
                [self.notepadpp_path, str(file_path), f"-n{line_number}"],
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )
            logger.info(
                "Opened %s at line %d in Notepad++ — waiting %.1fs for focus",
                Path(file_path).name, line_number, self.notepadpp_focus_delay,
            )
            time.sleep(self.notepadpp_focus_delay)
        except FileNotFoundError:
            logger.warning(
                "Notepad++ not found at %r — skipping pre-screenshot open",
                self.notepadpp_path,
            )
        except Exception as exc:
            logger.warning("Failed to open Notepad++: %s", exc)

    @staticmethod
    def _resolve_notepadpp(configured: str) -> str:
        """Return the best available path to the Notepad++ executable."""
        if configured and Path(configured).exists():
            return configured
        for candidate in _NPP_SEARCH_PATHS:
            if Path(candidate).exists():
                return candidate
        found = shutil.which("notepad++")
        return found or ""

    def _unique_path(self) -> Path:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        ext = "jpg" if self.fmt == "JPEG" else "png"
        return self.save_path / f"error_{ts}.{ext}"
