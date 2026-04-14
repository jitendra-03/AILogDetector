"""
Error Detector
==============
Classifies log lines by severity using configurable regex patterns.
Applies deduplication so the same error message is not re-processed
within a configurable time window.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Strip leading log timestamps (e.g. "2026-04-01 09:05:01,123 ERROR " or "[09:05:01] ERROR ")
_TS_RE = re.compile(
    r'^[\[\(]?\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[.,]?\d*[\]\)]?\s*'
    r'|^[\[\(]?\d{2}:\d{2}:\d{2}[.,]?\d*[\]\)]?\s*',
    re.IGNORECASE,
)

# Higher number = higher severity
_SEVERITY_RANK: Dict[str, int] = {
    "CRITICAL": 3,
    "ERROR":    2,
    "WARNING":  1,
}


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class DetectedError:
    """A confirmed error extracted from a log file."""
    file_path:   str
    message:     str
    severity:    str
    timestamp:   datetime
    context:     List[str]  = field(default_factory=list)
    line_number: int        = 0   # 1-based line number in the source file


# ── Detector ─────────────────────────────────────────────────────────────────

class ErrorDetector:
    """
    Detects errors in log lines.

    Config keys consumed (under root):
      error_patterns  – list of {level, patterns[]} dicts
      min_severity    – minimum level to keep (default: ERROR)
      monitoring.dedup_window – seconds to suppress duplicate alerts (default: 60)
    """

    def __init__(self, config: dict) -> None:
        self._min_rank: int = _SEVERITY_RANK.get(
            config.get("min_severity", "ERROR").upper(), 2
        )
        self._dedup_secs: float = float(
            config.get("monitoring", {}).get("dedup_window", 60)
        )

        # Compiled pattern groups sorted highest-severity first
        self._groups: List[Tuple[str, List[re.Pattern]]] = []
        for group in config.get("error_patterns", []):
            level = group.get("level", "ERROR").upper()
            compiled: List[re.Pattern] = []
            for raw in group.get("patterns", []):
                try:
                    compiled.append(re.compile(raw, re.IGNORECASE))
                except re.error as exc:
                    logger.warning("Invalid regex pattern %r: %s", raw, exc)
            if compiled:
                self._groups.append((level, compiled))

        # Sort groups so we test highest severity first
        self._groups.sort(key=lambda g: _SEVERITY_RANK.get(g[0], 0), reverse=True)

        # Deduplication: {key: last_seen_timestamp}
        self._seen: Dict[str, float] = {}

        total_patterns = sum(len(c) for _, c in self._groups)
        logger.info(
            "Error detector ready | %d pattern(s) across %d severity group(s) | min: %s",
            total_patterns, len(self._groups),
            config.get("min_severity", "ERROR"),
        )

    # ── Public API ────────────────────────────────────────────

    def detect(
        self,
        line: str,
        context: List[str],
        file_path: str,
    ) -> Optional[DetectedError]:
        """
        Evaluate a log line.

        Returns a DetectedError if the line matches a pattern at or above
        min_severity and has not been seen recently; otherwise None.
        """
        severity = self._classify(line)
        if severity is None:
            return None

        if _SEVERITY_RANK.get(severity, 0) < self._min_rank:
            return None

        if self._is_duplicate(file_path, line):
            return None

        return DetectedError(
            file_path=file_path,
            message=line,
            severity=severity,
            timestamp=datetime.now(),
            context=context,
        )

    # ── Private helpers ───────────────────────────────────────

    def _classify(self, line: str) -> Optional[str]:
        """Return the highest-severity level matched, or None."""
        for level, patterns in self._groups:
            for pattern in patterns:
                if pattern.search(line):
                    return level
        return None

    def _is_duplicate(self, file_path: str, line: str) -> bool:
        """Return True if an identical error (ignoring timestamp) was seen within the dedup window."""
        normalized = _TS_RE.sub('', line).strip()
        key = f"{file_path}:{normalized[:120]}"
        now = time.monotonic()

        if key in self._seen and (now - self._seen[key]) < self._dedup_secs:
            return True

        self._seen[key] = now

        # Prune stale entries to keep memory bounded
        if len(self._seen) > 2000:
            cutoff = now - self._dedup_secs
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

        return False
