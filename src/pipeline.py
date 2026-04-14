"""
Pipeline Orchestrator
=====================
Wires all components together and drives the end-to-end flow:

  Log Collector → Error Detector → Embedding Generator
  → Vector DB (Knowledge Base) → RAG + LLM → Screenshot → Teams Alert
"""

import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from rich.console import Console

from .log_collector import LogCollector
from .error_detector import ErrorDetector, DetectedError
from .embedding_generator import EmbeddingGenerator
from .knowledge_base_scraper import KnowledgeBaseScraper
from .vector_db import VectorDB
from .rag_engine import RAGEngine
from .screenshot_capture import ScreenshotCapture
from .teams_alert import TeamsAlert
from .email_alert import EmailAlert

logger = logging.getLogger(__name__)
console = Console()

_SEVERITY_STYLE = {
    "CRITICAL": "bold red",
    "ERROR":    "red",
    "WARNING":  "yellow",
}


class Pipeline:
    """Orchestrates the full AI log-error detection pipeline."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._running = False

        console.print("\n[bold]Initialising pipeline components…[/bold]")

        # ── Instantiate each stage ────────────────────────────
        console.print("  [cyan]1/7[/cyan]  Embedding Generator")
        self.embedding_gen = EmbeddingGenerator(config)

        console.print("  [cyan]2/7[/cyan]  Vector Database (Knowledge Base)")
        self.vector_db = VectorDB(config)

        console.print("  [cyan]3/8[/cyan]  Serena Knowledge Base Scraper")
        self.kb_scraper = KnowledgeBaseScraper(config)

        console.print("  [cyan]4/8[/cyan]  RAG + LLM Engine")
        self.rag_engine = RAGEngine(config, self.vector_db, self.embedding_gen, self.kb_scraper)

        console.print("  [cyan]5/8[/cyan]  Error Detector")
        self.error_detector = ErrorDetector(config)

        console.print("  [cyan]6/8[/cyan]  Screenshot Capture")
        self.screenshot = ScreenshotCapture(config)

        console.print("  [cyan]7/8[/cyan]  Teams Alert")
        self.teams = TeamsAlert(config)

        console.print("  [cyan]8/8[/cyan]  Email Alert")
        self.email = EmailAlert(config)

        console.print("  [cyan]8/8[/cyan]  Log Collector")
        self.log_collector = LogCollector(config, self._on_log_line)

        # ── Seed knowledge base on first start ────────────────
        self.seed_knowledge_base(force=False)

        # Permanent per-session cache: md5(message) → solution
        # Prevents re-running LLM + Teams for the same error text
        self._solution_cache: dict = {}

        # Burst buffer: accumulates consecutive errors until an INFO line
        # or a 3-second idle timer flushes them as a single grouped alert
        self._burst_buffer: List[DetectedError] = []
        self._burst_lock   = threading.Lock()
        self._burst_timer: Optional[threading.Timer] = None
        self._burst_timeout: float = float(
            config.get("monitoring", {}).get("burst_timeout", 3.0)
        )
        # After each error line, absorb this many subsequent non-error lines as
        # continuation text before allowing a burst flush.  This captures
        # multi-line error blocks (stack traces, script error detail lines, etc.)
        self._continuation_lines: int = int(
            config.get("monitoring", {}).get("continuation_lines", 3)
        )
        self._continuation_remaining: int = 0

        console.print("[green]Pipeline ready.[/green]\n")

    # ── Knowledge base seeding ────────────────────────────────────────────────

    def seed_knowledge_base(self, force: bool = False) -> None:
        """Seed ChromaDB from knowledge_base/seed.yaml (skipped if already seeded)."""
        seed_path = Path("knowledge_base/seed.yaml")
        if not seed_path.exists():
            return
        count = self.vector_db.seed_from_yaml(str(seed_path), self.embedding_gen, force=force)
        if count > 0:
            console.print(f"  [dim]Knowledge base seeded with {count} entries[/dim]")

    def scrape_knowledge_base(self, query: str) -> int:
        """Scrape Serena KB for a query, cache pages locally, and index them in ChromaDB."""
        documents = self.kb_scraper.search_and_cache(query)
        if not documents:
            console.print("  [yellow]No Serena KB documents were discovered for the query.[/yellow]")
            return 0
        count = self.vector_db.add_knowledge_documents(documents, self.embedding_gen)
        console.print(f"  [dim]Scraped and indexed {count} Serena KB document(s)[/dim]")
        return count

    # ── Log line callback ─────────────────────────────────────────────────────

    def _on_log_line(self, file_path: str, line: str, context: List[str], line_number: int = 0) -> None:
        """Invoked by LogCollector for every new line in a monitored file."""
        error = self.error_detector.detect(line, context, file_path)

        if error is not None:
            # Error line — buffer it, reset the idle flush timer, and open a
            # continuation window so the next N non-error lines are absorbed
            # into this error's message (multi-line error block support).
            style = _SEVERITY_STYLE.get(error.severity, "white")
            ts = datetime.now().strftime("%H:%M:%S")
            fname = Path(file_path).name
            console.print(
                f"[dim]{ts}[/dim]  [{style}]{error.severity:<8}[/{style}]  "
                f"[dim]{fname}[/dim]  {error.message[:120]}"
            )
            error.line_number = line_number
            with self._burst_lock:
                self._burst_buffer.append(error)
                self._continuation_remaining = self._continuation_lines
                self._reset_burst_timer()
        else:
            # Non-error line: while inside the continuation window, append it
            # to the most-recent error's message so multi-line error blocks
            # (User info, script context, ERR codes, etc.) travel together.
            # Once the window is exhausted and no new error arrives, flush.
            absorbed = False
            with self._burst_lock:
                if self._burst_buffer and self._continuation_remaining > 0:
                    self._burst_buffer[-1].message += "\n" + line
                    self._burst_buffer[-1].context.append(line)
                    self._continuation_remaining -= 1
                    self._reset_burst_timer()
                    absorbed = True
            if not absorbed:
                self._flush_burst()

    # ── Burst-grouping helpers ──────────────────────────────────────────

    def _reset_burst_timer(self) -> None:
        """(Re)start the idle-flush timer. Must be called with _burst_lock held."""
        if self._burst_timer is not None:
            self._burst_timer.cancel()
        self._burst_timer = threading.Timer(self._burst_timeout, self._flush_burst)
        self._burst_timer.daemon = True
        self._burst_timer.start()

    def _flush_burst(self) -> None:
        """Flush the burst buffer as a single grouped alert."""
        with self._burst_lock:
            if self._burst_timer is not None:
                self._burst_timer.cancel()
                self._burst_timer = None
            batch = self._burst_buffer[:]
            self._burst_buffer.clear()
            self._continuation_remaining = 0  # reset so next error starts fresh

        if not batch:
            return

        if len(batch) == 1:
            self._process_error(batch[0])
        else:
            self._process_error_batch(batch)

    # ── Full error processing ─────────────────────────────────────────────────

    def _process_error_batch(self, errors: List[DetectedError]) -> None:
        """Process a burst of consecutive errors as a single grouped alert."""
        # De-duplicate against session cache — skip any already resolved
        new_errors = []
        for e in errors:
            h = hashlib.md5(e.message.encode("utf-8")).hexdigest()[:16]
            if h not in self._solution_cache:
                new_errors.append(e)
            else:
                console.print("  [dim]↩  Duplicate (in batch) — skipping[/dim]")

        if not new_errors:
            return

        if len(new_errors) == 1:
            self._process_error(new_errors[0])
            return

        lead = max(new_errors, key=lambda e: {"CRITICAL": 3, "ERROR": 2, "WARNING": 1}.get(e.severity, 0))
        console.print(
            f"  [bold yellow]Burst of {len(new_errors)} errors — sending grouped alert[/bold yellow]"
        )
        try:
            solution, kb_matches = self.rag_engine.analyze_batch(new_errors)

            for e in new_errors:
                sol_id = self.vector_db.add_error(e, solution, self.embedding_gen)
                logger.info("Solution stored | solution_id=%s", sol_id)

            screenshot_path: Optional[str] = None
            if self.config.get("screenshot", {}).get("enabled", True):
                screenshot_path = self.screenshot.capture(
                    file_path=lead.file_path,
                    line_number=lead.line_number,
                )

            self.teams.send_batch_alert(new_errors, solution, screenshot_path, kb_matches=kb_matches)
            self.email.send_batch_alert(new_errors, solution, screenshot_path, kb_matches=kb_matches)

            for e in new_errors:
                h = hashlib.md5(e.message.encode("utf-8")).hexdigest()[:16]
                self._solution_cache[h] = solution

            console.print(
                f"  [green]✓ Grouped alert sent ({len(new_errors)} errors clubbed into 1)[/green]"
            )
        except Exception:
            logger.exception("Pipeline error while processing batch of %d errors", len(new_errors))

    def _process_error(self, error: DetectedError) -> None:
        """Run an error through the remaining pipeline stages."""
        import hashlib
        msg_hash = hashlib.md5(error.message.encode("utf-8")).hexdigest()[:16]

        if msg_hash in self._solution_cache:
            logger.info("Duplicate error suppressed (cached): %s", error.message[:80])
            console.print("  [dim]↩  Duplicate — solution already cached, skipping LLM + Teams[/dim]")
            return

        try:
            # Stage 1 — RAG + LLM analysis
            logger.info("RAG analysis: %s", error.message[:80])
            solution, kb_matches = self.rag_engine.analyze(error)

            # Stage 2 — Store error + solution in knowledge base
            solution_id = self.vector_db.add_error(error, solution, self.embedding_gen)
            logger.info("Solution stored | solution_id=%s", solution_id)

            # Stage 3 — Screenshot
            screenshot_path: Optional[str] = None
            if self.config.get("screenshot", {}).get("enabled", True):
                screenshot_path = self.screenshot.capture(
                    file_path=error.file_path,
                    line_number=error.line_number,
                )

            # Stage 4 — Teams alert
            self.teams.send_alert(error, solution, screenshot_path, kb_matches=kb_matches)
            self.email.send_alert(error, solution, screenshot_path, kb_matches=kb_matches)

            # Cache so the same error message never re-triggers LLM + Teams
            self._solution_cache[msg_hash] = solution

            console.print("  [green]✓ Solution generated & Teams alert sent[/green]")

        except Exception:
            logger.exception("Pipeline error while processing: %s", error.message[:80])

    # ── Self-test ─────────────────────────────────────────────────────────────

    def run_test(self) -> None:
        """Process a synthetic error line to verify the full pipeline."""
        console.print("\n[yellow]Running pipeline self-test…[/yellow]")
        test_line = (
            "2026-03-31 10:00:00 ERROR Failed to connect to database: "
            "Connection refused (host=localhost port=5432)"
        )
        context = [
            "2026-03-31 09:59:57 INFO  Starting application",
            "2026-03-31 09:59:58 INFO  Attempting DB connection (attempt 1/3)…",
            "2026-03-31 09:59:59 INFO  Attempting DB connection (attempt 2/3)…",
            test_line,
        ]
        console.print(f"  Test line: [dim]{test_line}[/dim]")
        self._on_log_line("test.log", test_line, context)
        # Send a non-error line to flush the burst buffer synchronously
        self._on_log_line("test.log", "2026-03-31 10:00:01 INFO  Retrying...", context)
        console.print("[green]Self-test complete.[/green]")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self.log_collector.start()
        logger.info("Pipeline started")

    def stop(self) -> None:
        self._running = False
        self.log_collector.stop()
        logger.info("Pipeline stopped")

    def is_running(self) -> bool:
        return self._running
