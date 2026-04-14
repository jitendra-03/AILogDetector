#!/usr/bin/env python3
"""
AI Log Error Detector
=====================
Entry point — loads configuration, validates settings, and starts the pipeline.

Pipeline:
  Log Collector → Error Detector → Embedding Generator
  → Vector DB → RAG + LLM → Screenshot Capture → Teams Alert
"""

import sys
import os
import signal
import logging
import argparse
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Load .env before anything else
load_dotenv()

console = Console()


# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO") -> None:
    Path("logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.FileHandler("logs/ai_detector.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpcore", "httpx", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Allow .env to override webhook URL
    webhook = os.getenv("TEAMS_WEBHOOK_URL", "").strip()
    if webhook:
        config.setdefault("teams", {})["webhook_url"] = webhook

    # Allow .env to influence LLM provider
    for key, cfg_key in (("LLM_PROVIDER", "provider"), ("LLM_MODEL", "model")):
        val = os.getenv(key, "").strip()
        if val:
            config.setdefault("llm", {})[cfg_key] = val

    return config


def validate_config(config: dict, require_llm: bool = True) -> bool:
    warnings, errors = [], []

    if not config.get("log_paths"):
        errors.append("No log_paths defined in config.yaml")

    provider = config.get("llm", {}).get("provider", "openai")
    if require_llm:
        if provider == "openai" and not os.getenv("OPENAI_API_KEY", "").strip():
            errors.append("OPENAI_API_KEY is not set (required for llm.provider = openai)")
        elif provider == "groq" and not os.getenv("GROQ_API_KEY", "").strip():
            errors.append("GROQ_API_KEY is not set (required for llm.provider = groq). Get a free key at https://console.groq.com")

    if not config.get("teams", {}).get("webhook_url", "").strip():
        warnings.append("TEAMS_WEBHOOK_URL not configured — Teams alerts will be disabled")

    for w in warnings:
        console.print(f"[yellow]⚠  {w}[/yellow]")
    for e in errors:
        console.print(f"[red]✗  {e}[/red]")

    return len(errors) == 0


# ── Banner ───────────────────────────────────────────────────────────────────

def print_banner(config: dict) -> None:
    t = Text()
    t.append("AI Log Error Detector\n", style="bold cyan")
    t.append(
        "Log Collector → Error Detector → Embeddings → Vector DB → RAG+LLM → Screenshot → Teams\n",
        style="dim cyan",
    )

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("LLM Provider", f"{config.get('llm', {}).get('provider', 'openai')} / {config.get('llm', {}).get('model', 'gpt-4o')}")
    table.add_row("Embedding Model", config.get("embedding", {}).get("model", "all-MiniLM-L6-v2"))
    table.add_row("Min Severity", config.get("min_severity", "ERROR"))
    table.add_row("Log Files", str(len(config.get("log_paths", []))))
    table.add_row("KB Scraping", "enabled" if config.get("knowledge_base", {}).get("enabled", False) else "disabled")
    table.add_row("Teams Alerts", "enabled" if config.get("teams", {}).get("webhook_url") else "disabled")

    console.print(Panel(t, border_style="cyan", subtitle="[dim]Ctrl+C to stop[/dim]"))
    console.print(table)
    console.print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Log Error Detector — monitors log files and alerts via Teams"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    parser.add_argument("--seed-kb", action="store_true", help="Seed the knowledge base from knowledge_base/seed.yaml and exit")
    parser.add_argument("--scrape-kb-query", metavar="QUERY", help="Scrape Serena knowledge base for a query, cache results, index them, and exit")
    parser.add_argument("--test", action="store_true", help="Process a sample error line and exit (smoke test)")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = load_config(args.config)
    print_banner(config)

    require_llm = not bool(args.seed_kb or args.scrape_kb_query)
    if not validate_config(config, require_llm=require_llm):
        console.print("\n[red]Fix the errors above before starting.[/red]")
        sys.exit(1)

    # Deferred import so config validation happens first
    from src.pipeline import Pipeline

    pipeline = Pipeline(config)

    if args.seed_kb:
        pipeline.seed_knowledge_base(force=True)
        console.print("[green]Knowledge base seeded.[/green]")
        return

    if args.scrape_kb_query:
        indexed = pipeline.scrape_knowledge_base(args.scrape_kb_query)
        console.print(f"[green]Knowledge base scrape complete. Indexed {indexed} document(s).[/green]")
        return

    if args.test:
        pipeline.run_test()
        return

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(sig, frame):  # noqa: ARG001
        console.print("\n[yellow]Shutting down — please wait...[/yellow]")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pipeline.start()

    console.print(f"[green]Monitoring {len(config['log_paths'])} log file(s):[/green]")
    for p in config["log_paths"]:
        console.print(f"  [dim]→[/dim] {p}")
    console.print("\n[dim]Waiting for errors... (Ctrl+C to stop)[/dim]\n")

    try:
        while pipeline.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
