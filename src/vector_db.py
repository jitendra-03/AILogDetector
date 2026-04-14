"""
Vector Database (Knowledge Base)
=================================
Pure-Python/numpy vector store with JSON persistence.
Replaces ChromaDB to avoid C-extension incompatibilities on Python 3.14.

Schema (per document):
  id        – MD5 of the error message (deduplication key)
  embedding – float list (384-dim)
  document  – raw error message text
  metadata  –
    severity   : str
    file_path  : str
    solution   : str   (AI-generated fix, up to 2 000 chars)
    context    : str   (surrounding log lines, up to 1 000 chars)
    timestamp  : str   (ISO-8601)
    source     : str   "live" | "seed" | "serena_kb"
"""

import hashlib
import json
import logging
import math
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_META_SOLUTION_MAX = 2000
_META_CONTEXT_MAX  = 1000


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float lists."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorDB:
    """
    JSON-file-backed knowledge base with in-memory cosine similarity search.

    Config keys consumed (under ``vector_db``):
        path             – directory for persistent storage
        collection_name  – used as the JSON filename stem
        n_results        – default number of similar results to return
    """

    def __init__(self, config: dict) -> None:
        db_cfg = config.get("vector_db", {})
        db_path = db_cfg.get("path", "./knowledge_base/chromadb")
        self._collection_name = db_cfg.get("collection_name", "log_errors")
        self._n_results: int = db_cfg.get("n_results", 3)

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._store_path = Path(db_path) / f"{self._collection_name}.json"

        # In-memory store: {id: {"embedding": [...], "document": str, "metadata": {}}
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load()

        logger.info(
            "Vector DB ready | collection=%s | stored=%d",
            self._collection_name, len(self._store),
        )

    # ── Persistence ──────────────────────────────────────────

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                with open(self._store_path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except Exception as exc:
                logger.warning("Could not load vector store (%s) — starting empty", exc)
                self._store = {}

    def _save(self) -> None:
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f)
        except Exception as exc:
            logger.warning("Could not persist vector store: %s", exc)

    # ── Write ─────────────────────────────────────────────────

    def _upsert(self, doc_id: str, embedding: List[float], document: str, metadata: dict) -> None:
        self._store[doc_id] = {
            "embedding": embedding,
            "document": document,
            "metadata": metadata,
        }
        self._save()

    def add_error(self, error, solution: str, embedding_gen) -> str:
        """
        Store a DetectedError together with its AI-generated solution.
        Uses UPSERT so re-processing the same log line is idempotent.
        Returns the document ID.
        """
        doc_id = self._make_id(error.message)
        embedding = embedding_gen.encode_error(error.message, error.context)
        self._upsert(doc_id, embedding, error.message, {
            "severity":  error.severity,
            "file_path": str(error.file_path),
            "solution":  solution[:_META_SOLUTION_MAX],
            "context":   "\n".join(error.context[-5:])[:_META_CONTEXT_MAX],
            "timestamp": error.timestamp.isoformat(),
            "source":    "live",
        })
        logger.info("Solution stored in knowledge base | solution_id=%s", doc_id)
        return doc_id

    def add_knowledge_documents(self, documents: List[object], embedding_gen) -> int:
        """Store scraped Serena knowledge-base documents for later retrieval."""
        added = 0
        _BAD_KB_TITLES = ("knowledgebase - results", "search results", "serena - login")
        for doc in documents:
            title_val = (getattr(doc, "title", "") or "").lower()
            if any(m in title_val for m in _BAD_KB_TITLES) or \
               re.search(r'[-\s]results\s*$', title_val):
                logger.debug("Skipping poisoned KB document: %r", getattr(doc, "title", ""))
                continue
            doc_id = f"kb_{self._make_id(getattr(doc, 'url', '') or getattr(doc, 'title', ''))}"
            text = f"{getattr(doc, 'title', '')}\n{getattr(doc, 'content', '')}".strip()
            if not text:
                continue
            embedding = embedding_gen.encode(text)
            self._upsert(doc_id, embedding, text, {
                "severity":      "INFO",
                "file_path":     getattr(doc, "url", "knowledge_base/serena"),
                "solution":      getattr(doc, "content", "")[:_META_SOLUTION_MAX],
                "context":       "",
                "timestamp":     datetime.now().isoformat(),
                "source":        getattr(doc, "source", "serena_kb"),
                "title":         getattr(doc, "title", ""),
                "url":           getattr(doc, "url", ""),
                "article_id":    getattr(doc, "article_id", ""),
                "search_query":  getattr(doc, "search_query", ""),
                "channel":       getattr(doc, "channel", "solutions"),
                "defect_id":     getattr(doc, "defect_id", ""),
                "product":       getattr(doc, "product", ""),
                "release_found": getattr(doc, "release_found", ""),
                "release_fixed": getattr(doc, "release_fixed", ""),
            })
            added += 1
        if added:
            logger.info("Indexed %d scraped knowledge-base document(s)", added)
        return added

    def seed_from_yaml(self, yaml_path: str, embedding_gen, force: bool = False) -> int:
        """
        Populate the knowledge base from a YAML file.

        YAML format::

            errors:
              - error: "Connection refused"
                solution: "Check that the target service is running…"
                severity: ERROR    # optional, default ERROR

        Returns the number of new entries added.
        """
        path = Path(yaml_path)
        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        entries = data.get("errors", [])
        added = 0

        for entry in entries:
            err_msg  = (entry.get("error")    or "").strip()
            solution = (entry.get("solution") or "").strip()
            severity = (entry.get("severity") or "ERROR").upper()

            if not err_msg or not solution:
                continue

            doc_id = f"seed_{self._make_id(err_msg)}"
            if not force and doc_id in self._store:
                continue

            embedding = embedding_gen.encode(err_msg)
            self._upsert(doc_id, embedding, err_msg, {
                "severity":  severity,
                "file_path": "knowledge_base/seed",
                "solution":  solution[:_META_SOLUTION_MAX],
                "context":   "",
                "timestamp": datetime.now().isoformat(),
                "source":    "seed",
            })
            added += 1

        if added:
            logger.info("Seeded %d entries from %s", added, yaml_path)
        return added

    # ── Read ──────────────────────────────────────────────────

    def query_similar(
        self,
        error_message: str,
        embedding: List[float],
        min_similarity: float = 0.35,
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar past errors from the knowledge base.

        Returns a list of dicts with keys:
            text, solution, severity, similarity (0–1)
        """
        if not self._store:
            return []

        scored = []
        for doc_id, record in self._store.items():
            meta = record["metadata"]
            if sources and meta.get("source") not in sources:
                continue
            sim = _cosine_similarity(embedding, record["embedding"])
            if sim >= min_similarity:
                scored.append((sim, doc_id, record))

        scored.sort(key=lambda x: x[0], reverse=True)

        similar: List[Dict[str, Any]] = []
        _BAD_KB_TITLES = ("knowledgebase - results", "search results", "serena - login")
        for sim, doc_id, record in scored[: self._n_results]:
            meta = record["metadata"]
            # Skip poisoned entries that were indexed before validation was in place
            if meta.get("source") == "serena_kb":
                title_stored = (meta.get("title", "") or "").lower()
                if any(m in title_stored for m in _BAD_KB_TITLES) or \
                   re.search(r'[-\s]results\s*$', title_stored):
                    continue
            similar.append({
                "text":          record["document"],
                "solution":      meta.get("solution", ""),
                "severity":      meta.get("severity", ""),
                "title":         meta.get("title", ""),
                "url":           meta.get("url", ""),
                "article_id":    meta.get("article_id", ""),
                "source":        meta.get("source", ""),
                "search_query":  meta.get("search_query", ""),
                "channel":       meta.get("channel", "solutions"),
                "defect_id":     meta.get("defect_id", ""),
                "product":       meta.get("product", ""),
                "release_found": meta.get("release_found", ""),
                "release_fixed": meta.get("release_fixed", ""),
                "similarity":    round(sim, 3),
            })

        return similar

    # ── Helpers ───────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._store)

    @staticmethod
    def _make_id(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]



