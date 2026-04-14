"""
RAG + LLM Engine
================
Retrieval-Augmented Generation (RAG) for log error analysis.

Flow:
  1. Embed the incoming error.
  2. Query the vector DB for the top-k semantically similar past errors.
  3. Build a structured prompt that includes the error, its log context,
     and the similar historical examples + known solutions.
  4. Call the configured LLM (OpenAI-compatible API or Ollama).
  5. Return the generated solution text.

Supports:
  - OpenAI (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, …)
  - Azure OpenAI  (set AZURE_OPENAI_* env vars + api_base in config)
  - Ollama        (provider: ollama, api_base: http://localhost:11434/v1)
  - Groq          (provider: groq, free tier — https://console.groq.com)
"""

import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert system administrator and software engineer specialising in \
log-file analysis and production incident resolution.

When given an error from a log file you will:
1. Identify the **root cause** concisely.
2. Provide a **step-by-step solution** (commands, config changes, code fixes).
3. Suggest a **prevention strategy** to stop it recurring.

Rules:
- Be direct and actionable — avoid unnecessary preamble.
- Use Markdown headings: **Root Cause**, **Solution**, **Prevention**.
- If you cannot determine a solution, say so clearly and suggest next-steps.
- Never fabricate file paths or commands that do not exist.
"""


# ── Engine ────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Config keys consumed (under ``llm``):
        provider    – "openai" | "ollama" | "groq"  (default: openai)
        model       – model identifier               (default: gpt-4o)
        api_base    – override base URL              (optional)
        temperature – sampling temperature           (default: 0.3)
        max_tokens  – max response tokens            (default: 1000)
    """

    def __init__(self, config: dict, vector_db, embedding_gen, kb_scraper) -> None:
        self.vector_db     = vector_db
        self.embedding_gen = embedding_gen
        self.kb_scraper    = kb_scraper

        llm = config.get("llm", {})
        kb = config.get("knowledge_base", {})
        self._provider:    str   = llm.get("provider", "openai").lower()
        self._model:       str   = llm.get("model", "gpt-4o")
        self._temperature: float = float(llm.get("temperature", 0.3))
        self._max_tokens:  int   = int(llm.get("max_tokens", 1000))
        self._api_base:    str   = llm.get("api_base", "") or ""
        self._kb_enabled:  bool  = bool(kb.get("enabled", False))
        self._kb_auto:     bool  = bool(kb.get("auto_search_on_error", True))
        self._kb_threshold: float = float(kb.get("similarity_threshold", 0.90))

        self._client = None  # lazy-initialised
        logger.info(
            "RAG engine configured | provider=%s model=%s kb_threshold=%.2f",
            self._provider, self._model, self._kb_threshold,
        )

    # ── Lazy LLM client ───────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self):
        from openai import OpenAI, AzureOpenAI

        if self._provider == "azure":
            return AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )

        kwargs: dict = {}
        if self._api_base:
            kwargs["base_url"] = self._api_base

        if self._provider == "ollama":
            kwargs.setdefault("base_url", "http://localhost:11434/v1")
            kwargs["api_key"] = "ollama"
        elif self._provider == "groq":
            kwargs["base_url"] = "https://api.groq.com/openai/v1"
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. "
                    "Get a free key at https://console.groq.com and add it to your .env file."
                )
            kwargs["api_key"] = api_key
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file or set it as an environment variable."
                )
            kwargs["api_key"] = api_key

        return OpenAI(**kwargs)

    # ── Public API ────────────────────────────────────────────

    def analyze(self, error) -> tuple:
        """
        Analyse a DetectedError with RAG + LLM.
        Returns (solution: str, kb_matches: list).
        """
        embedding = self.embedding_gen.encode_error(error.message, error.context)

        fresh_docs = []
        if self._kb_enabled and self._kb_auto:
            try:
                kb_query = self._clean_kb_query(error.message)
                logger.info("KB search query: %r", kb_query[:120])
                fresh_docs = self.kb_scraper.search_and_cache(kb_query) or []
                if fresh_docs:
                    self.vector_db.add_knowledge_documents(fresh_docs, self.embedding_gen)
            except Exception as exc:
                logger.warning("Knowledge-base scrape failed for %r: %s", error.message[:80], exc)

        similar = self.vector_db.query_similar(
            error.message, embedding,
            min_similarity=self._kb_threshold, sources=["serena_kb"],
        )
        similar = self._merge_fresh_docs(similar, fresh_docs)

        self._log_kb_matches(similar)
        prompt = self._build_prompt(error, similar)
        solution = self._call_llm(prompt, fallback_message=error.message)
        return solution, similar

    def analyze_batch(self, errors: List) -> tuple:
        """
        Analyse a burst of consecutive errors with a single LLM call.
        Returns (solution: str, kb_matches: list).
        """
        lead = max(errors, key=lambda e: {"CRITICAL": 3, "ERROR": 2, "WARNING": 1}.get(e.severity, 0))

        # Scrape KB using the most specific error message in the batch
        fresh_docs = []
        if self._kb_enabled and self._kb_auto:
            try:
                kb_query = self._best_kb_query([e.message for e in errors])
                logger.info("KB search query (batch): %r", kb_query[:120])
                fresh_docs = self.kb_scraper.search_and_cache(kb_query) or []
                if fresh_docs:
                    self.vector_db.add_knowledge_documents(fresh_docs, self.embedding_gen)
            except Exception as exc:
                logger.warning("KB scrape failed for batch lead: %s", exc)

        embedding = self.embedding_gen.encode_error(lead.message, lead.context)
        similar = self.vector_db.query_similar(
            lead.message, embedding,
            min_similarity=self._kb_threshold, sources=["serena_kb"],
        )
        similar = self._merge_fresh_docs(similar, fresh_docs)

        self._log_kb_matches(similar)
        prompt = self._build_batch_prompt(errors, similar)
        solution = self._call_llm(prompt, fallback_message=f"Batch of {len(errors)} errors")
        return solution, similar

    # ── KB helpers ────────────────────────────────────────────

    @staticmethod
    def _best_kb_query(messages: List[str]) -> str:
        """
        Given a list of raw log messages from a burst, pick the one whose
        cleaned text is most useful for a KB search.

        Scoring (higher = better):
          +10  contains a product error code  e.g. ORA-02158, HTTP-404, SQLSTATE
          +8   contains a Java/app exception  e.g. NullPointerException, SQLException
          +5   contains a version string       e.g. 12.2.1, 8.3.4
          +3   longer cleaned message         (more context = more specific)
          -5   message is very generic        e.g. "failed to initialize"
        """
        _CODE_RE   = re.compile(r'\b[A-Z]{2,}-\d{3,}\b|SQLSTATE|HTTP\s*\d{3}', re.IGNORECASE)
        _EXC_RE    = re.compile(r'\b\w+(?:Exception|Error|Fault)\b')
        _VER_RE    = re.compile(r'\b\d+\.\d+[\.\d]*\b')
        _GENERIC   = re.compile(r'^(failed to (initialize|start|connect)|service (unavailable|failed))$', re.IGNORECASE)

        best_query, best_score = "", -999
        for msg in messages:
            q = RAGEngine._clean_kb_query(msg)
            if not q:
                continue
            score = 0
            if _CODE_RE.search(q):  score += 10
            if _EXC_RE.search(q):   score += 8
            if _VER_RE.search(q):   score += 5
            score += min(len(q) // 20, 3)   # length bonus, capped at +3
            if _GENERIC.match(q):   score -= 5
            if score > best_score:
                best_score, best_query = score, q
        return best_query or RAGEngine._clean_kb_query(messages[0])

    @staticmethod
    def _clean_kb_query(message: str) -> str:
        """
        Strip log-formatting noise from a raw log message before sending it
        to the Serena KB search engine.

        Removes:
          - Leading timestamps   e.g. "2025-04-09 18:47:51,095 "
          - Log level keywords   e.g. "ERROR ", "FATAL ", "WARN "
          - Java thread names    e.g. "[Catalina-utility-10] "
          - Java class + line    e.g. "[com.serena.eventlog.EventManager:268] "
          - MDC brackets         e.g. "[::] "
          - Leading dashes/pipes e.g. "-- "

        The remaining text is the human-readable error message that the KB
        search engine can match against article titles and content.
        """
        q = message.strip()

        # Multi-line message (continuation lines appended to the triggering error):
        # pick the single most informative line for KB search rather than sending
        # the whole block to the search engine.
        if '\n' in q:
            _candidates = [ln.strip() for ln in q.splitlines() if ln.strip()]
            def _line_score(ln: str) -> int:  # noqa: E306
                # A complete timestamped INFO/DEBUG/TRACE line is a coincidental
                # continuation, never the right KB query — exclude it entirely.
                if re.match(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', ln) and \
                        re.search(r'\b(INFO|DEBUG|TRACE)\b', ln[:80], re.IGNORECASE):
                    return -999
                # Strip timestamp + level to get the actual content for length scoring
                _content = re.sub(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[,.\d]*\s*', '', ln)
                _content = re.sub(r'^(FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|CRITICAL)\b\s*',
                                  '', _content, flags=re.IGNORECASE).strip()
                s = min(len(_content), 80)   # length bonus on stripped content (capped)
                if re.search(r'\bERR(?:OR)?\s*[#:]\s*\d+\b', ln, re.IGNORECASE): s += 500
                if re.search(r'\b[A-Z]{2,}-\d{3,}\b', ln):                       s += 400
                if re.search(r'\b\w+(?:Exception|Error|Fault)\b', ln):            s += 300
                if re.search(r'"[^"]{3,}"', ln):                                   s += 200
                # Penalise pure-context lines that add no searchable signal
                if re.search(r'^(?:User\s*:|Line\s+-?\d+>|\(called from)', ln,
                             re.IGNORECASE):                                        s -= 800
                return s
            q = max(_candidates, key=_line_score)

        # ISO-8601-style timestamp: 2025-04-09 18:47:51,095
        q = re.sub(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[,.\d]*\s*', '', q)
        # Log levels
        q = re.sub(r'^(FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|CRITICAL)\s+', '', q, flags=re.IGNORECASE)
        # Java thread name: [Catalina-utility-10] or [http-nio-8080-exec-1]
        q = re.sub(r'^\[[\w\-\. ]+\]\s*', '', q)
        # Java class + line number: [com.serena.anything.ClassName:123]
        q = re.sub(r'^\[(?:[a-z][a-z0-9]*\.)+\w+(?::\d+)?\]\s*', '', q, flags=re.IGNORECASE)
        # MDC/context brackets like [::] or [user::session]
        q = re.sub(r'^\[[^\[\]]{0,40}\]\s*', '', q)
        # Leading dashes, pipes, colons
        q = re.sub(r'^[-|:]+\s*', '', q)
        # Any remaining leading log-level word
        q = re.sub(r'^(FATAL|ERROR|WARN(?:ING)?|EXCEPTION|Exception):\s*', '', q, flags=re.IGNORECASE)

        # Strip "at/in file <path>" location markers (pure noise — path adds nothing useful)
        q = re.sub(r'\b(?:at|in)\s+file\s+\S+', '', q, flags=re.IGNORECASE)
        # Strip "file <absolute-path>" keeping the preceding verb context (e.g. "Cannot open")
        q = re.sub(r'\bfile\s+(?:[A-Za-z]:[/\\]\S+|/\S+)', '', q, flags=re.IGNORECASE)
        # Strip any remaining bare Windows drive-letter paths: C:\...\foo.xml
        q = re.sub(r'\b[A-Za-z]:[/\\]\S+', '', q)
        # Strip any remaining bare Unix absolute paths: /opt/tomcat/conf/server.xml
        q = re.sub(r'(?<![A-Za-z0-9_])/[a-zA-Z0-9_.\-/]+', '', q)
        # Strip line/char/column references: ", line 307" / "char 9" / "(line 307)"
        q = re.sub(r',?\s*(?:at\s+)?(?:line|char|col(?:umn)?)\s+\d+', '', q, flags=re.IGNORECASE)
        # Collapse multiple spaces left by the above removals
        q = re.sub(r'\s{2,}', ' ', q).strip()
        # Strip a trailing comma or dash left after path/line removal
        q = re.sub(r'[,\-]+\s*$', '', q).strip()

        q = q.strip()
        # If everything was stripped fall back to a shortened original
        return q if q else message[:200]

    def _merge_fresh_docs(self, similar: List[dict], fresh_docs: list) -> List[dict]:
        """
        Merge freshly-scraped KB documents into the `similar` list.
        Fresh docs are always relevant (the KB search engine already ranked them),
        so they bypass the embedding-similarity threshold.
        Deduplicates by URL; preserves existing entries that have a real similarity score.
        """
        if not fresh_docs:
            return similar

        existing_urls = {h.get("url") for h in similar}
        for doc in fresh_docs:
            url   = getattr(doc, "url", "")
            title = getattr(doc, "title", "") or ""
            # Skip poisoned pages (search-results pages)
            _BAD = ("knowledgebase - results", "search results", "serena - login")
            if any(m in title.lower() for m in _BAD):
                continue
            if url and url in existing_urls:
                continue
            similar.append({
                "text":          f"{title}\n{getattr(doc, 'content', '')}",
                "solution":      getattr(doc, "content", "")[:800],
                "severity":      "INFO",
                "title":         title,
                "url":           url,
                "article_id":    getattr(doc, "article_id", ""),
                "source":        getattr(doc, "source", "serena_kb"),
                "search_query":  getattr(doc, "search_query", ""),
                "channel":       getattr(doc, "channel", "solutions"),
                "defect_id":     getattr(doc, "defect_id", ""),
                "product":       getattr(doc, "product", ""),
                "release_found": getattr(doc, "release_found", ""),
                "release_fixed": getattr(doc, "release_fixed", ""),
                "similarity":    0.0,   # not ranked by embedding; included via direct scrape
            })
            if url:
                existing_urls.add(url)
        return similar

    def _log_kb_matches(self, similar: List[dict]) -> None:
        """Log each KB match with its solution/defect ID and similarity score."""
        for hit in similar:
            if hit.get("channel") == "defects" and hit.get("defect_id"):
                logger.info(
                    "KB defect match | defect_id=%s product=%s score=%.4f | %s",
                    hit["defect_id"],
                    hit.get("product", "—"),
                    hit.get("similarity", 0.0),
                    (hit.get("title") or "")[:120],
                )
            elif hit.get("source") == "serena_kb" and hit.get("title"):
                solution_id = hit.get("article_id") or hit.get("defect_id") or "—"
                logger.info(
                    "KB solution match | solution_id=%s score=%.4f | %s",
                    solution_id,
                    hit.get("similarity", 0.0),
                    (hit.get("title") or hit.get("text", ""))[:120],
                )

    # ── Prompt construction ───────────────────────────────────

    def _build_prompt(self, error, similar: List[dict]) -> str:
        lines: List[str] = [
            "## New Error Detected",
            f"**Severity:** {error.severity}",
            f"**File:** `{error.file_path}`",
            f"**Timestamp:** {error.timestamp.isoformat()}",
            "",
            "**Error Message:**",
            "```",
            error.message,
            "```",
        ]

        if error.context and len(error.context) > 1:
            lines += [
                "",
                "**Surrounding Log Context:**",
                "```",
                *error.context[-6:],
                "```",
            ]

        if similar:
            defects   = [s for s in similar if s.get("channel") == "defects"]
            solutions = [s for s in similar if s.get("channel") != "defects"]

            if defects:
                lines += ["", f"## Matched Serena KB Defects (>= {self._kb_threshold:.0%} similarity)"]
                for i, s in enumerate(defects, 1):
                    lines += [
                        "",
                        f"### Defect {i}  (similarity: {s['similarity']:.0%})",
                        f"**Title:** {s.get('title', '')[:200]}",
                    ]
                    if s.get("defect_id"):  lines.append(f"**Defect ID:** {s['defect_id']}")
                    if s.get("product"):    lines.append(f"**Product:** {s['product']}")
                    if s.get("release_found"): lines.append(f"**Found In:** {s['release_found']}")
                    if s.get("release_fixed"): lines.append(f"**Fixed In:** {s['release_fixed']}")
                    lines += [f"**URL:** {s.get('url', '')}", f"**Content:** {s['text'][:600]}"]

            if solutions:
                lines += ["", f"## Matched Serena KB Solutions (>= {self._kb_threshold:.0%} similarity)"]
                for i, s in enumerate(solutions, 1):
                    lines += [
                        "",
                        f"### Solution {i}  (similarity: {s['similarity']:.0%})",
                        f"**Title:** {s.get('title', '')[:200]}",
                        f"**URL:** {s.get('url', '')}",
                        f"**Content:** {s['text'][:600]}",
                    ]
        else:
            lines += ["", f"*(No Serena KB results met the {self._kb_threshold:.0%} similarity threshold.)*"]

        lines += ["", "---", "Please analyse the error above and provide a solution."]
        return "\n".join(lines)

    def _build_batch_prompt(self, errors: List, similar: List[dict]) -> str:
        """Build a combined prompt for a burst of consecutive errors."""
        lines: List[str] = [
            f"## Burst of {len(errors)} Consecutive Errors (no INFO lines between them)",
            "",
            "These errors occurred back-to-back in the same log stream — analyse them together "
            "and provide a single unified root cause, solution, and prevention strategy.",
            "",
        ]
        for i, e in enumerate(errors, 1):
            lines += [
                f"### Error {i} — {e.severity}",
                "```",
                e.message,
                "```",
            ]
            if e.context and len(e.context) > 1:
                lines += ["*Context:*", "```", *e.context[-3:], "```"]
            lines.append("")

        if similar:
            lines += [f"## High-Confidence KB Matches (>= {self._kb_threshold:.0%})"]
            for i, s in enumerate(similar, 1):
                lines += [
                    f"### KB {i}  (similarity: {s['similarity']:.0%})",
                    f"**Title:** {s.get('title', '')[:200]}",
                    f"**Content:** {s['text'][:600]}",
                    "",
                ]
        else:
            lines += [f"*(No KB results met the {self._kb_threshold:.0%} threshold.)*", ""]

        lines += ["---", "Provide a unified **Root Cause**, **Solution**, and **Prevention** for all errors above."]
        return "\n".join(lines)

    def _call_llm(self, prompt: str, fallback_message: str = "") -> str:
        """Send prompt to the configured LLM and return the response text."""
        try:
            model_arg = (
                os.getenv("AZURE_OPENAI_DEPLOYMENT", self._model)
                if self._provider == "azure"
                else self._model
            )
            response = self.client.chat.completions.create(
                model=model_arg,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            solution = response.choices[0].message.content.strip()
            logger.info("LLM solution generated (%d chars)", len(solution))
            return solution

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                f"**Root Cause:** LLM unavailable — manual investigation required.\n\n"
                f"**Error:** `{fallback_message}`\n\n"
                f"**LLM Error:** `{exc}`"
            )

