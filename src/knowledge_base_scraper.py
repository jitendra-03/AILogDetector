"""
Serena Knowledge Base Scraper
=============================
Searches the Serena / Micro Focus knowledge base at
https://knowledgebase.serena.com, scrapes article content, and caches
normalized documents locally for later reuse.

Supports multi-channel search: Defects, Solutions, Alerts, Patches.
Defect articles include structured metadata: defect ID, product, versions.

How the site works
------------------
1. GET /InfoCenter/index?page=home  â†’ redirect to ?page=home&token=<hex>
   The session token must be preserved for all subsequent requests.
2. Search:  ?page=answers&type=search&question_box=<q>&fac=COLLECTIONS.<Ch>&token=<t>
   Results: <a href="index?page=answerlink&url=..."> per article
   Excerpts in: <div class="im-result-excerpt-block">
3. Article: follow answerlink â†’ ?page=content&id=<ID>&...
   Defect fields: DEFECTENHANCEMENT_ID, PRODUCT_NAME, RELEASE_FOUND_IN,
                  RELEASE_FIXED_IN, DESCRIPTION, RESOLUTION
   Solution fields: DESCRIPTION, RESOLUTION, content, contentbody
"""

import json
import logging
import re
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BASE = "https://knowledgebase.serena.com/InfoCenter/"
_HOME = _BASE + "index?page=home"

# CSS classes that hold content on article pages (priority order)
_ARTICLE_CONTENT_CLASSES = [
    "DESCRIPTION", "RESOLUTION", "content", "contentbody",
]

# CSS classes unique to defect articles (multi-class elements like "attr DEFECT_ID")
_DEFECT_META_CLASSES = {
    "defect_id":       "DEFECT_ID",
    "product":         "REPORTED_AGAINST",
    "release_found":   "ALSO_AFFECTS",
    "release_fixed":   "RESOLVED_IN",
}
# Label prefixes embedded in get_text() for each field — stripped before storing
_DEFECT_LABEL_STRIP = {
    "DEFECT_ID":         "Defect Id",
    "REPORTED_AGAINST":  "Originally Reported Against",
    "ALSO_AFFECTS":      "Also Affects",
    "RESOLVED_IN":       "Resolved In",
}

# Channel â†’ fac query parameter
_CHANNEL_FAC = {
    "defects":   "COLLECTIONS.Defects",
    "solutions": "COLLECTIONS.Solutions",
    "alerts":    "COLLECTIONS.Alerts",
    "patches":   "COLLECTIONS.Patches",
}


@dataclass
class KnowledgeBaseDocument:
    title:        str
    url:          str
    content:      str
    source:       str  = "serena_kb"
    search_query: str  = ""
    channel:      str  = "solutions"   # "defects" | "solutions" | "alerts" | "patches"
    # Defect-specific metadata (empty for non-defect articles)
    defect_id:      str = ""
    product:        str = ""
    release_found:  str = ""
    release_fixed:  str = ""
    article_id:     str = ""   # Serena KB article ID, e.g. S134348 or D22912


class KnowledgeBaseScraper:
    """Scrapes Serena knowledge-base pages and stores a local cache."""

    def __init__(self, config: dict) -> None:
        kb = config.get("knowledge_base", {})
        self.enabled           = bool(kb.get("enabled", False))
        self.base_url          = kb.get("base_url", _HOME)
        self.search_url_tpl    = kb.get(
            "search_url",
            _BASE + "index?page=answers&type=search&question_box={query}&token={token}",
        )
        self.cache_path        = Path(kb.get("cache_path", "./knowledge_base/serena_cache.jsonl"))
        self.max_results       = int(kb.get("max_results", 5))
        self.timeout           = int(kb.get("timeout", 20))
        self.max_article_chars = int(kb.get("max_article_chars", 4000))
        self.user_agent        = kb.get("user_agent", "AILogDetector/1.0")

        # Channels to search: list of "defects", "solutions", "alerts", "patches"
        raw_channels = kb.get("search_channels", ["defects", "solutions"])
        self.search_channels: List[str] = [c.lower() for c in raw_channels]

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent":        self.user_agent,
            "Accept":            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language":   "en-US,en;q=0.9",
            # Disable compression so the server never returns a binary-encoded body;
            # this prevents urllib3 from failing on a malformed Content-Length header
            # that can accompany gzip/deflate responses from some KB server versions.
            "Accept-Encoding":   "identity",
        })
        self._token:            str  = ""
        self._token_session_ok: bool = False

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def search_and_cache(self, query: str) -> List[KnowledgeBaseDocument]:
        """Search all configured channels for *query*, scrape articles, update cache."""
        if not self.enabled or not query.strip():
            return []

        self._ensure_token()
        all_docs: Dict[str, KnowledgeBaseDocument] = {}

        for channel in self.search_channels:
            try:
                docs = self._search_channel(query, channel)
            except Exception as exc:
                logger.warning("KB channel %r search failed for %r: %s", channel, query[:60], exc)
                continue
            for d in docs:
                if d.url not in all_docs:
                    all_docs[d.url] = d

        result = list(all_docs.values())
        if result:
            self._merge_cache(result)
            defect_count   = sum(1 for d in result if d.channel == "defects")
            solution_count = sum(1 for d in result if d.channel != "defects")
            logger.info(
                "Scraped %d KB doc(s) for %r â€” %d defect(s), %d solution(s)",
                len(result), query[:60], defect_count, solution_count,
            )
        return result

    # â”€â”€ Per-channel search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _search_channel(self, query: str, channel: str) -> List[KnowledgeBaseDocument]:
        fac = _CHANNEL_FAC.get(channel)
        if fac:
            search_url = self._build_search_url(query, fac=fac)
        else:
            search_url = self._build_search_url(query)

        html = self._fetch(search_url)
        if not html[0]:
            return []
        html, _ = html

        soup         = BeautifulSoup(html, "html.parser")
        answerlinks  = self._extract_answerlinks(soup, search_url)
        excerpts     = self._extract_excerpts(soup)
        docs: Dict[str, KnowledgeBaseDocument] = {}

        for idx, (title, link_url) in enumerate(answerlinks):
            if len(docs) >= self.max_results:
                break
            cached = self._load_cached_url(link_url)
            if cached:
                docs[link_url] = cached
                continue

            article_html, final_url = self._fetch(link_url)
            if not article_html:
                excerpt = excerpts[idx] if idx < len(excerpts) else ""
                if excerpt:
                    docs[link_url] = KnowledgeBaseDocument(
                        title=title, url=link_url,
                        content=excerpt, search_query=query, channel=channel,
                    )
                continue

            doc = self._parse_article(article_html, title, link_url, query, channel,
                                      final_url=final_url)
            if doc.content:
                docs[link_url] = doc

        return list(docs.values())

    # â”€â”€ Session / token â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _ensure_token(self) -> None:
        if self._token_session_ok:
            return
        try:
            r = self._session.get(_HOME, timeout=self.timeout, allow_redirects=True)
            m = re.search(r"token=([a-f0-9]+)", r.url)
            if m:
                self._token = m.group(1)
                logger.debug("Serena KB token: %sâ€¦", self._token[:8])
            self._token_session_ok = True
        except requests.RequestException as exc:
            logger.warning("Could not acquire Serena KB token: %s", exc)

    def _build_search_url(self, query: str, fac: str = "") -> str:
        tpl = self.search_url_tpl
        base = tpl.format(query=quote_plus(query.strip()), token=self._token) \
               if "{token}" in tpl else tpl.format(query=quote_plus(query.strip()))
        if fac:
            base += f"&fac={fac}"
        return base

    # â”€â”€ HTTP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _fetch(self, url: str) -> tuple:
        """Return (html: str, final_url: str) after following redirects."""
        try:
            r = self._session.get(url, timeout=self.timeout, allow_redirects=True)
            r.raise_for_status()
            # Skip non-text responses (binary downloads, unexpected MIME types) so
            # BeautifulSoup never receives raw bytes and no int()-on-bytes error occurs.
            ct = r.headers.get("Content-Type", "")
            if ct and not any(t in ct for t in ("text", "html", "xml")):
                logger.debug("Ignoring non-HTML response for %s (Content-Type: %s)", url[:80], ct)
                return "", r.url
            return r.text, r.url
        except requests.RequestException as exc:
            logger.warning("KB fetch failed %s: %s", url[:80], exc)
            return "", url

    # â”€â”€ Parsing search results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _extract_answerlinks(self, soup: BeautifulSoup, page_url: str) -> List[tuple]:
        results, seen = [], set()
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "page=answerlink" not in href:
                continue
            absolute = urljoin(page_url, href)
            title = " ".join(a.get_text(" ", strip=True).split())
            if not title or absolute in seen:
                continue
            seen.add(absolute)
            results.append((title[:200], absolute))
        return results

    def _extract_excerpts(self, soup: BeautifulSoup) -> List[str]:
        return [
            " ".join(div.get_text(" ", strip=True).split())
            for div in soup.find_all("div", class_="im-result-excerpt-block")
        ]

    # â”€â”€ Parsing article pages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _parse_article(
        self, html: str, fallback_title: str, url: str, query: str, channel: str,
        final_url: str = "",
    ) -> KnowledgeBaseDocument:
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title = fallback_title
        if soup.title and soup.title.get_text(strip=True):
            raw = soup.title.get_text(strip=True)
            title = re.sub(r"\s*[-|]\s*Serena.*$", "", raw).strip() or raw

        # Detect search-results / login pages (happen when session token expires).
        # Match titles that end with "Results" (the search results page) or
        # look like a login/redirect page â€” but NOT normal article titles like
        # "Serena Knowledgebase - Error connecting toâ€¦"
        _lc = title.lower()
        _is_bad = (
            re.search(r'[-\s]results\s*$', _lc) is not None   # "â€¦ - Results"
            or _lc in ("results", "search results")
            or "login" in _lc
            or "sign in" in _lc
        )
        if _is_bad:
            logger.debug(
                "Article URL returned a non-article page (title=%r) â€” resetting token: %s",
                title, url[:80],
            )
            self._token_session_ok = False  # Force re-auth on next scrape
            return KnowledgeBaseDocument(title="", url=url, content="")

        # Article ID: prefer final redirected URL (e.g. ?id=S125970), fall back to original URL
        article_id = ""
        for candidate_url in (final_url, url):
            id_match = re.search(r"[?&]id=([A-Za-z0-9]+)", candidate_url or "")
            if id_match:
                article_id = id_match.group(1)
                break

        # Defect-specific metadata
        defect_id     = self._defect_field(soup, "DEFECT_ID")
        product       = self._defect_field(soup, "REPORTED_AGAINST")
        release_found = self._defect_field(soup, "ALSO_AFFECTS")
        release_fixed = self._defect_field(soup, "RESOLVED_IN")

        # For defect_id, extract DEF\d+ pattern to get a clean ID
        m_def = re.search(r'\bDEF\d+\b', defect_id)
        if m_def:
            defect_id = m_def.group(0)

        # Main content
        parts: List[str] = []
        for cls in _ARTICLE_CONTENT_CLASSES:
            el = soup.find(class_=cls)
            if el:
                text = self._clean_text(el.get_text("\n", strip=True))
                if text and text not in parts:
                    parts.append(f"[{cls}]\n{text}")

        if not parts:
            body = soup.body or soup
            parts.append(self._clean_text(body.get_text("\n", strip=True)))

        # Prepend structured header
        header_parts = []
        if article_id:
            header_parts.append(f"Article ID: {article_id}")
        if defect_id:
            header_parts.append(f"Defect ID: {defect_id}")
        if product:
            header_parts.append(f"Product: {product}")
        if release_found:
            header_parts.append(f"Found In: {release_found}")
        if release_fixed:
            header_parts.append(f"Fixed In: {release_fixed}")

        full_content = ""
        if header_parts:
            full_content = "\n".join(header_parts) + "\n\n"
        full_content += "\n\n".join(parts)
        full_content = full_content[: self.max_article_chars]

        return KnowledgeBaseDocument(
            title=title, url=url, content=full_content,
            search_query=query, channel=channel,
            article_id=article_id,
            defect_id=defect_id, product=product,
            release_found=release_found, release_fixed=release_fixed,
        )

    # â”€â”€ Cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _load_cached_url(self, url: str) -> Optional[KnowledgeBaseDocument]:
        if not self.cache_path.exists():
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("url") == url and row.get("content"):
                        cached_title = row.get("title", "")
                        # Skip poisoned entries (captured when session/token was invalid)
                        _lc = cached_title.lower()
                        _is_poisoned = (
                            re.search(r'[-\s]results\s*$', _lc) is not None
                            or _lc in ("results", "search results")
                            or "login" in _lc
                        )
                        if _is_poisoned:
                            continue
                        fields = KnowledgeBaseDocument.__dataclass_fields__
                        data = {}
                        for k, f in fields.items():
                            if k in row:
                                data[k] = row[k]
                            elif f.default is not dataclasses.MISSING:
                                data[k] = f.default
                            else:
                                data[k] = ""
                        return KnowledgeBaseDocument(**data)
        except Exception:
            pass
        return None

    def _merge_cache(self, documents: List[KnowledgeBaseDocument]) -> None:
        existing: Dict[str, dict] = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row.get("url"):
                            existing[row["url"]] = row
            except Exception:
                pass
        for doc in documents:
            existing[doc.url] = asdict(doc)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            for row in existing.values():
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _defect_field(self, soup: BeautifulSoup, attr_name: str) -> str:
        """
        Extract a defect-metadata field value, stripping the embedded label prefix.
        E.g. "Defect Id DEF231500" -> "DEF231500"
             "Originally Reported Against SBM 10.1.2" -> "SBM 10.1.2"
        """
        el = soup.find(class_=attr_name) or soup.find(id=attr_name)
        if not el:
            return ""
        raw = " ".join(el.get_text(" ", strip=True).split())
        label = _DEFECT_LABEL_STRIP.get(attr_name, "")
        if label and raw.lower().startswith(label.lower()):
            raw = raw[len(label):].strip()
        return raw

    def _text(self, soup: BeautifulSoup, attr_name: str) -> str:
        """Extract text from an element identified by CSS class OR id attribute."""
        el = (
            soup.find(class_=attr_name)
            or soup.find(id=attr_name)
            or soup.find(attrs={"name": attr_name})
        )
        return " ".join(el.get_text(" ", strip=True).split()) if el else ""

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [" ".join(ln.split()) for ln in text.splitlines()]
        result, prev_blank = [], False
        for ln in lines:
            is_blank = not ln
            if is_blank and prev_blank:
                continue
            result.append(ln)
            prev_blank = is_blank
        return "\n".join(result).strip()

