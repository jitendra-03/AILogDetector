"""
Teams Alert
===========
Sends formatted error alerts and AI-generated solutions to a Microsoft Teams
channel via an Incoming Webhook (MessageCard format — works with both legacy
Office 365 Connectors and Power Automate / Workflows webhooks).

Each alert card contains:
  - Severity, file name, timestamp
  - The raw error message
  - AI-generated root cause + solution
  - Path to the screenshot (if captured)
  - Log context snippet

Per-error cooldown prevents alert spam for the same repeated error.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_SEVERITY_COLOUR = {
    "CRITICAL": "FF0000",   # red
    "ERROR":    "FF4500",   # orange-red
    "WARNING":  "FFA500",   # orange
}
_SEVERITY_ICON = {
    "CRITICAL": "🔴",
    "ERROR":    "🟠",
    "WARNING":  "🟡",
}


class TeamsAlert:
    """
    Posts MessageCard-formatted alerts to a Teams channel.

    Config keys consumed (under ``teams``):
        webhook_url          – Teams Incoming Webhook URL (or TEAMS_WEBHOOK_URL env var)
        alert_cooldown       – seconds between duplicate alerts per error (default: 300)
        max_solution_length  – chars of solution shown in card (default: 800)
    """

    def __init__(self, config: dict) -> None:
        t = config.get("teams", {})
        self._webhook:     str   = (t.get("webhook_url") or "").strip() or \
                                   os.getenv("TEAMS_WEBHOOK_URL", "").strip()
        self._cooldown:    float = float(t.get("alert_cooldown", 300))
        self._max_sol_len: int   = int(t.get("max_solution_length", 800))

        # cooldown tracker: error_hash → last_sent_monotonic
        self._sent: dict = {}

        if self._webhook:
            logger.info("Teams alerts enabled")
        else:
            logger.warning("Teams webhook URL not configured — alerts will be skipped")

    # ── Public API ────────────────────────────────────────────

    def send_alert(
        self,
        error,
        solution: str,
        screenshot_path: Optional[str] = None,
        kb_matches: Optional[list] = None,
    ) -> bool:
        """
        Send a Teams alert card.

        Returns True on success, False if skipped (no webhook / cooldown) or on error.
        """
        if not self._webhook:
            logger.debug("Teams alert skipped — no webhook configured")
            return False

        if self._in_cooldown(error):
            logger.debug("Teams alert suppressed (cooldown active)")
            return False

        payload = self._build_adaptive_payload(error, solution, screenshot_path, kb_matches)

        try:
            resp = requests.post(
                self._webhook,
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            # Teams returns "1" (text/plain) on success
            if resp.status_code in (200, 201) or resp.text.strip() == "1":
                logger.info("Teams alert sent (HTTP %d)", resp.status_code)
                return True
            logger.warning(
                "Teams webhook returned unexpected status %d: %s",
                resp.status_code, resp.text[:200],
            )
            return False

        except requests.exceptions.RequestException as exc:
            logger.error("Failed to send Teams alert: %s", exc)
            return False

    # ── Adaptive Card builder ─────────────────────────────────

    def _build_adaptive_payload(
        self,
        error,
        solution: str,
        screenshot_path: Optional[str],
        kb_matches: Optional[list] = None,
    ) -> dict:
        """
        Build a Teams Adaptive Card payload.
        Screenshots are embedded as base64 data URIs so they render inline.
        """
        severity  = error.severity
        ac_colour = {"CRITICAL": "attention", "ERROR": "attention", "WARNING": "warning"}.get(severity, "default")
        icon      = _SEVERITY_ICON.get(severity, "⚪")
        ts        = error.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        fname     = Path(str(error.file_path)).name
        line_num  = getattr(error, "line_number", 0) or "—"

        short_sol = solution[:self._max_sol_len]
        if len(solution) > self._max_sol_len:
            short_sol += "\n\n*… [truncated — see logs for full solution]*"

        body = [
            {
                "type":   "TextBlock",
                "text":   f"{icon} AI Log Error Detected — {severity}",
                "weight": "Bolder",
                "size":   "Large",
                "color":  ac_colour,
                "wrap":   True,
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Severity",    "value": f"{icon} {severity}"},
                    {"title": "File",        "value": fname},
                    {"title": "Full Path",   "value": str(error.file_path)},
                    {"title": "Line Number", "value": str(line_num)},
                    {"title": "Detected At", "value": ts},
                ],
            },
            {
                "type": "TextBlock", "text": "**⚠ Error Message**",
                "weight": "Bolder", "separator": True, "wrap": True,
            },
            {
                "type":     "TextBlock",
                "text":     error.message[:600],
                "wrap":     True,
                "fontType": "Monospace",
                "color":    ac_colour,
            },
        ]

        # Inline screenshot — encoded as base64 so it renders in the Teams card
        if screenshot_path and os.path.exists(screenshot_path):
            b64, mime = self._encode_b64_image(screenshot_path)
            if b64:
                body += [
                    {
                        "type": "TextBlock",
                        "text": f"**📸 Screenshot at Error Line {line_num}**",
                        "weight": "Bolder", "separator": True,
                    },
                    {
                        "type":    "Image",
                        "url":     f"data:image/{mime};base64,{b64}",
                        "size":    "Stretch",
                        "altText": "Screenshot",
                    },
                ]

        body += [
            {
                "type": "TextBlock", "text": "**🤖 AI-Generated Solution**",
                "weight": "Bolder", "separator": True, "wrap": True,
            },
            {"type": "TextBlock", "text": short_sol, "wrap": True},
        ]

        defects = [m for m in (kb_matches or []) if m.get("channel") == "defects"]
        if defects:
            body.append({
                "type": "TextBlock", "text": "**🔍 Matching Serena KB Defects**",
                "weight": "Bolder", "separator": True, "wrap": True,
            })
            for d in defects[:3]:
                did   = d.get("defect_id") or "—"
                prod  = d.get("product") or "—"
                found = d.get("release_found") or "—"
                fixed = d.get("release_fixed") or "—"
                title = (d.get("title") or "")[:100]
                url   = d.get("url") or ""
                ln    = f"**{did}** — {title}\nProduct: {prod} | Found: {found} | Fixed: {fixed}"
                if url:
                    ln += f"\n[View Article]({url})"
                body.append({"type": "TextBlock", "text": ln, "wrap": True})

        if error.context and len(error.context) > 1:
            ctx = "\n".join(error.context[-5:])
            body += [
                {
                    "type": "TextBlock", "text": "**📋 Log Context**",
                    "weight": "Bolder", "separator": True, "wrap": True,
                },
                {"type": "TextBlock", "text": ctx[:600], "wrap": True, "fontType": "Monospace"},
            ]

        return {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl":  None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type":    "AdaptiveCard",
                    "version": "1.4",
                    "body":    body,
                },
            }],
        }

    # ── Cooldown helpers ──────────────────────────────────────

    def _build_batch_adaptive_payload(
        self,
        errors: list,
        lead,
        solution: str,
        screenshot_path: Optional[str],
        kb_matches: Optional[list] = None,
    ) -> dict:
        """Build a batch Adaptive Card payload for a burst of consecutive errors."""
        severity  = lead.severity
        ac_colour = {"CRITICAL": "attention", "ERROR": "attention", "WARNING": "warning"}.get(severity, "default")
        icon      = _SEVERITY_ICON.get(severity, "⚪")
        ts        = lead.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        fname     = Path(str(lead.file_path)).name

        short_sol = solution[:self._max_sol_len]
        if len(solution) > self._max_sol_len:
            short_sol += "\n\n*… [truncated — see logs for full solution]*"

        body = [
            {
                "type":   "TextBlock",
                "text":   f"{icon} Burst Alert — {len(errors)} errors in {fname}",
                "weight": "Bolder",
                "size":   "Large",
                "color":  ac_colour,
                "wrap":   True,
            },
            {
                "type": "TextBlock",
                "text": f"{len(errors)} consecutive errors | Highest: **{severity}** | {ts}",
                "wrap": True,
            },
            {
                "type": "TextBlock", "text": "**⚠ Errors in Burst**",
                "weight": "Bolder", "separator": True, "wrap": True,
            },
        ]

        for i, e in enumerate(errors, 1):
            ln = getattr(e, "line_number", 0) or "—"
            body.append({
                "type":     "TextBlock",
                "text":     f"**#{i} {e.severity} (line {ln}):** {e.message[:200]}",
                "wrap":     True,
                "fontType": "Monospace",
                "color":    ac_colour,
            })

        # Inline screenshot
        if screenshot_path and os.path.exists(screenshot_path):
            b64, mime = self._encode_b64_image(screenshot_path)
            if b64:
                body += [
                    {
                        "type": "TextBlock",
                        "text": "**📸 Screenshot at Error Line**",
                        "weight": "Bolder", "separator": True,
                    },
                    {
                        "type":    "Image",
                        "url":     f"data:image/{mime};base64,{b64}",
                        "size":    "Stretch",
                        "altText": "Screenshot",
                    },
                ]

        body += [
            {
                "type": "TextBlock", "text": "**🤖 AI-Generated Unified Solution**",
                "weight": "Bolder", "separator": True, "wrap": True,
            },
            {"type": "TextBlock", "text": short_sol, "wrap": True},
        ]

        defects = [m for m in (kb_matches or []) if m.get("channel") == "defects"]
        if defects:
            body.append({
                "type": "TextBlock", "text": "**🔍 Matching Serena KB Defects**",
                "weight": "Bolder", "separator": True, "wrap": True,
            })
            for d in defects[:3]:
                did   = d.get("defect_id") or "—"
                prod  = d.get("product") or "—"
                found = d.get("release_found") or "—"
                fixed = d.get("release_fixed") or "—"
                title = (d.get("title") or "")[:100]
                url   = d.get("url") or ""
                ln    = f"**{did}** — {title}\nProduct: {prod} | Found: {found} | Fixed: {fixed}"
                if url:
                    ln += f"\n[View Article]({url})"
                body.append({"type": "TextBlock", "text": ln, "wrap": True})

        return {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl":  None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type":    "AdaptiveCard",
                    "version": "1.4",
                    "body":    body,
                },
            }],
        }

    @staticmethod
    def _encode_b64_image(path: str, max_width: int = 800, quality: int = 65) -> tuple:
        """
        Resize image to at most *max_width* pixels wide and return a
        (base64_string, mime_type) tuple for embedding in an Adaptive Card.
        Uses Pillow when available (produces a compact JPEG); falls back to
        raw base64 encoding of the original file.
        Returns ("", "") on any error.
        """
        import base64
        try:
            from io import BytesIO
            try:
                from PIL import Image
                img = Image.open(path)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                if img.width > max_width:
                    new_h = int(img.height * max_width / img.width)
                    img = img.resize((max_width, new_h), Image.LANCZOS)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                return base64.b64encode(buf.getvalue()).decode("ascii"), "jpeg"
            except ImportError:
                with open(path, "rb") as f:
                    data = f.read()
                ext = Path(path).suffix.lower().lstrip(".")
                mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
                return base64.b64encode(data).decode("ascii"), mime
        except Exception as exc:
            logger.debug("Image encode failed for Teams: %s", exc)
            return "", ""

    def _in_cooldown(self, error) -> bool:
        key  = hashlib.md5(f"{error.file_path}:{error.message[:80]}".encode()).hexdigest()
        now  = time.monotonic()
        last = self._sent.get(key, 0.0)
        if (now - last) < self._cooldown:
            return True
        self._sent[key] = now
        return False

    # ── Batch / burst alert ───────────────────────────────────

    def send_batch_alert(
        self,
        errors: list,
        solution: str,
        screenshot_path: Optional[str] = None,
        kb_matches: Optional[list] = None,
    ) -> bool:
        """
        Send a single grouped Teams card for a burst of consecutive errors.
        Uses the most severe error's colour/icon as the card theme.
        """
        if not self._webhook:
            logger.debug("Teams batch alert skipped — no webhook configured")
            return False

        _rank = {"CRITICAL": 3, "ERROR": 2, "WARNING": 1}
        lead  = max(errors, key=lambda e: _rank.get(e.severity, 0))

        # Cooldown keyed on the batch signature (all messages combined)
        batch_key = hashlib.md5(
            "".join(e.message[:60] for e in errors).encode()
        ).hexdigest()
        now = time.monotonic()
        if (now - self._sent.get(batch_key, 0.0)) < self._cooldown:
            logger.debug("Teams batch alert suppressed (cooldown)")
            return False
        self._sent[batch_key] = now

        payload = self._build_batch_adaptive_payload(
            errors, lead, solution, screenshot_path, kb_matches,
        )

        try:
            resp = requests.post(
                self._webhook,
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code in (200, 201) or resp.text.strip() == "1":
                logger.info("Teams batch alert sent (%d errors, HTTP %d)", len(errors), resp.status_code)
                return True
            logger.warning("Teams webhook returned %d: %s", resp.status_code, resp.text[:200])
            return False
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to send Teams batch alert: %s", exc)
            return False

