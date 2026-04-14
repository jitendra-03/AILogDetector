"""
Email Alert
===========
Sends formatted HTML error alerts (with screenshot attachments and inline images)
via SMTP whenever the pipeline detects an error.

Each email contains:
  - Subject with error count, file path, and line number
  - Error severity, file path, timestamp
  - The raw error message
  - AI-generated root cause + solution
  - Log context snippet
  - Screenshot attached + embedded inline (when captured)

Per-error cooldown prevents alert spam for the same repeated error.

Config keys consumed (under ``email``):
    enabled          – whether to send emails at all (default: true)
    smtp_host        – SMTP server hostname
    smtp_port        – SMTP server port (default: 587)
    use_starttls     – upgrade plain connection with STARTTLS (default: true)
    use_ssl          – use SMTP_SSL from the start (default: false)
    username         – SMTP login username (or SMTP_USERNAME env var)
    password         – SMTP login password (or SMTP_PASSWORD env var)
    sender           – From address
    recipients       – list of To addresses
    alert_cooldown   – seconds between duplicate alerts per error (default: 300)
"""

import hashlib
import logging
import mimetypes
import os
import smtplib
import ssl
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_SEVERITY_COLOUR = {
    "CRITICAL": "#FF0000",
    "ERROR":    "#FF4500",
    "WARNING":  "#FFA500",
}
_SEVERITY_ICON = {
    "CRITICAL": "🔴",
    "ERROR":    "🟠",
    "WARNING":  "🟡",
}


class EmailAlert:
    """
    Sends HTML email alerts to a configured list of recipients.

    The ``send_alert`` signature is intentionally identical to
    ``TeamsAlert.send_alert`` so the pipeline can call them uniformly.
    """

    def __init__(self, config: dict) -> None:
        e = config.get("email", {})
        self._enabled:   bool  = e.get("enabled", True)
        self._host:      str   = e.get("smtp_host", "").strip()
        self._port:      int   = int(e.get("smtp_port", 587))
        self._starttls:  bool  = e.get("use_starttls", True)
        self._ssl:       bool  = e.get("use_ssl", False)
        self._username:  str   = (
            (e.get("username") or "").strip()
            or os.getenv("SMTP_USERNAME", "").strip()
        )
        self._password:  str   = (
            (e.get("password") or "").strip()
            or os.getenv("SMTP_PASSWORD", "").strip()
        )
        self._sender:    str   = e.get("sender", "").strip()
        self._recipients: List[str] = [
            r.strip() for r in e.get("recipients", []) if r.strip()
        ]
        self._cooldown:  float = float(e.get("alert_cooldown", 300))
        self._timeout:   int   = int(e.get("timeout", 30))

        # cooldown tracker: error_hash → last_sent_monotonic
        self._sent: dict = {}

        if self._enabled and self._host and self._sender and self._recipients:
            logger.info(
                "Email alerts enabled → %s:%d → %s",
                self._host, self._port, ", ".join(self._recipients),
            )
        elif self._enabled:
            logger.warning(
                "Email alerts enabled but smtp_host / sender / recipients are "
                "not fully configured — alerts will be skipped"
            )
        else:
            logger.info("Email alerts disabled")

    # ── Public API ────────────────────────────────────────────

    def send_alert(
        self,
        error,
        solution: str,
        screenshot_path: Optional[str] = None,
        kb_matches: Optional[list] = None,
    ) -> bool:
        """
        Send an HTML email alert for a detected error.

        Returns True on success, False if skipped or on error.
        """
        if not self._enabled:
            logger.debug("Email alert skipped — disabled in config")
            return False

        if not (self._host and self._sender and self._recipients):
            logger.debug("Email alert skipped — SMTP not fully configured")
            return False

        if self._in_cooldown(error):
            logger.debug("Email alert suppressed (cooldown active)")
            return False

        has_screenshot = bool(screenshot_path and os.path.exists(screenshot_path))
        subject = self._build_subject(error)
        html_body = self._build_html(
            error, solution, has_screenshot=has_screenshot, kb_matches=kb_matches
        )
        inline_images = [screenshot_path] if has_screenshot else []

        try:
            self._send(
                subject=subject,
                html_body=html_body,
                inline_image_paths=inline_images,
            )
            logger.info("Email alert sent to %s", ", ".join(self._recipients))
            return True
        except Exception as exc:
            logger.error("Failed to send email alert: %s", exc)
            try:
                from rich.console import Console as _Console
                _Console().print(f"  [red]✗ Email alert failed: {exc}[/red]")
            except Exception:
                print(f"Email alert failed: {exc}")
            return False

    def send_batch_alert(
        self,
        errors: list,
        solution: str,
        screenshot_path: Optional[str] = None,
        kb_matches: Optional[list] = None,
    ) -> bool:
        """
        Send a single grouped HTML email for a burst of consecutive errors.
        Uses the most severe error as the lead.
        """
        if not self._enabled:
            return False
        if not (self._host and self._sender and self._recipients):
            return False

        _rank = {"CRITICAL": 3, "ERROR": 2, "WARNING": 1}
        lead  = max(errors, key=lambda e: _rank.get(e.severity, 0))

        # Cooldown keyed on combined batch signature
        batch_key = hashlib.md5(
            "".join(e.message[:60] for e in errors).encode()
        ).hexdigest()
        now = time.monotonic()
        if (now - self._sent.get(batch_key, 0.0)) < self._cooldown:
            logger.debug("Email batch alert suppressed (cooldown)")
            return False
        self._sent[batch_key] = now

        has_screenshot = bool(screenshot_path and os.path.exists(screenshot_path))
        subject = (
            f"[{lead.severity}] Burst of {len(errors)} errors in "
            f"{Path(str(lead.file_path)).name} — "
            f"{lead.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        html_body = self._build_batch_html(
            errors, lead, solution,
            has_screenshot=has_screenshot, kb_matches=kb_matches,
        )
        inline_images = [screenshot_path] if has_screenshot else []

        try:
            self._send(
                subject=subject,
                html_body=html_body,
                inline_image_paths=inline_images,
            )
            logger.info(
                "Email batch alert (%d errors) sent to %s",
                len(errors), ", ".join(self._recipients),
            )
            return True
        except Exception as exc:
            logger.error("Failed to send batch email alert: %s", exc)
            try:
                from rich.console import Console as _Console
                _Console().print(f"  [red]✗ Email batch alert failed: {exc}[/red]")
            except Exception:
                print(f"Email batch alert failed: {exc}")
            return False

    # ── Email construction ────────────────────────────────────

    def _build_subject(self, error) -> str:
        fname = Path(str(error.file_path)).name
        return (
            f"[{error.severity}] Error detected in {fname} "
            f"— {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def _build_html(
        self,
        error,
        solution: str,
        has_screenshot: bool = False,
        kb_matches: Optional[list] = None,
    ) -> str:
        severity  = error.severity
        colour    = _SEVERITY_COLOUR.get(severity, "#808080")
        icon      = _SEVERITY_ICON.get(severity, "⚪")
        ts        = error.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        fname     = Path(str(error.file_path)).name
        full_path = str(error.file_path)
        line_num  = getattr(error, "line_number", 0)
        line_str  = str(line_num) if line_num else "—"

        ctx_lines = error.context[-5:] if error.context else []
        ctx_html  = ""
        if ctx_lines:
            escaped_ctx = (
                "\n".join(ctx_lines)
                .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            ctx_html = (
                "<h3 style='color:#555;margin-top:20px;'>📋 Log Context</h3>"
                "<pre style='white-space:pre-wrap;background:#f8f8f8;padding:10px;"
                "border:1px solid #ddd;font-size:12px;border-radius:4px;'>"
                f"{escaped_ctx}</pre>"
            )

        escaped_solution = (
            solution.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        escaped_message  = (
            error.message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

        kb_html = self._build_kb_html(kb_matches)

        # Screenshot placed INLINE — immediately after the error message, before the solution
        screenshot_html = ""
        if has_screenshot:
            screenshot_html = (
                f"<h3 style='color:#555;margin-top:16px;'>📸 Screenshot at Error Line {line_str}</h3>"
                f"<div style='margin:8px 0;border:2px solid {colour};border-radius:6px;"
                "padding:4px;display:inline-block;max-width:100%;'>"
                "<img src='{inline0}' alt='screenshot' "
                "style='max-width:800px;display:block;border-radius:4px;' /></div>"
            ).replace("{inline0}", "{{inline0}}")

        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head><meta charset=\"utf-8\"></head>\n"
            f"<body style=\"font-family:Arial,sans-serif;font-size:14px;color:#333;"
            "max-width:900px;margin:20px auto;padding:0 20px;\">\n\n"
            f"  <div style=\"background:{colour};color:#fff;padding:12px 18px;border-radius:6px 6px 0 0;\">\n"
            f"    <h2 style=\"margin:0;\">{icon} AI Log Error Detected — {severity}</h2>\n"
            "  </div>\n\n"
            "  <div style=\"border:1px solid #ddd;border-top:none;padding:18px;border-radius:0 0 6px 6px;\">\n\n"
            "    <table style=\"width:100%;border-collapse:collapse;margin-bottom:16px;\">\n"
            "      <tr><td style=\"padding:4px 8px;color:#777;width:130px;\">Severity</td>\n"
            f"          <td style=\"padding:4px 8px;\"><strong style=\"color:{colour};\">{icon} {severity}</strong></td></tr>\n"
            "      <tr style=\"background:#f9f9f9;\"><td style=\"padding:4px 8px;color:#777;\">File</td>\n"
            f"          <td style=\"padding:4px 8px;\">{fname}</td></tr>\n"
            "      <tr><td style=\"padding:4px 8px;color:#777;\">Full Path</td>\n"
            f"          <td style=\"padding:4px 8px;font-size:12px;\">{full_path}</td></tr>\n"
            "      <tr style=\"background:#f9f9f9;\"><td style=\"padding:4px 8px;color:#777;\">Line Number</td>\n"
            f"          <td style=\"padding:4px 8px;\"><strong style=\"color:{colour};\">{line_str}</strong></td></tr>\n"
            "      <tr><td style=\"padding:4px 8px;color:#777;\">Detected At</td>\n"
            f"          <td style=\"padding:4px 8px;\">{ts}</td></tr>\n"
            "    </table>\n\n"
            "    <h3 style=\"color:#555;margin-top:20px;\">⚠ Error Message</h3>\n"
            "    <pre style=\"white-space:pre-wrap;background:#fff3f3;padding:10px;border:1px solid #fcc;"
            f"font-size:13px;border-radius:4px;\">{escaped_message}</pre>\n\n"
            f"    {screenshot_html}\n\n"
            "    <h3 style=\"color:#555;margin-top:20px;\">🤖 AI-Generated Solution</h3>\n"
            "    <div style=\"background:#f0f8f0;padding:12px;border:1px solid #c3e6c3;border-radius:4px;"
            f"white-space:pre-wrap;font-size:13px;\">{escaped_solution}</div>\n\n"
            f"    {kb_html}\n"
            f"    {ctx_html}\n\n"
            "    <p style=\"margin-top:24px;color:#888;font-size:12px;\">\n"
            "      Regards,<br/>AI Log Error Detector (Automated Notifier)\n"
            "    </p>\n"
            "  </div>\n"
            "</body>\n"
            "</html>"
        )

    def _build_batch_html(
        self,
        errors: list,
        lead,
        solution: str,
        has_screenshot: bool = False,
        kb_matches: Optional[list] = None,
    ) -> str:
        severity  = lead.severity
        colour    = _SEVERITY_COLOUR.get(severity, "#808080")
        icon      = _SEVERITY_ICON.get(severity, "⚪")
        ts        = lead.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        fname     = Path(str(lead.file_path)).name

        escaped_solution = (
            solution.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

        error_rows = ""
        for i, e in enumerate(errors, 1):
            em = e.message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            ln = str(getattr(e, "line_number", 0) or "—")
            bg = "" if i % 2 else "background:#f9f9f9;"
            error_rows += (
                f"<tr style='{bg}'>"
                f"<td style='padding:4px 8px;color:#777;width:80px;vertical-align:top;'>#{i} {e.severity}</td>"
                f"<td style='padding:4px 8px;font-size:11px;color:#777;width:55px;vertical-align:top;'>line {ln}</td>"
                f"<td style='padding:4px 8px;font-size:12px;font-family:monospace;'>{em[:300]}</td></tr>"
            )

        kb_html = self._build_kb_html(kb_matches)

        # Screenshot placed INLINE — immediately after the error list, before the solution
        screenshot_html = ""
        if has_screenshot:
            screenshot_html = (
                f"<h3 style='color:#555;margin-top:16px;'>📸 Screenshot at Error Line</h3>"
                f"<div style='margin:8px 0;border:2px solid {colour};border-radius:6px;"
                "padding:4px;display:inline-block;max-width:100%;'>"
                "<img src='{inline0}' alt='screenshot' "
                "style='max-width:800px;display:block;border-radius:4px;' /></div>"
            ).replace("{inline0}", "{{inline0}}")

        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head><meta charset=\"utf-8\"></head>\n"
            f"<body style=\"font-family:Arial,sans-serif;font-size:14px;color:#333;"
            "max-width:900px;margin:20px auto;padding:0 20px;\">\n\n"
            f"  <div style=\"background:{colour};color:#fff;padding:12px 18px;border-radius:6px 6px 0 0;\">\n"
            f"    <h2 style=\"margin:0;\">{icon} Burst Alert — {len(errors)} errors in {fname}</h2>\n"
            "  </div>\n\n"
            "  <div style=\"border:1px solid #ddd;border-top:none;padding:18px;border-radius:0 0 6px 6px;\">\n\n"
            f"    <p style=\"margin-top:0;\">{len(errors)} consecutive errors detected with no INFO lines between them. "
            f"Highest severity: <strong style=\"color:{colour};\">{severity}</strong> &nbsp;|&nbsp; {ts}</p>\n\n"
            "    <h3 style=\"color:#555;\">⚠ Errors in Burst</h3>\n"
            "    <table style=\"width:100%;border-collapse:collapse;font-size:13px;\">\n"
            "      <thead><tr style=\"background:#eee;\">\n"
            "        <th style=\"padding:4px 8px;text-align:left;\">#</th>\n"
            "        <th style=\"padding:4px 8px;text-align:left;\">Line</th>\n"
            "        <th style=\"padding:4px 8px;text-align:left;\">Error Message</th>\n"
            "      </tr></thead>\n"
            f"      <tbody>{error_rows}</tbody>\n"
            "    </table>\n\n"
            f"    {screenshot_html}\n\n"
            "    <h3 style=\"color:#555;margin-top:20px;\">🤖 AI-Generated Unified Solution</h3>\n"
            "    <div style=\"background:#f0f8f0;padding:12px;border:1px solid #c3e6c3;border-radius:4px;"
            f"white-space:pre-wrap;font-size:13px;\">{escaped_solution}</div>\n\n"
            f"    {kb_html}\n\n"
            "    <p style=\"margin-top:24px;color:#888;font-size:12px;\">\n"
            "      Regards,<br/>AI Log Error Detector (Automated Notifier)\n"
            "    </p>\n"
            "  </div>\n"
            "</body>\n"
            "</html>"
        )

    @staticmethod
    def _build_kb_html(kb_matches: Optional[list]) -> str:
        """Render the Serena KB defect match table, if any matches exist."""
        defects = [m for m in (kb_matches or []) if m.get("channel") == "defects"]
        if not defects:
            return ""
        rows = ""
        for d in defects[:5]:
            did   = (d.get("defect_id") or "—").replace("&", "&amp;")
            prod  = (d.get("product") or "—").replace("&", "&amp;")
            found = (d.get("release_found") or "—").replace("&", "&amp;")
            fixed = (d.get("release_fixed") or "—").replace("&", "&amp;")
            title = (d.get("title") or "")[:120].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            url   = d.get("url") or ""
            title_cell = f'<a href="{url}" style="color:#0563C1;">{title}</a>' if url else title
            rows += (
                f"<tr>"
                f"<td style='padding:4px 8px;font-weight:bold;white-space:nowrap;'>{did}</td>"
                f"<td style='padding:4px 8px;'>{title_cell}</td>"
                f"<td style='padding:4px 8px;'>{prod}</td>"
                f"<td style='padding:4px 8px;'>{found}</td>"
                f"<td style='padding:4px 8px;'>{fixed}</td>"
                f"</tr>"
            )
        return (
            "<h3 style='color:#555;margin-top:20px;'>🔍 Matching Serena KB Defects</h3>"
            "<table style='width:100%;border-collapse:collapse;font-size:12px;border:1px solid #ddd;'>"
            "<thead><tr style='background:#eee;'>"
            "<th style='padding:4px 8px;text-align:left;'>Defect ID</th>"
            "<th style='padding:4px 8px;text-align:left;'>Title</th>"
            "<th style='padding:4px 8px;text-align:left;'>Product</th>"
            "<th style='padding:4px 8px;text-align:left;'>Found In</th>"
            "<th style='padding:4px 8px;text-align:left;'>Fixed In</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    # ── SMTP sending ──────────────────────────────────────────

    def _send(
        self,
        subject: str,
        html_body: str,
        inline_image_paths: List[str],
    ) -> None:
        # ── Correct MIME hierarchy (fixes ATT00001.htm in Outlook) ───────────
        #
        #   multipart/related      ← ROOT; binds HTML to its CID images
        #     multipart/alternative
        #       text/plain         ← plain-text fallback (stays inside alternative)
        #       text/html          ← HTML with cid: references
        #     <inline images>      ← attached directly to 'related', NOT to 'alternative'
        #
        # The previous structure nested 'related' INSIDE 'alternative', which
        # caused Outlook to extract the HTML as ATT00001.htm and strip the images.
        # ─────────────────────────────────────────────────────────────────────
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"]    = self._sender
        msg["To"]      = ", ".join(self._recipients)
        msg["Date"]    = formatdate(localtime=True)

        part_alt = MIMEMultipart("alternative")
        msg.attach(part_alt)

        part_alt.attach(MIMEText(
            "This message is formatted in HTML. "
            "Please open it in an HTML-capable mail client to see the full content "
            "including the inline screenshot.",
            "plain",
        ))

        # Inline images are embedded into the root 'related' part first,
        # then the adjusted HTML (with cid: refs) goes into 'alternative'.
        html_adjusted, _ = self._attach_inline_images(msg, html_body, inline_image_paths)
        part_alt.attach(MIMEText(html_adjusted, "html", "utf-8"))

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE

        smtp: Optional[smtplib.SMTP] = None
        try:
            if self._ssl:
                smtp = smtplib.SMTP_SSL(
                    self._host, self._port, timeout=self._timeout, context=ctx
                )
            else:
                smtp = smtplib.SMTP(self._host, self._port, timeout=self._timeout)

            smtp.ehlo()
            if self._starttls and not self._ssl:
                smtp.starttls(context=ctx)
                smtp.ehlo()
            if self._username and self._password:
                smtp.login(self._username, self._password)
            smtp.sendmail(self._sender, self._recipients, msg.as_string())
            smtp.quit()
        except Exception:
            if smtp is not None:
                try:
                    smtp.close()
                except Exception:
                    pass
            raise

    # ── Attachment helpers ────────────────────────────────────

    @staticmethod
    def _attach_file(msg: MIMEMultipart, filepath: str) -> None:
        """Attach a file as a regular (downloaded) attachment."""
        if not os.path.exists(filepath):
            logger.warning("Attachment not found, skipping: %s", filepath)
            return
        ctype, encoding = mimetypes.guess_type(filepath)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(filepath, "rb") as f:
            data = f.read()
        part = MIMEBase(maintype, subtype)
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{os.path.basename(filepath)}"',
        )
        msg.attach(part)

    @staticmethod
    def _attach_inline_images(
        msg_root: MIMEMultipart, html_body: str, image_paths: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Embed images inline via Content-ID.
        Replaces ``{{inline0}}``, ``{{inline1}}``, … placeholders in html_body
        with ``cid:<id>`` references.
        """
        cids: List[str] = []
        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                logger.warning("Inline image not found, skipping: %s", path)
                continue
            with open(path, "rb") as f:
                img_data = f.read()
            img = MIMEImage(img_data)
            cid = f"img{i}"
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header(
                "Content-Disposition", "inline",
                filename=os.path.basename(path),
            )
            msg_root.attach(img)
            cids.append(cid)

        for i, cid in enumerate(cids):
            html_body = html_body.replace(f"{{{{inline{i}}}}}", f"cid:{cid}")
        return html_body, cids

    # ── Cooldown helpers ──────────────────────────────────────

    def _in_cooldown(self, error) -> bool:
        key  = hashlib.md5(
            f"{error.file_path}:{error.message[:80]}".encode()
        ).hexdigest()
        now  = time.monotonic()
        last = self._sent.get(key, 0.0)
        if (now - last) < self._cooldown:
            return True
        self._sent[key] = now
        return False
