"""
Mobile Push Notification Dispatcher.
Supports Telegram, ntfy, Bark, Pushover, and Generic Webhooks.
"""

import json
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional

logger = logging.getLogger("factuality.notifier")


class Notifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channel = config.get("push_channel", "telegram").lower()

    def send(self, title: str, message: str) -> bool:
        """Dispatches notification to the configured channel."""
        if not self.channel:
            logger.warning("No push notification channel configured.")
            return False

        logger.info(f"Dispatching notification via [{self.channel}]...")

        if self.channel == "telegram":
            return self._send_telegram(title, message)
        elif self.channel == "ntfy":
            return self._send_ntfy(title, message)
        elif self.channel == "bark":
            return self._send_bark(title, message)
        elif self.channel == "pushover":
            return self._send_pushover(title, message)
        elif self.channel == "webhook":
            return self._send_webhook(title, message)
        else:
            logger.error(f"Unsupported push channel: {self.channel}")
            return False

    def _send_telegram(self, title: str, message: str) -> bool:
        tg_cfg = self.config.get("telegram", {})
        bot_token = tg_cfg.get("bot_token", "").strip()
        chat_id = tg_cfg.get("chat_id", "").strip()

        if not bot_token or not chat_id or "YOUR_" in bot_token:
            logger.warning("Telegram bot_token or chat_id not configured in config.json.")
            return False

        import html

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        # Format as Telegram HTML for clean bold headers and reliable parsing
        escaped_title = html.escape(title)
        escaped_body = html.escape(message)
        # Convert markdown headers ### to bold
        lines = []
        for line in escaped_body.split("\n"):
            if line.startswith("### "):
                lines.append(f"\n<b>{line[4:].strip()}</b>")
            elif line.startswith("## "):
                lines.append(f"\n<b>{line[3:].strip()}</b>")
            else:
                lines.append(line)

        html_text = f"<b>{escaped_title}</b>\n" + "\n".join(lines).strip()

        payload = {
            "chat_id": chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("ok"):
                    logger.info("Telegram notification sent successfully (HTML).")
                    return True
                else:
                    logger.error(f"Telegram API returned error: {result}")
        except Exception as e:
            logger.warning(f"Telegram HTML send failed ({e}), retrying plain text...")
            try:
                plain_payload = {
                    "chat_id": chat_id,
                    "text": f"[{title}]\n\n{message}",
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(plain_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    if result.get("ok"):
                        logger.info("Telegram notification sent (plain text).")
                        return True
            except Exception as retry_err:
                logger.error(f"Failed to send Telegram notification: {retry_err}")
                return False

        return False

    def _send_ntfy(self, title: str, message: str) -> bool:
        ntfy_cfg = self.config.get("ntfy", {})
        topic = ntfy_cfg.get("topic", "").strip()
        server = ntfy_cfg.get("server_url", "https://ntfy.sh").rstrip("/")

        if not topic or "YOUR_" in topic:
            logger.warning("ntfy topic not configured in config.json.")
            return False

        url = f"{server}/{topic}"
        headers = {
            "Title": title.encode("utf-8"),
            "Priority": "default",
            "Content-Type": "text/plain; charset=utf-8",
        }

        try:
            req = urllib.request.Request(
                url,
                data=message.encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.info(f"ntfy notification status: {resp.status}")
                return resp.status in (200, 201)
        except Exception as e:
            logger.error(f"Failed to send ntfy notification: {e}")
            return False

    def _send_bark(self, title: str, message: str) -> bool:
        bark_cfg = self.config.get("bark", {})
        device_key = bark_cfg.get("device_key", "").strip()
        server = bark_cfg.get("server_url", "https://api.day.app").rstrip("/")

        if not device_key or "YOUR_" in device_key:
            logger.warning("Bark device_key not configured.")
            return False

        url = f"{server}/push"
        payload = {
            "device_key": device_key,
            "title": title,
            "body": message,
            "group": "FactualityCheck",
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.info("Bark notification sent successfully.")
                return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to send Bark notification: {e}")
            return False

    def _send_pushover(self, title: str, message: str) -> bool:
        pushover_cfg = self.config.get("pushover", {})
        token = pushover_cfg.get("token", "").strip()
        user = pushover_cfg.get("user", "").strip()

        if not token or not user or "YOUR_" in token:
            logger.warning("Pushover token or user not configured.")
            return False

        data = urllib.parse.urlencode({
            "token": token,
            "user": user,
            "title": title,
            "message": message,
        }).encode("utf-8")

        try:
            req = urllib.request.Request("https://api.pushover.net/1/messages.json", data=data, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.info("Pushover notification sent successfully.")
                return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to send Pushover notification: {e}")
            return False

    def _send_webhook(self, title: str, message: str) -> bool:
        webhook_cfg = self.config.get("webhook", {})
        url = webhook_cfg.get("url", "").strip()

        if not url:
            logger.warning("Generic webhook URL not configured.")
            return False

        payload = {"title": title, "summary": message}

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.info("Webhook notification sent successfully.")
                return resp.status in (200, 204)
        except Exception as e:
            logger.error(f"Failed to send Webhook notification: {e}")
            return False
