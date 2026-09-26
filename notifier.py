"""
Mobile Push Notification Dispatcher.
Supports Telegram, ntfy, Bark, Pushover, and Generic Webhooks.
Includes robust SSL and retry handling for VPN/proxy environments.
"""

import re
import json
import logging
import html
import ssl
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    import certifi
    CA_FILE = certifi.where()
except ImportError:
    CA_FILE = None

logger = logging.getLogger("factuality.notifier")


def _get_ssl_context():
    """Returns an SSL context that handles self-signed proxy certs gracefully."""
    try:
        if CA_FILE:
            return ssl.create_default_context(cafile=CA_FILE)
        return ssl.create_default_context()
    except Exception:
        return ssl._create_unverified_context()


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
        chat_id = str(tg_cfg.get("chat_id", "")).strip()

        if not bot_token or not chat_id or "YOUR_" in bot_token:
            logger.warning("Telegram bot_token or chat_id not configured in config.json.")
            return False

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        # Preserve the two dimensions separated by a blank line
        clean_body = re.sub(r"\r\n", "\n", message).strip()
        clean_body = re.sub(r"\n{3,}", "\n\n", clean_body)

        escaped_title = html.escape(title)
        escaped_body = html.escape(clean_body)

        # Single paragraph delivery
        html_text = f"<b>{escaped_title}</b>\n{escaped_body}".strip()
        plain_text = f"[{title}]\n{clean_body}".strip()

        # 1. Prefer requests library (handles system proxies & certs cleanly)
        if requests is not None:
            # Try HTML first
            for parse_mode, text in [("HTML", html_text), (None, plain_text)]:
                payload = {
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                }
                if parse_mode:
                    payload["parse_mode"] = parse_mode

                try:
                    resp = requests.post(url, json=payload, timeout=15)
                    if resp.status_code == 200 and resp.json().get("ok"):
                        logger.info(f"Telegram notification sent successfully (mode: {parse_mode or 'plain'}).")
                        return True
                    else:
                        logger.warning(f"Telegram requests.post returned {resp.status_code}: {resp.text}")
                except Exception as req_err:
                    logger.warning(f"Telegram requests.post error ({req_err}), trying with verify=False...")
                    try:
                        resp = requests.post(url, json=payload, verify=False, timeout=15)
                        if resp.status_code == 200 and resp.json().get("ok"):
                            logger.info("Telegram notification sent successfully (unverified SSL).")
                            return True
                    except Exception as retry_err:
                        logger.warning(f"Telegram retry failed: {retry_err}")

        # 2. Fallback to urllib with safe SSL context
        return self._send_telegram_urllib(url, chat_id, title, plain_text, html_text)

    def _send_telegram_urllib(self, url: str, chat_id: str, title: str, plain_text: str, html_text: str) -> bool:
        for parse_mode, text in [("HTML", html_text), (None, plain_text)]:
            payload = {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # Try default SSL context, then unverified SSL context if VPN/proxy MITM blocks
            for ctx in [_get_ssl_context(), ssl._create_unverified_context()]:
                try:
                    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                        if result.get("ok"):
                            logger.info("Telegram notification sent successfully via urllib.")
                            return True
                except Exception as e:
                    logger.warning(f"Urllib Telegram send failed ({e}), trying next fallback...")

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

        if requests is not None:
            try:
                resp = requests.post(url, data=message.encode("utf-8"), headers={"Title": title}, timeout=15)
                return resp.status_code in (200, 201)
            except Exception as e:
                logger.warning(f"ntfy via requests failed ({e}), trying urllib...")

        try:
            req = urllib.request.Request(url, data=message.encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15, context=_get_ssl_context()) as resp:
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

        if requests is not None:
            try:
                resp = requests.post(url, json=payload, timeout=15)
                return resp.status_code == 200
            except Exception:
                pass

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15, context=_get_ssl_context()) as resp:
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

        data = {"token": token, "user": user, "title": title, "message": message}

        if requests is not None:
            try:
                resp = requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=15)
                return resp.status_code == 200
            except Exception:
                pass

        try:
            encoded_data = urllib.parse.urlencode(data).encode("utf-8")
            req = urllib.request.Request("https://api.pushover.net/1/messages.json", data=encoded_data, method="POST")
            with urllib.request.urlopen(req, timeout=15, context=_get_ssl_context()) as resp:
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

        if requests is not None:
            try:
                resp = requests.post(url, json=payload, timeout=15)
                return resp.status_code in (200, 204)
            except Exception:
                pass

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15, context=_get_ssl_context()) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            logger.error(f"Failed to send Webhook notification: {e}")
            return False
