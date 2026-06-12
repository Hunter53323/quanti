"""
Alerting: WeChat Work webhook (primary) + console (fallback).
"""

import json
import urllib.request
from datetime import datetime
from enum import Enum

from quanti.config import settings


class AlertLevel(Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class Alerter:
    """Sends alerts via WeChat Work webhook and console."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or settings.WECHAT_WEBHOOK_URL

    def send(self, level: AlertLevel, title: str, body: str) -> None:
        """Send an alert through all configured channels."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Always log to console
        prefix = "[CRITICAL]" if level == AlertLevel.CRITICAL else "[WARNING]"
        print(f"{prefix} [{timestamp}] {title}: {body}")

        # WeChat Work for critical only
        if level == AlertLevel.CRITICAL and self.webhook_url:
            self._send_wechat(title, body)

    def _send_wechat(self, title: str, body: str) -> None:
        """Send a markdown message via WeChat Work webhook."""
        if not self.webhook_url:
            return

        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": (
                    f"## {title}\n"
                    f"> {body}\n\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ),
            },
        }

        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[ALERT] Failed to send WeChat alert: {e}")


# Global singleton
_alerter = Alerter()


def get_alerter() -> Alerter:
    return _alerter


def alert(level: AlertLevel, title: str, body: str) -> None:
    """Convenience function for one-line alerts."""
    _alerter.send(level, title, body)
