import os
import requests


def notify(title: str, message: str) -> None:
    """Send a push notification via ntfy.sh.

    Topic is read from the NTFY_TOPIC environment variable.
    No-op (and never raises) if NTFY_TOPIC is unset or empty.
    """
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "high"},
            timeout=10,
        )
    except Exception:
        pass
