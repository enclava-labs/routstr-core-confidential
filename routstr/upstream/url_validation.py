from __future__ import annotations

from urllib.parse import urlparse


def https_url_violation(value: object, label: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{label} must be a non-empty string"
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return f"{label} must be an absolute URL"
    if not parsed.scheme or not parsed.netloc:
        return f"{label} must be an absolute URL"
    if parsed.username or parsed.password or "@" in parsed.netloc:
        return f"{label} must not include credentials"
    if parsed.scheme != "https":
        return f"{label} must use https"
    if not parsed.hostname:
        return f"{label} must include a host"
    try:
        parsed.port
    except ValueError:
        return f"{label} must include a valid port"
    return None


def https_origin_tuple(value: str) -> tuple[str, str, int]:
    parsed = urlparse(value.strip())
    port = parsed.port if parsed.port is not None else 443
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), port
