"""
Logging configuration for Routstr.

CRITICAL LOG MESSAGES FOR USAGE STATISTICS:
===========================================
The following log messages are parsed by the usage tracking system
(routstr/core/usage_analytics_store.py and routstr/core/log_manager.py).
DO NOT modify or remove these messages without updating the usage tracking logic:

1. "Received proxy request" (INFO) - routstr/proxy.py
   - Used to count total incoming requests
   - Includes model information in context

2. "Calculated token-based cost" (INFO) - routstr/auth.py
   - Used to track successful completions and revenue
   - The 'token_cost', 'model', 'input_tokens', and 'output_tokens' fields are extracted for dashboard metrics

3. "Max cost payment finalized" (INFO) - routstr/auth.py
   - Used as the successful completion fallback when token usage is unavailable
   - The 'charged_amount', 'model', 'input_tokens', and 'output_tokens' fields are extracted for dashboard metrics

4. "Payment processed successfully" (INFO) - routstr/auth.py
   - Used to count successful payment processing events
   - Tracks payment-related metrics

5. "Upstream request failed, revert payment" (WARNING) - routstr/proxy.py
   - Used to track failed requests and refunds
   - The 'max_cost_for_model' field is extracted for refund calculation
   - Must include 'max_cost_for_model' in extra dict

6. Any ERROR level logs with "upstream" in the message
   - Used to count upstream provider errors
   - Helps identify service reliability issues

If you need to modify these messages, ensure you also update the parsing logic in:
- routstr/core/usage_analytics_store.py
- routstr/core/log_manager.py
"""

import hashlib
import logging.config
import logging.handlers
import os
import re
import sys
import tomllib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pythonjsonlogger import jsonlogger
from rich.console import Console
from rich.logging import RichHandler

# Only use RichHandler when stdout is a real TTY. In non-TTY contexts
# (docker logs, pipes, CI) Rich pads every line to width and wraps long
# records, producing visually-empty trailing whitespace and split records.
# A plain StreamHandler avoids both problems.
_stdout_is_tty = sys.stdout.isatty()
_console = Console(soft_wrap=True) if _stdout_is_tty else None

# Define custom TRACE level
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
_STANDARD_LOG_RECORD_ATTRS = {
    *logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys(),
    "asctime",
    "message",
}
_SENSITIVE_KEY_NAMES = {
    "authorization",
    "x-cashu",
    "bearer",
    "bearertoken",
    "token",
    "accesstoken",
    "access_token",
    "refreshtoken",
    "refresh_token",
    "key",
    "secret",
    "clientsecret",
    "client_secret",
    "password",
    "cashu_token",
    "bearer_key",
    "apikey",
    "api_key",
    "nsec",
    "upstream_api_key",
    "upstreamapikey",
    "refund_address",
    "refundaddress",
    "prompt",
    "raw_prompt",
    "rawprompt",
    "input",
    "content",
}


def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log with TRACE level"""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


# Add the trace method to Logger class
setattr(logging.Logger, "trace", trace)


def redact_url_userinfo(value: object) -> object:
    """Return URL-like strings without username/password userinfo."""
    if not isinstance(value, str):
        return value
    raw_url = value.strip()
    if not raw_url:
        return value
    try:
        parsed = urlsplit(raw_url)
    except Exception:
        return "<redacted-url>" if "@" in raw_url else value
    if "@" in parsed.netloc:
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        return urlunsplit(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    if "@" in raw_url and not parsed.netloc:
        return "<redacted-url>"
    return value


def credential_fingerprint(value: object, *, length: int = 16) -> str | None:
    """Return a stable non-secret fingerprint for credential correlation."""
    if not isinstance(value, str) or not value:
        return None
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:length]}"


def redact_sensitive_text(value: object) -> object:
    """Return text with common secret-bearing diagnostics redacted."""
    if not isinstance(value, str):
        return value

    text = re.sub(
        r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@]+@([^\s]+)",
        r"\1\2",
        value,
    )
    standalone_patterns = [
        r"Bearer\s+([a-zA-Z0-9_\-\.]{10,})",
        r"cashu[A-Z]+([a-zA-Z0-9_\-\.=/+]+)",
        r"nsec[a-z0-9]+",
    ]
    for pattern in standalone_patterns:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)

    for key in _SENSITIVE_KEY_NAMES:
        key_patterns = [
            rf'["\']({key})["\']\s*[:=]\s*["\']([^"\']+)["\']',
            rf'["\']({key})["\']\s*[:=]\s*([a-zA-Z0-9_\-\.=/+]+)',
            rf"({key})\s*[:=]\s*([a-zA-Z0-9_\-\.=/+]+)",
            rf'({key})\s*[:=]\s*["\']([^"\']+)["\']',
        ]
        for pattern in key_patterns:
            text = re.sub(
                pattern,
                lambda match: f"{match.group(1).lower()}: [REDACTED]",
                text,
                flags=re.IGNORECASE,
            )
    text = re.sub(
        r"\bsk-[a-zA-Z0-9_\-\.]{9,}\b",
        "[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _redact_structured_log_value(value: object, key: object | None = None) -> object:
    if isinstance(key, str) and key.lower() in _SENSITIVE_KEY_NAMES:
        return "[REDACTED]" if value is not None else None
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, Mapping):
        return {
            item_key: _redact_structured_log_value(item_value, item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_structured_log_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_structured_log_value(item) for item in value)
    return value


class DailyRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Custom TimedRotatingFileHandler that creates date-based filenames."""

    def __init__(self, filename: str, **kwargs: Any) -> None:
        """Initialize with a base filename pattern."""
        self.base_dir = os.path.dirname(filename)
        self.base_name = os.path.basename(filename).replace(".log", "")

        today = datetime.now().strftime("%Y-%m-%d")
        self.current_date = today
        dated_filename = os.path.join(self.base_dir, f"{self.base_name}_{today}.log")

        super().__init__(dated_filename, **kwargs)

    def doRollover(self) -> None:
        """Override rollover to create new date-based filename."""
        if self.stream:
            self.stream.close()

        new_date = datetime.now().strftime("%Y-%m-%d")
        new_filename = os.path.join(self.base_dir, f"{self.base_name}_{new_date}.log")

        self.baseFilename = new_filename
        self.current_date = new_date

        # FIX ME: not sure if we need this
        # self._cleanup_old_files()

        if not self.delay:
            self.stream = self._open()

    def _cleanup_old_files(self) -> None:
        """Remove old log files beyond backupCount."""
        if self.backupCount > 0:
            log_files = []
            if os.path.exists(self.base_dir):
                for file in os.listdir(self.base_dir):
                    if file.startswith(f"{self.base_name}_") and file.endswith(".log"):
                        file_path = os.path.join(self.base_dir, file)
                        log_files.append((file_path, os.path.getmtime(file_path)))

            log_files.sort(key=lambda x: x[1], reverse=True)

            for file_path, _ in log_files[self.backupCount :]:
                try:
                    os.remove(file_path)
                except OSError:
                    pass


def get_package_version() -> str:
    """Read the package version from pyproject.toml."""
    try:
        # Find project root by looking for pyproject.toml
        current_path = Path(__file__).parent
        while current_path != current_path.parent:
            pyproject_path = current_path / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomllib.load(f)
                version = pyproject_data.get("project", {}).get("version", "unknown")
                return version
            current_path = current_path.parent

        # Fallback: try the simple path resolution (3 levels up for routstr/logging/logging_config.py)
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
            version = pyproject_data.get("project", {}).get("version", "unknown")
            return version

        return "unknown"
    except Exception:
        return "unknown"


class VersionFilter(logging.Filter):
    """Filter to add package version to all log records."""

    def __init__(self) -> None:
        super().__init__()
        self.version = get_package_version()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add version information to the log record."""
        record.version = self.version
        return True


class RequestIdFilter(logging.Filter):
    """Filter to add request ID to all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request ID to the log record if available."""
        try:
            # Import here to avoid circular imports
            from .middleware import request_id_context

            request_id = request_id_context.get(None)
            record.request_id = request_id if request_id else "no-request-id"
        except ImportError:
            # If middleware isn't available yet, just use default
            record.request_id = "no-request-id"
        return True


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""

    SENSITIVE_KEYS = _SENSITIVE_KEY_NAMES

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out sensitive information from log records."""
        try:
            message = str(redact_sensitive_text(record.getMessage()))
            record.msg = message
            record.args = ()
            for attr, value in list(record.__dict__.items()):
                if attr in _STANDARD_LOG_RECORD_ATTRS:
                    continue
                setattr(record, attr, _redact_structured_log_value(value, attr))

        except Exception:
            pass

        return True


def get_log_level() -> str:
    """Get log level from environment variable."""
    try:
        from .settings import settings

        level = settings.log_level.upper()
    except Exception:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    # Validate log level - if invalid, default to INFO
    valid_levels = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        level = "INFO"
    return level


def should_enable_console_logging() -> bool:
    """Check if console logging should be enabled."""
    try:
        from .settings import settings

        return bool(settings.enable_console_logging)
    except Exception:
        return os.environ.get("ENABLE_CONSOLE_LOGGING", "true").lower() in (
            "true",
            "1",
            "yes",
        )


def get_log_dir() -> str:
    """Return the directory used for Routstr file logs."""
    return os.environ.get("ROUTSTR_LOG_DIR", "logs") or "logs"


def setup_logging() -> None:
    """Configure centralized logging for the application."""

    log_level = get_log_level()
    console_enabled = should_enable_console_logging()
    log_dir = get_log_dir()
    log_file = os.path.join(log_dir, "app.log")

    # Determine which handlers to use
    handlers = ["file"]
    if console_enabled:
        handlers.append("console")

    if _stdout_is_tty:
        console_handler: dict[str, Any] = {
            "()": RichHandler,
            "level": log_level,
            "show_time": False,
            "show_path": False,
            "rich_tracebacks": True,
            "markup": True,
            "console": _console,
            "filters": ["request_id_filter", "security_filter"],
        }
    else:
        console_handler = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "plain",
            "stream": "ext://sys.stdout",
            "filters": ["request_id_filter", "security_filter"],
        }

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": jsonlogger.JsonFormatter,
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d %(version)s %(request_id)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "plain": {
                "format": "%(asctime)s %(levelname)-7s %(name)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "filters": {
            "version_filter": {"()": VersionFilter},
            "request_id_filter": {"()": RequestIdFilter},
            "security_filter": {"()": SecurityFilter},
        },
        "handlers": {
            "console": console_handler,
            "file": {
                "()": DailyRotatingFileHandler,
                "level": log_level,
                "formatter": "json",
                "filename": log_file,
                "when": "midnight",  # Rotate at midnight each day
                "interval": 1,  # Every 1 day
                "backupCount": 30,  # Keep 30 days of logs
                "atTime": None,  # Rotate at midnight (00:00)
                "filters": ["version_filter", "request_id_filter", "security_filter"],
            },
        },
        "loggers": {
            "routstr": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "routstr.payment": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "routstr.proxy": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "routstr.auth": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "routstr.payment.models": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "routstr.core.exceptions": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "routstr.core.middleware": {
                "level": log_level,
                "handlers": ["file"],
                "propagate": False,
            },
            # Suppress verbose third-party logging
            "httpx": {
                "level": "WARNING",
                "handlers": ["console"] if console_enabled else [],
                "propagate": False,
            },
            "openai": {
                "level": "WARNING",
                "handlers": ["console"] if console_enabled else [],
                "propagate": False,
            },
            "httpcore": {
                "level": "WARNING",
                "handlers": ["console"] if console_enabled else [],
                "propagate": False,
            },
            "websockets": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "watchfiles.main": {"level": "WARNING", "handlers": [], "propagate": False},
            "aiosqlite": {"level": "ERROR", "handlers": [], "propagate": False},
            "alembic": {
                "level": "WARNING",
                "handlers": ["console"] if console_enabled else [],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"] if console_enabled else [],
        },
    }

    os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(name)
