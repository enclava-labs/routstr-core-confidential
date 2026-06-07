from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

CAP_STATE_DATA_DIR = Path("/state/app-data")
DEFAULT_EPHEMERAL_LOG_DIR = Path("/tmp/routstr/logs")
LEGACY_SQLITE_URL = "sqlite+aiosqlite:///keys.db"
LEGACY_LOG_DIR = Path("logs")


def _env_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_data_dir(
    environ: Mapping[str, str] | None = None,
    *,
    cap_state_data_dir: Path = CAP_STATE_DATA_DIR,
) -> Path:
    """Return Routstr's durable data directory for this runtime."""
    env = os.environ if environ is None else environ
    explicit = _env_value(env, "ROUTSTR_DATA_DIR")
    if explicit:
        return Path(explicit)
    if cap_state_data_dir.is_dir():
        return cap_state_data_dir
    return Path(".")


def resolve_database_url(
    environ: Mapping[str, str] | None = None,
    *,
    cap_state_data_dir: Path = CAP_STATE_DATA_DIR,
) -> str:
    """Return the SQLAlchemy database URL for Routstr state."""
    env = os.environ if environ is None else environ
    explicit = _env_value(env, "DATABASE_URL")
    if explicit:
        return explicit

    data_dir = resolve_data_dir(env, cap_state_data_dir=cap_state_data_dir)
    if data_dir == Path("."):
        return LEGACY_SQLITE_URL
    return f"sqlite+aiosqlite:///{data_dir / 'keys.db'}"


def resolve_log_dir(
    environ: Mapping[str, str] | None = None,
    *,
    cap_state_data_dir: Path = CAP_STATE_DATA_DIR,
) -> Path:
    """Return the directory used for Routstr file logs.

    CAP deployments keep application state durable under /state/app-data, but
    request logging is synchronous. Keep default logs off the encrypted state
    mount so state I/O stalls cannot block every app route before handlers run.
    Operators can still opt into durable logs with ROUTSTR_LOG_DIR.
    """
    env = os.environ if environ is None else environ
    explicit = _env_value(env, "ROUTSTR_LOG_DIR")
    if explicit:
        return Path(explicit)

    if cap_state_data_dir.is_dir() or _env_value(env, "ROUTSTR_DATA_DIR"):
        return DEFAULT_EPHEMERAL_LOG_DIR
    if resolve_data_dir(env, cap_state_data_dir=cap_state_data_dir) == Path("."):
        return LEGACY_LOG_DIR
    return DEFAULT_EPHEMERAL_LOG_DIR
