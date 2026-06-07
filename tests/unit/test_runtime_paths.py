from pathlib import Path

from routstr.core.runtime_paths import (
    CAP_APP_DATA_DIR,
    DEFAULT_EPHEMERAL_LOG_DIR,
    resolve_database_url,
    resolve_log_dir,
)


def test_uses_cap_app_data_bind_mount_for_default_sqlite_but_ephemeral_logs(
    tmp_path: Path,
) -> None:
    cap_state_data_dir = tmp_path / "state" / "app-data"
    cap_state_data_dir.mkdir(parents=True)

    assert (
        resolve_database_url(
            {},
            cap_state_data_dir=cap_state_data_dir,
            cap_app_data_dir=CAP_APP_DATA_DIR,
        )
        == f"sqlite+aiosqlite:///{CAP_APP_DATA_DIR / 'keys.db'}"
    )
    assert resolve_log_dir({}, cap_state_data_dir=cap_state_data_dir) == (
        DEFAULT_EPHEMERAL_LOG_DIR
    )


def test_explicit_database_and_log_env_wins(tmp_path: Path) -> None:
    cap_state_data_dir = tmp_path / "state" / "app-data"
    cap_state_data_dir.mkdir(parents=True)

    assert (
        resolve_database_url(
            {"DATABASE_URL": "sqlite+aiosqlite:////custom/keys.db"},
            cap_state_data_dir=cap_state_data_dir,
        )
        == "sqlite+aiosqlite:////custom/keys.db"
    )
    assert resolve_log_dir(
        {"ROUTSTR_LOG_DIR": "/custom/logs"},
        cap_state_data_dir=cap_state_data_dir,
    ) == Path("/custom/logs")
