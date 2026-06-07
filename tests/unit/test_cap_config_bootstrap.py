from pathlib import Path

from routstr.core.cap_config import (
    configured_api_key,
    configured_api_key_balance_msats,
    read_cap_config_json,
    read_cap_config_text,
)


def test_reads_cap_config_files(tmp_path: Path) -> None:
    config_dir = tmp_path / ".enclava" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ADMIN_PASSWORD").write_text("from-file\n", encoding="utf-8")
    (config_dir / "TINFOIL_PROVIDER_SETTINGS_JSON").write_text(
        '{"confidentiality":{"enabled":true}}',
        encoding="utf-8",
    )

    assert (
        read_cap_config_text("ADMIN_PASSWORD", {}, config_dir=config_dir) == "from-file"
    )
    assert read_cap_config_json(
        "TINFOIL_PROVIDER_SETTINGS_JSON", {}, config_dir=config_dir
    ) == {"confidentiality": {"enabled": True}}


def test_environment_overrides_cap_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / ".enclava" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ADMIN_PASSWORD").write_text("from-file", encoding="utf-8")

    assert (
        read_cap_config_text(
            "ADMIN_PASSWORD",
            {"ADMIN_PASSWORD": "from-env"},
            config_dir=config_dir,
        )
        == "from-env"
    )


def test_configured_api_key_strips_openai_prefix(tmp_path: Path) -> None:
    config_dir = tmp_path / ".enclava" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ROUTSTR_API_KEY").write_text("sk-test-key", encoding="utf-8")
    (config_dir / "ROUTSTR_API_KEY_BALANCE_MSATS").write_text("12345", encoding="utf-8")

    assert configured_api_key({}, config_dir=config_dir) == "test-key"
    assert configured_api_key_balance_msats({}, config_dir=config_dir) == 12345


def test_resolve_bootstrap_uses_cap_config_admin_password(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / ".enclava" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ADMIN_PASSWORD").write_text("from-cap-config", encoding="utf-8")
    monkeypatch.setenv("ROUTSTR_CAP_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    from routstr.core.settings import resolve_bootstrap

    assert resolve_bootstrap().admin_password == "from-cap-config"
