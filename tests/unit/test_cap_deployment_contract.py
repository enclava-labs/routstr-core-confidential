import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_enclava_toml_matches_cap_restricted_runtime_contract() -> None:
    config = tomllib.loads((ROOT / "enclava.toml").read_text())

    assert config["app"]["port"] == 8000
    assert config["app"]["command"] == ["/usr/local/bin/routstr-cap-entrypoint"]
    assert config["storage"]["paths"] == ["/app/data"]
    assert config["unlock"]["mode"] == "auto"

    egress_hosts = {rule["host"] for rule in config["egress"]["allow"]}
    assert "inference.tinfoil.sh" in egress_hosts
    assert "api.ppq.ai" in egress_hosts
    assert "api.github.com" in egress_hosts
    assert "rekor.sigstore.dev" in egress_hosts


def test_dockerfile_defaults_mutable_paths_to_cap_storage_mount() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    required = [
        "ENV ROUTSTR_DATA_DIR=/app/data",
        "ENV DATABASE_URL=sqlite+aiosqlite:////app/data/keys.db",
        "ENV ROUTSTR_LOG_DIR=/app/data/logs",
        "ENV HOME=/app/data",
        "ENV TMPDIR=/app/data/tmp",
        "ENV XDG_CACHE_HOME=/app/data/.cache",
        "ENV XDG_CONFIG_HOME=/app/data/.config",
        "ENV XDG_DATA_HOME=/app/data/.local/share",
        "ENV PYTHONDONTWRITEBYTECODE=1",
        "ENV CONFIDENTIAL_ROUTING_MODE=required",
        "ENV ROUTSTR_TEE_ATTESTATION_REQUIRED=true",
        "COPY docker/routstr-cap-entrypoint /usr/local/bin/routstr-cap-entrypoint",
        "USER 10001:10001",
        'CMD ["/usr/local/bin/routstr-cap-entrypoint"]',
    ]
    for text in required:
        assert text in dockerfile

    assert " git " not in dockerfile


def test_cap_entrypoint_creates_only_state_backed_runtime_dirs() -> None:
    entrypoint = (ROOT / "docker" / "routstr-cap-entrypoint").read_text()

    assert ": \"${ROUTSTR_DATA_DIR:=/app/data}\"" in entrypoint
    assert ": \"${ROUTSTR_LOG_DIR:=${ROUTSTR_DATA_DIR}/logs}\"" in entrypoint
    assert "mkdir -p" in entrypoint
    assert "exec /.venv/bin/fastapi run routstr" in entrypoint
    assert "TMPDIR:=/tmp" not in entrypoint
