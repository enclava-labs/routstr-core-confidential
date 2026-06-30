import importlib.util
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_cap_image_wrapper():
    spec = importlib.util.spec_from_file_location(
        "cap_image_wrapper",
        ROOT / "scripts" / "cap_image_wrapper.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_build_context_generates_cap_wrapper_files(tmp_path: Path) -> None:
    module = load_cap_image_wrapper()
    config = module.WrapperConfig(
        source_image="example/app:1.0",
        target_image="example/app:cap",
        command=["/usr/local/bin/app", "--serve"],
        output_dir=tmp_path,
        app_name="example-app",
        mode="proxy",
        port=8000,
        upstream_port=9000,
    )

    module.write_build_context(config)

    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert "FROM ${SOURCE_IMAGE}" in dockerfile
    assert "COPY --from=cap-helper /cap-root/ /" in dockerfile
    assert "ENV CAP_MODE=proxy" in dockerfile
    assert "ENV CAP_UPSTREAM_PORT=9000" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/cap-wrap"]' in dockerfile
    assert "USER 10001:10001" in dockerfile

    assert json.loads((tmp_path / "cap-command.json").read_text()) == [
        "/usr/local/bin/app",
        "--serve",
    ]
    enclava = tomllib.loads((tmp_path / "enclava.toml").read_text())
    assert enclava["app"]["command"] == ["/usr/local/bin/cap-wrap"]
    assert enclava["app"]["port"] == 8000
    assert enclava["storage"]["paths"] == ["/app/data"]
    assert enclava["health"]["path"] == "/health"
    assert "The wrapper serves /health itself" in (
        tmp_path / "README.cap-wrapper.md"
    ).read_text()


def test_default_app_name_is_cap_safe() -> None:
    module = load_cap_image_wrapper()

    assert module.default_app_name("ghcr.io/Acme/My_App:latest") == "my-app"
    assert module.default_app_name("registry.example.com/ns/app@sha256:abc") == "app"


def test_parse_command_json_requires_string_array() -> None:
    module = load_cap_image_wrapper()

    assert module.parse_command_json('["/bin/app", "--flag"]') == [
        "/bin/app",
        "--flag",
    ]
    try:
        module.parse_command_json('{"cmd": "/bin/app"}')
    except ValueError as exc:
        assert "JSON string array" in str(exc)
    else:
        raise AssertionError("expected ValueError")
