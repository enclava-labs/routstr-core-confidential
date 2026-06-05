from __future__ import annotations

import hashlib
import json
import sys
from importlib import util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"

GENERATOR_SPEC = util.spec_from_file_location(
    "generate_privatemode_policy",
    SCRIPT_DIR / "generate_privatemode_policy.py",
)
assert GENERATOR_SPEC is not None
assert GENERATOR_SPEC.loader is not None
generator = util.module_from_spec(GENERATOR_SPEC)
sys.modules[GENERATOR_SPEC.name] = generator
GENERATOR_SPEC.loader.exec_module(generator)

PREFLIGHT_SPEC = util.spec_from_file_location(
    "confidential_routing_preflight",
    SCRIPT_DIR / "confidential_routing_preflight.py",
)
assert PREFLIGHT_SPEC is not None
assert PREFLIGHT_SPEC.loader is not None
preflight = util.module_from_spec(PREFLIGHT_SPEC)
sys.modules[PREFLIGHT_SPEC.name] = preflight
PREFLIGHT_SPEC.loader.exec_module(preflight)

PLACEHOLDER_DIGEST = "sha256:" + ("5" * 64)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


MANIFEST_DIGEST = _digest("privatemode-manifest")
VERIFIER_DIGEST = _digest("privatemode-verifier")
COORDINATOR_DIGEST = _digest("privatemode-coordinator")
SECRET_SERVICE_DIGEST = _digest("privatemode-secret-service")
AI_WORKER_DIGEST = _digest("privatemode-ai-worker")
KEY_RELEASE_DIGEST = _digest("privatemode-key-release")


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _proxy_binary(path: Path, contents: bytes = b"privatemode-proxy") -> Path:
    path.write_bytes(contents)
    return path


def _manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "verifiers": [
                {
                    "name": "privatemode",
                    "modes": ["privatemode"],
                    "sha256": VERIFIER_DIGEST,
                }
            ]
        },
    )


def _provider_catalog(path: Path, models: list[dict[str, object]]) -> Path:
    return _write_json(
        path,
        {
            "providers": [
                {
                    "slug": "privatemode",
                    "models": models,
                }
            ]
        },
    )


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--model-id",
        "privatemode/gpt-oss-120b",
        "--manifest-digest",
        MANIFEST_DIGEST,
        "--proxy-binary-path",
        str(_proxy_binary(tmp_path / "privatemode-proxy")),
        "--verifier-manifest-json",
        str(_manifest(tmp_path / "manifest.json")),
        "--expected-coordinator-measurement",
        COORDINATOR_DIGEST,
        "--expected-secret-service-measurement",
        SECRET_SERVICE_DIGEST,
        "--expected-ai-worker-measurement",
        AI_WORKER_DIGEST,
        "--expected-gpu-attestation-policy",
        "nvidia-ocsp-good-only",
        "--expected-key-release-binding",
        KEY_RELEASE_DIGEST,
        "--workload-san",
        "gpt-oss-120b.default.svc.cluster.local",
    ]


def test_generate_privatemode_policy_emits_strict_preflight_valid_policy(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "privatemode-policy.json"

    exit_code = generator.main(
        [
            *_base_args(tmp_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    policy = document["confidentiality"]["policy"]
    expected_proxy_digest = _digest("privatemode-proxy")
    assert document["base_url"] == "http://127.0.0.1:8080/v1"
    assert document["confidentiality"]["enabled"] is True
    assert document["confidentiality"]["mode"] == "privatemode"
    assert document["confidentiality"]["model_ids"] == ["privatemode/gpt-oss-120b"]
    assert policy["manifest_digest"] == MANIFEST_DIGEST
    assert policy["proxy_binary_digest"] == expected_proxy_digest
    assert policy["dump_requests"] is False
    assert policy["shared_prompt_cache"] is False
    assert policy["expected_trust_tier"] == "app-e2ee"
    assert policy["nvidia_ocsp_allow_unknown"] is False
    assert policy["nvidia_ocsp_revoked_grace_period_hours"] == 0
    assert policy["model_workload_bindings"] == {
        "privatemode/gpt-oss-120b": {
            "workload_sans": ["gpt-oss-120b.default.svc.cluster.local"]
        }
    }

    validation = preflight.validate_provider_policy(output_path)

    assert validation["ok"] is True
    assert validation["issues"] == []


def test_generate_privatemode_policy_refuses_runtime_unloadable_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_build_policy_document = generator.build_policy_document

    def build_invalid_policy_document(*args: object, **kwargs: object) -> dict[str, object]:
        document = original_build_policy_document(*args, **kwargs)
        document["confidentiality"]["mode"] = "future-privatemode"
        return document

    monkeypatch.setattr(
        generator,
        "build_policy_document",
        build_invalid_policy_document,
    )
    output_path = tmp_path / "privatemode-policy.json"

    exit_code = generator.main(
        [
            *_base_args(tmp_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert not output_path.exists()


def test_generate_privatemode_policy_rejects_placeholder_digest(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--manifest-digest",
            PLACEHOLDER_DIGEST,
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "must not be a placeholder digest" in str(exc)
    else:
        raise AssertionError("expected placeholder digest to fail")


def test_generate_privatemode_policy_rejects_gpu_policy_not_emitted_by_verifier(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--expected-gpu-attestation-policy",
            "strict-ocsp",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "expected_gpu_attestation_policy must be nvidia-ocsp-good-only" in str(
            exc
        )
    else:
        raise AssertionError("expected unsupported GPU policy label to fail")


def test_generate_privatemode_policy_rejects_model_absent_from_provider_catalog(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--provider-catalog-json",
            str(
                _provider_catalog(
                    tmp_path / "providers.json",
                    [{"slug": "qwen3-embedding-4b"}],
                )
            ),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "selected Privatemode model privatemode/gpt-oss-120b is not present" in str(exc)
    else:
        raise AssertionError("expected missing Privatemode provider catalog model to fail")


def test_generate_privatemode_policy_rejects_non_standard_json_constants(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "providers.json"
    catalog_path.write_text(
        '{"providers":[{"slug":"privatemode","models":[]}],"bad":NaN}',
        encoding="utf-8",
    )
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--provider-catalog-json",
            str(catalog_path),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "must not contain NaN" in str(exc)
    else:
        raise AssertionError("expected non-standard Privatemode JSON constant to fail")


def test_generate_privatemode_policy_refuses_non_standard_json_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generator, "generate_policy", lambda _args: {"bad": float("nan")})

    exit_code = generator.main(_base_args(tmp_path))

    assert exit_code == 1


def test_generate_privatemode_policy_accepts_provider_catalog_model(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--provider-catalog-json",
            str(
                _provider_catalog(
                    tmp_path / "providers.json",
                    [{"id": "gpt-oss-120b"}],
                )
            ),
        ]
    )

    document = generator.generate_policy(args)

    assert document["confidentiality"]["model_ids"] == ["privatemode/gpt-oss-120b"]


def test_generate_privatemode_policy_rejects_non_loopback_proxy_url(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--base-url",
            "https://api.privatemode.ai/v1",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "base_url must be a loopback URL" in str(exc)
    else:
        raise AssertionError("expected non-loopback Privatemode URL to fail")


def test_generate_privatemode_policy_rejects_invalid_proxy_url_port(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--base-url",
            "http://127.0.0.1:bad/v1",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "base_url must include a valid port" in str(exc)
    else:
        raise AssertionError("expected invalid Privatemode proxy URL port to fail")


def test_generate_privatemode_policy_rejects_empty_verifier_command(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--verifier-command",
            "   ",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "verifier_command is empty" in str(exc)
    else:
        raise AssertionError("expected empty Privatemode verifier command to fail")


def test_generate_privatemode_policy_rejects_blank_manifest_path_fields(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--manifest-path",
            "   ",
            "--manifest-log-dir",
            "   ",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "manifest_path must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected blank Privatemode manifest paths to fail")


def test_generate_privatemode_policy_rejects_missing_workload_selectors(
    tmp_path: Path,
) -> None:
    base_args = _base_args(tmp_path)
    workload_index = base_args.index("--workload-san")
    args = generator.parse_args(
        base_args[:workload_index] + base_args[workload_index + 2 :]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "at least one --workload-san or --workload-id is required" in str(exc)
    else:
        raise AssertionError("expected missing workload selector to fail")


def test_generate_privatemode_policy_requires_explicit_multi_model_bindings(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--model-id",
            "privatemode/kimi-k2-6",
            "--workload-san",
            "kimi-k2-6.default.svc.cluster.local",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "explicit --model-workload" in str(exc)
    else:
        raise AssertionError("expected implicit multi-model binding to fail")


def test_generate_privatemode_policy_rejects_binding_outside_expected_workloads(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            *_base_args(tmp_path),
            "--model-workload-san",
            "privatemode/gpt-oss-120b=unlisted.default.svc.cluster.local",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "not listed in expected_workload_sans" in str(exc)
    else:
        raise AssertionError("expected unlisted workload binding to fail")
