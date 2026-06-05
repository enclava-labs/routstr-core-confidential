from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from importlib import util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"

GENERATOR_SPEC = util.spec_from_file_location(
    "generate_tinfoil_policy",
    SCRIPT_DIR / "generate_tinfoil_policy.py",
)
assert GENERATOR_SPEC is not None
assert GENERATOR_SPEC.loader is not None
generator = util.module_from_spec(GENERATOR_SPEC)
sys.modules[GENERATOR_SPEC.name] = generator
GENERATOR_SPEC.loader.exec_module(generator)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


ROUTER_DIGEST = _digest("tinfoil-router-release")
MODEL_DIGEST = _digest("tinfoil-kimi-release")
LLAMA_DIGEST = _digest("tinfoil-llama-release")
VERIFIER_DIGEST = _digest("tinfoil-verifier")


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _results(checks: list[dict[str, object]], **overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "last_run": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "checks": checks,
    }
    document.update(overrides)
    return document


def _targets(path: Path) -> Path:
    return _write_json(
        path,
        {
            "providers": [
                {
                    "slug": "tinfoil",
                    "checks": [
                        {
                            "id": "tinfoil-router",
                            "label": "Router",
                            "type": "tinfoil",
                            "url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                            "host": "inference.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                        {
                            "id": "tinfoil-llama",
                            "label": "Llama",
                            "type": "tinfoil",
                            "url": "https://llama3-3-70b.tinfoil.containers.tinfoil.dev/.well-known/tinfoil-attestation",
                            "host": "llama3-3-70b.tinfoil.containers.tinfoil.dev",
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6",
                            "type": "tinfoil",
                            "url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                        {
                            "id": "tinfoil-qwen3-vl",
                            "label": "Qwen3-VL",
                            "type": "tinfoil",
                            "url": "https://qwen3-vl-30b.inf10.tinfoil.sh/.well-known/tinfoil-attestation",
                            "host": "qwen3-vl-30b.inf10.tinfoil.sh",
                        },
                    ],
                }
            ]
        },
    )


def _verified_router(**overrides: object) -> dict[str, object]:
    check: dict[str, object] = {
        "provider": "tinfoil",
        "id": "tinfoil-router",
        "status": "verified",
        "tls_matches": True,
        "release_digest": ROUTER_DIGEST,
    }
    check.update(overrides)
    return check


def _verified_model(**overrides: object) -> dict[str, object]:
    check: dict[str, object] = {
        "provider": "tinfoil",
        "id": "tinfoil-kimi-k2-6",
        "status": "verified",
        "tls_matches": True,
        "sigstore_match": True,
        "sigstore_bundle_verified": True,
        "images_verified": True,
        "release_digest": MODEL_DIGEST,
    }
    check.update(overrides)
    return check


def _verified_llama(**overrides: object) -> dict[str, object]:
    check: dict[str, object] = {
        "provider": "tinfoil",
        "id": "tinfoil-llama",
        "status": "verified",
        "tls_matches": True,
        "sigstore_match": True,
        "sigstore_bundle_verified": True,
        "images_verified": True,
        "release_digest": LLAMA_DIGEST,
    }
    check.update(overrides)
    return check


def _verified_qwen3_vl(**overrides: object) -> dict[str, object]:
    check: dict[str, object] = {
        "provider": "tinfoil",
        "id": "tinfoil-qwen3-vl",
        "model": "qwen3-vl-30b",
        "status": "verified",
        "tls_matches": True,
        "sigstore_match": True,
        "sigstore_bundle_verified": True,
        "images_verified": True,
        "release_digest": _digest("tinfoil-qwen3-vl-release"),
    }
    check.update(overrides)
    return check


def _manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "verifiers": [
                {
                    "name": "tinfoil-ppq",
                    "modes": ["tinfoil", "ppq-private-tee"],
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
                    "slug": "tinfoil",
                    "models": models,
                }
            ]
        },
    )


def test_generate_tinfoil_policy_maps_selected_model_from_target_host(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_router(), _verified_model()]),
    )
    output_path = tmp_path / "tinfoil-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["enabled"] is True
    assert document["confidentiality"]["mode"] == "tinfoil"
    assert document["confidentiality"]["model_ids"] == ["tinfoil/kimi-k2-6"]
    policy = document["confidentiality"]["policy"]
    assert policy["transport_security"] == "ehbp"
    assert policy["expected_release_digest"] == ROUTER_DIGEST
    assert policy["verifier_command_digest"] == VERIFIER_DIGEST
    assert policy["model_attestation_targets"]["tinfoil/kimi-k2-6"] == {
        "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
        "host": "kimi-k2-6.inf13.tinfoil.sh",
        "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
        "expected_release_digest": MODEL_DIGEST,
    }


def test_generate_tinfoil_policy_refuses_runtime_unloadable_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_build_policy_document = generator.build_policy_document

    def build_invalid_policy_document(*args: object, **kwargs: object) -> dict[str, object]:
        document = original_build_policy_document(*args, **kwargs)
        document["confidentiality"]["mode"] = "future-tinfoil"
        return document

    monkeypatch.setattr(
        generator,
        "build_policy_document",
        build_invalid_policy_document,
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_router(), _verified_model()]),
    )
    output_path = tmp_path / "tinfoil-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert not output_path.exists()


def test_generate_tinfoil_policy_resolves_ambiguous_target_to_selected_alias(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_router(), _verified_llama()]),
    )
    output_path = tmp_path / "tinfoil-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/llama3-3-70b",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["model_ids"] == ["tinfoil/llama3-3-70b"]
    assert document["confidentiality"]["policy"]["model_attestation_targets"] == {
        "tinfoil/llama3-3-70b": {
            "attestation_url": "https://llama3-3-70b.tinfoil.containers.tinfoil.dev/.well-known/tinfoil-attestation",
            "host": "llama3-3-70b.tinfoil.containers.tinfoil.dev",
            "repo": "tinfoilsh/confidential-llama3-3-70b",
            "expected_release_digest": LLAMA_DIGEST,
        }
    }


def test_generate_tinfoil_policy_accepts_result_model_alias_for_selected_target(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results(
            [
                _verified_router(
                    status="failed",
                    tls_matches=False,
                    error=(
                        "Trust decision failed: TLS certificate binding to "
                        "attested report_data is false"
                    ),
                ),
                _verified_llama(model="llama3-3-70b"),
            ]
        ),
    )
    output_path = tmp_path / "tinfoil-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/llama-3.3-70b",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["model_ids"] == ["tinfoil/llama-3.3-70b"]
    assert document["confidentiality"]["policy"]["transport_security"] == "ehbp"


def test_generate_tinfoil_policy_accepts_qwen_a3b_catalog_alias(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_router(tls_matches=False), _verified_qwen3_vl()]),
    )
    output_path = tmp_path / "tinfoil-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/qwen3-vl-30b-a3b",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["model_ids"] == ["tinfoil/qwen3-vl-30b-a3b"]


def test_generate_tinfoil_policy_rejects_missing_model_release_digest(
    tmp_path: Path,
) -> None:
    model = _verified_model()
    model.pop("release_digest")
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), model]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "tinfoil/kimi-k2-6 release_digest" in str(exc)
    else:
        raise AssertionError("expected missing Tinfoil model release digest to fail")


def test_generate_tinfoil_policy_rejects_non_standard_json_constants(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        '{"last_run":"2026-06-03T00:00:00Z","checks":[],"bad":NaN}',
        encoding="utf-8",
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "must not contain NaN" in str(exc)
    else:
        raise AssertionError("expected non-standard Tinfoil JSON constant to fail")


def test_generate_tinfoil_policy_refuses_non_standard_json_output(
    monkeypatch,
) -> None:
    monkeypatch.setattr(generator, "generate_policy", lambda _args: {"bad": float("nan")})

    exit_code = generator.main(
        [
            "--attestation-results-json",
            "unused-results.json",
            "--attestation-targets-json",
            "unused-targets.json",
            "--model-id",
            "tinfoil/kimi-k2-6",
        ]
    )

    assert exit_code == 1


def test_generate_tinfoil_policy_rejects_model_absent_from_provider_catalog(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--provider-catalog-json",
            str(
                _provider_catalog(
                    tmp_path / "providers.json",
                    [{"slug": "llama3-3-70b"}],
                )
            ),
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "selected Tinfoil model tinfoil/kimi-k2-6 is not present" in str(exc)
    else:
        raise AssertionError("expected missing Tinfoil provider catalog model to fail")


def test_generate_tinfoil_policy_accepts_provider_catalog_model(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--provider-catalog-json",
            str(_provider_catalog(tmp_path / "providers.json", [{"id": "kimi-k2-6"}])),
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    document = generator.generate_policy(args)

    assert document["confidentiality"]["model_ids"] == ["tinfoil/kimi-k2-6"]


def test_generate_tinfoil_policy_rejects_conflicting_target_host_url(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "tinfoil",
                    "checks": [
                        {
                            "id": "tinfoil-router",
                            "type": "tinfoil",
                            "host": "inference.tinfoil.sh",
                            "url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "url": "https://llama3-3-70b.tinfoil.containers.tinfoil.dev/.well-known/tinfoil-attestation",
                            "model": "tinfoil/kimi-k2-6",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "Tinfoil attestation target host aliases must match" in str(exc)
    else:
        raise AssertionError("expected conflicting Tinfoil target host/url to fail")


def test_generate_tinfoil_policy_rejects_credential_bearing_base_url(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--base-url",
            "https://token@inference.tinfoil.sh/v1",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "base_url must not contain credentials" in str(exc)
    else:
        raise AssertionError("expected credential-bearing Tinfoil base_url to fail")


def test_generate_tinfoil_policy_rejects_non_tinfoil_base_url(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--base-url",
            "https://example.com/v1",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "base_url host must be inference.tinfoil.sh" in str(exc)
    else:
        raise AssertionError("expected non-Tinfoil base_url to fail")


def test_generate_tinfoil_policy_rejects_empty_verifier_command(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--verifier-command",
            "   ",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "verifier_command is empty" in str(exc)
    else:
        raise AssertionError("expected empty Tinfoil verifier command to fail")


def test_generate_tinfoil_policy_rejects_empty_router_repo(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--router-repo",
            "   ",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "router_repo must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected empty Tinfoil router repo to fail")


def test_generate_tinfoil_policy_rejects_router_repo_target_mismatch(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--router-repo",
            "attacker/conflicting-router",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "Tinfoil router_repo does not match attestation target" in str(exc)
    else:
        raise AssertionError("expected conflicting Tinfoil router repo to fail")


def test_generate_tinfoil_policy_accepts_attestation_url_target_alias(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "tinfoil",
                    "checks": [
                        {
                            "id": "tinfoil-router",
                            "type": "tinfoil",
                            "attestation_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "type": "tinfoil",
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "model": "tinfoil/kimi-k2-6",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    output_path = tmp_path / "tinfoil-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["policy"]["model_attestation_targets"][
        "tinfoil/kimi-k2-6"
    ]["attestation_url"] == (
        "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation"
    )


def test_generate_tinfoil_policy_rejects_conflicting_target_url_aliases(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "tinfoil",
                    "checks": [
                        {
                            "id": "tinfoil-router",
                            "type": "tinfoil",
                            "url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "type": "tinfoil",
                            "url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/other",
                            "model": "tinfoil/kimi-k2-6",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "Tinfoil attestation target URL aliases must match" in str(exc)
    else:
        raise AssertionError("expected conflicting Tinfoil target URL aliases to fail")


def test_generate_tinfoil_policy_rejects_target_model_host_mismatch(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "tinfoil",
                    "checks": [
                        {
                            "id": "tinfoil-router",
                            "type": "tinfoil",
                            "host": "inference.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "type": "tinfoil",
                            "host": "llama3-3-70b.tinfoil.containers.tinfoil.dev",
                            "model": "tinfoil/kimi-k2-6",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "known host/model mismatch" in str(exc)
    else:
        raise AssertionError("expected Tinfoil target model/host mismatch to fail")


def test_generate_tinfoil_policy_rejects_target_repo_host_mismatch(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "tinfoil",
                    "checks": [
                        {
                            "id": "tinfoil-router",
                            "type": "tinfoil",
                            "host": "inference.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "model": "tinfoil/kimi-k2-6",
                            "repo": "tinfoilsh/confidential-llama3-3-70b",
                        },
                    ],
                }
            ]
        },
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "known host/repo mismatch" in str(exc)
    else:
        raise AssertionError("expected Tinfoil target repo/host mismatch to fail")


def test_generate_tinfoil_policy_rejects_duplicate_selected_model_rows(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "duplicate Tinfoil result rows for selected model tinfoil/kimi-k2-6" in str(exc)
    else:
        raise AssertionError("expected duplicate Tinfoil selected model rows to fail")


def test_generate_tinfoil_policy_rejects_duplicate_requested_model_ids(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--model-id",
            " TINFOIL/KIMI-K2-6 ",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "duplicate --model-id value tinfoil/kimi-k2-6" in str(exc)
    else:
        raise AssertionError("expected duplicate Tinfoil model selectors to fail")


def test_generate_tinfoil_policy_rejects_unprefixed_requested_model_id(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "tinfoil",
                    "checks": [
                        {
                            "id": "tinfoil-router",
                            "type": "tinfoil",
                            "host": "inference.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                        {
                            "id": "tinfoil-custom",
                            "type": "tinfoil",
                            "host": "custom-model.tinfoil.example",
                            "model": "kimi-k2-6",
                            "repo": "tinfoilsh/confidential-custom-model",
                        },
                    ],
                }
            ]
        },
    )
    result = _verified_model(
        id="tinfoil-custom",
        model="kimi-k2-6",
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), result]),
                )
            ),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "Tinfoil selected models must use exact tinfoil/* model_ids" in str(exc)
    else:
        raise AssertionError("expected unprefixed Tinfoil model selector to fail")


def test_generate_tinfoil_policy_rejects_duplicate_router_rows(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results(
                        [
                            _verified_router(),
                            _verified_router(),
                            _verified_model(),
                        ]
                    ),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "duplicate Tinfoil router result rows" in str(exc)
    else:
        raise AssertionError("expected duplicate Tinfoil router rows to fail")


def test_generate_tinfoil_policy_rejects_tls_missing_in_default_mode(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    {
                        "checks": [
                            _verified_router(tls_matches=False),
                            _verified_model(),
                        ],
                        "last_run": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    },
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "Tinfoil router requires tls_matches=true" in str(exc)
    else:
        raise AssertionError("expected default Tinfoil TLS binding mode to fail")


def test_generate_tinfoil_policy_rejects_stale_attestation_results(
    tmp_path: Path,
) -> None:
    stale_timestamp = (datetime.now(UTC) - timedelta(seconds=120)).isoformat().replace(
        "+00:00",
        "Z",
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results(
                        [_verified_router(), _verified_model()],
                        last_run=stale_timestamp,
                    ),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--max-attestation-result-age-seconds",
            "60",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "attestation results are stale" in str(exc)
    else:
        raise AssertionError("expected stale Tinfoil attestation results to fail")


def test_generate_tinfoil_policy_rejects_stale_selected_model_row(
    tmp_path: Path,
) -> None:
    stale_timestamp = (datetime.now(UTC) - timedelta(seconds=120)).isoformat().replace(
        "+00:00",
        "Z",
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results(
                        [_verified_router(), _verified_model(ts=stale_timestamp)],
                    ),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--max-attestation-result-age-seconds",
            "60",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert (
            "selected Tinfoil model tinfoil/kimi-k2-6 attestation result is stale"
            in str(exc)
        )
    else:
        raise AssertionError("expected stale Tinfoil model row to fail")


def test_generate_tinfoil_policy_rejects_stale_router_row(
    tmp_path: Path,
) -> None:
    stale_timestamp = (datetime.now(UTC) - timedelta(seconds=120)).isoformat().replace(
        "+00:00",
        "Z",
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results(
                        [_verified_router(ts=stale_timestamp), _verified_model()],
                    ),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--max-attestation-result-age-seconds",
            "60",
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "Tinfoil router attestation result is stale" in str(exc)
    else:
        raise AssertionError("expected stale Tinfoil router row to fail")


def test_generate_tinfoil_policy_rejects_malformed_selected_model_row_timestamp(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(), _verified_model(ts="not-a-time")]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert (
            "selected Tinfoil model tinfoil/kimi-k2-6 attestation result timestamp "
            "is invalid"
        ) in str(exc)
    else:
        raise AssertionError("expected malformed Tinfoil model row timestamp to fail")


def test_generate_tinfoil_policy_rejects_malformed_router_row_timestamp(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_router(ts="not-a-time"), _verified_model()]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "Tinfoil router attestation result timestamp is invalid" in str(exc)
    else:
        raise AssertionError("expected malformed Tinfoil router row timestamp to fail")


def test_generate_tinfoil_policy_rejects_missing_transparency_boolean(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results(
                        [
                            _verified_router(),
                            _verified_model(sigstore_match=False),
                        ]
                    ),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "tinfoil/kimi-k2-6",
            "--transport-security",
            "ehbp",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "requires sigstore_match=true" in str(exc)
    else:
        raise AssertionError("expected incomplete Tinfoil transparency proof to fail")
