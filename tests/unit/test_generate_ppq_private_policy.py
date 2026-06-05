from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from importlib import util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"

GENERATOR_SPEC = util.spec_from_file_location(
    "generate_ppq_private_policy",
    SCRIPT_DIR / "generate_ppq_private_policy.py",
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


ROUTER_DIGEST = _digest("ppq-router-release")
BACKEND_DIGEST = _digest("ppq-backend-release")
PROXY_DIGEST = _digest("ppq-proxy")
VERIFIER_DIGEST = _digest("ppq-verifier")


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


def _proxy_binary(path: Path, contents: bytes = b"ppq-proxy") -> Path:
    path.write_bytes(contents)
    return path


def _verified_ppq_check(**overrides: object) -> dict[str, object]:
    check: dict[str, object] = {
        "provider": "ppq",
        "host": "api.ppq.ai",
        "model": "private/gpt-oss-120b",
        "status": "verified",
        "release_digest": ROUTER_DIGEST,
        "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
        "backend_release_digest": BACKEND_DIGEST,
        "backend_tls_matches": True,
        "backend_sigstore_match": True,
        "backend_sigstore_bundle_verified": True,
        "backend_images_verified": True,
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


def _targets(path: Path) -> Path:
    return _write_json(
        path,
        {
            "providers": [
                {
                    "slug": "ppq",
                    "checks": [
                        {
                            "id": "ppq-gpt-oss-120b",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                            "repo": "tinfoilsh/confidential-model-router",
                            "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                        }
                    ],
                }
            ]
        },
    )


def test_generate_ppq_policy_does_not_require_proxy_binary_identity(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )
    output_path = tmp_path / "ppq-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    policy = document["confidentiality"]["policy"]
    assert "proxy_binary_digest" not in policy
    assert "proxy_binary_path" not in policy


def test_generate_ppq_policy_ignores_legacy_missing_proxy_binary_path(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(tmp_path / "missing-ppq-proxy"),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    document = generator.generate_policy(args)
    policy = document["confidentiality"]["policy"]

    assert "proxy_binary_digest" not in policy
    assert "proxy_binary_path" not in policy


def test_generate_ppq_policy_ignores_legacy_pinned_proxy_binary_digest(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )
    output_path = tmp_path / "ppq-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-digest",
            PROXY_DIGEST,
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    policy = document["confidentiality"]["policy"]
    assert "proxy_binary_digest" not in policy
    assert "proxy_binary_path" not in policy

    results = preflight.load_attestation_results_directory(results_path)
    validation = preflight.validate_provider_policy(
        output_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert validation["ok"] is True
    assert validation["issues"] == []


def test_generate_ppq_policy_rejects_unverified_selected_model(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check(status="unreachable")]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "is not verified" in str(exc)
    else:
        raise AssertionError("expected unverified selected PPQ model to fail")


def test_generate_ppq_policy_rejects_missing_attestation_targets(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check(id="ppq-gpt-oss-120b")]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "PPQ attestation targets are required" in str(exc)
    else:
        raise AssertionError("expected missing PPQ attestation targets to fail")


def test_generate_ppq_policy_ignores_legacy_proxy_binary_digest_mismatch(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-digest",
            _digest("wrong-proxy"),
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    document = generator.generate_policy(args)
    policy = document["confidentiality"]["policy"]

    assert "proxy_binary_digest" not in policy
    assert "proxy_binary_path" not in policy


def test_generate_ppq_policy_rejects_attestation_bundle_origin_mismatch(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--attestation-bundle-url",
            "https://attest.ppq.ai/private",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "attestation_bundle_url origin must match base_url origin" in str(exc)
    else:
        raise AssertionError("expected mismatched PPQ bundle origin to fail")


def test_generate_ppq_policy_rejects_empty_verifier_command(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-command",
            "   ",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "verifier_command is empty" in str(exc)
    else:
        raise AssertionError("expected empty PPQ verifier command to fail")


def test_generate_ppq_policy_rejects_empty_router_repo(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--router-repo",
            "   ",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "router_repo must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected empty PPQ router repo to fail")


def test_generate_ppq_policy_rejects_router_repo_target_mismatch(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--router-repo",
            "attacker/conflicting-router",
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "PPQ router_repo does not match attestation target" in str(exc)
    else:
        raise AssertionError("expected conflicting PPQ router repo to fail")


def test_generate_ppq_policy_rejects_placeholder_verifier_digest(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-command-digest",
            PLACEHOLDER_DIGEST,
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "placeholder digest" in str(exc)
    else:
        raise AssertionError("expected placeholder verifier digest to fail")


def test_generate_ppq_policy_rejects_missing_backend_verification_boolean(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check(backend_sigstore_match=False)]),
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "requires backend_sigstore_match=true" in str(exc)
    else:
        raise AssertionError("expected incomplete PPQ backend evidence to fail")


def test_generate_ppq_policy_maps_selected_model_from_attestation_targets(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "ppq",
                    "checks": [
                        {
                            "id": "ppq-gpt-oss-120b",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                        }
                    ],
                }
            ]
        },
    )
    result_without_model = _verified_ppq_check(id="ppq-gpt-oss-120b")
    result_without_model.pop("model")
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([result_without_model]),
    )
    output_path = tmp_path / "ppq-policy.json"
    proxy_path = _proxy_binary(tmp_path / "ppq-proxy")

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(proxy_path),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["model_ids"] == ["private/gpt-oss-120b"]
    assert (
        document["confidentiality"]["policy"]["model_attestation_targets"][
            "private/gpt-oss-120b"
        ]["host"]
        == "gpt-oss-120b-1.inf10.tinfoil.sh"
    )


def test_generate_ppq_policy_rejects_target_backend_host_mismatch(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "ppq",
                    "checks": [
                        {
                            "id": "ppq-gpt-oss-120b",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                            "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results(
            [
                _verified_ppq_check(
                    id="ppq-gpt-oss-120b",
                    backend_host="llama3-3-70b.tinfoil.containers.tinfoil.dev",
                    backend_release_digest=BACKEND_DIGEST,
                )
            ]
        ),
    )
    proxy_path = _proxy_binary(tmp_path / "ppq-proxy")

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(proxy_path),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "backend_host does not match attestation target" in str(exc)
    else:
        raise AssertionError("expected mismatched PPQ backend host to fail")


def test_generate_ppq_policy_rejects_conflicting_backend_host_aliases(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results(
            [
                _verified_ppq_check(
                    backendHost="llama3-3-70b.tinfoil.containers.tinfoil.dev"
                )
            ]
        ),
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "backend_host aliases must match" in str(exc)
    else:
        raise AssertionError("expected conflicting PPQ backend host aliases to fail")


def test_generate_ppq_policy_accepts_camel_case_backend_host_alias(
    tmp_path: Path,
) -> None:
    result = _verified_ppq_check()
    backend_host = str(result.pop("backend_host"))
    result["backendHost"] = backend_host
    output_path = tmp_path / "ppq-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(_write_json(tmp_path / "attestation-results.json", _results([result]))),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        document["confidentiality"]["policy"]["model_attestation_targets"][
            "private/gpt-oss-120b"
        ]["host"]
        == backend_host
    )


def test_generate_ppq_policy_rejects_conflicting_backend_repo_aliases(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "ppq",
                    "checks": [
                        {
                            "id": "ppq-gpt-oss-120b",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                            "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                            "backendRepo": "tinfoilsh/confidential-other",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check(id="ppq-gpt-oss-120b")]),
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(targets_path),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "backend_repo aliases must match" in str(exc)
    else:
        raise AssertionError("expected conflicting PPQ backend repo aliases to fail")


def test_generate_ppq_policy_rejects_duplicate_selected_model_rows(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check(), _verified_ppq_check()]),
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "duplicate PPQ result rows for selected model private/gpt-oss-120b" in str(exc)
    else:
        raise AssertionError("expected duplicate PPQ selected model rows to fail")


def test_generate_ppq_policy_rejects_duplicate_requested_model_ids(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--model-id",
            "private/gpt-oss-120b",
            "--model-id",
            " PRIVATE/GPT-OSS-120B ",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "duplicate --model-id value private/gpt-oss-120b" in str(exc)
    else:
        raise AssertionError("expected duplicate PPQ model selectors to fail")


def test_generate_ppq_policy_rejects_non_private_requested_model_id(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check(model="gpt-oss-120b")]),
    )
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--model-id",
            "gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "PPQ private model_ids must start with private/" in str(exc)
    else:
        raise AssertionError("expected non-private PPQ model selector to fail")


def test_generate_ppq_policy_rejects_model_absent_from_provider_catalog(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [{"slug": "ppq", "models": [{"slug": "private/not-selected"}]}],
    )

    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--provider-catalog-json",
            str(catalog_path),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert (
            "selected PPQ model private/gpt-oss-120b is not present in "
            "provider catalog"
        ) in str(exc)
    else:
        raise AssertionError("expected missing PPQ catalog model to fail")


def test_generate_ppq_policy_rejects_non_standard_json_constants(
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
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert "must not contain NaN" in str(exc)
    else:
        raise AssertionError("expected non-standard PPQ JSON constant to fail")


def test_generate_ppq_policy_refuses_non_standard_json_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generator, "generate_policy", lambda _args: {"bad": float("nan")})

    exit_code = generator.main(
        [
            "--attestation-results-json",
            "unused-results.json",
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(tmp_path / "unused-ppq-proxy"),
        ]
    )

    assert exit_code == 1


def test_generate_ppq_policy_accepts_provider_catalog_descriptive_alias(
    tmp_path: Path,
) -> None:
    result = _verified_ppq_check(
        id="ppq-qwen3-vl",
        model="private/qwen3-vl-30b",
        backend_host="qwen3-vl-30b.inf10.tinfoil.sh",
        backend_release_digest=BACKEND_DIGEST,
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([result]),
    )
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "ppq",
                    "checks": [
                        {
                            "id": "ppq-qwen3-vl",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/qwen3-vl-30b",
                            "backend_host": "qwen3-vl-30b.inf10.tinfoil.sh",
                            "backend_repo": "tinfoilsh/confidential-qwen3-vl-30b",
                        }
                    ],
                }
            ]
        },
    )
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [{"slug": "ppq", "models": [{"slug": "qwen3-vl-30b-a3b"}]}],
    )
    output_path = tmp_path / "ppq-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(targets_path),
            "--provider-catalog-json",
            str(catalog_path),
            "--model-id",
            "private/qwen3-vl-30b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["model_ids"] == ["private/qwen3-vl-30b"]


def test_generate_ppq_policy_emits_strict_preflight_valid_policy(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )
    output_path = tmp_path / "ppq-policy.json"
    proxy_path = _proxy_binary(tmp_path / "ppq-proxy")

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(proxy_path),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["confidentiality"]["enabled"] is True
    assert document["confidentiality"]["mode"] == "ppq-private-tee"
    policy = document["confidentiality"]["policy"]
    assert policy["expected_release_digest"] == ROUTER_DIGEST
    assert "proxy_binary_digest" not in policy
    assert "proxy_binary_path" not in policy
    assert policy["verifier_command_digest"] == VERIFIER_DIGEST
    assert policy["model_attestation_targets"]["private/gpt-oss-120b"] == {
        "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
        "repo": "tinfoilsh/confidential-gpt-oss-120b",
        "expected_release_digest": BACKEND_DIGEST,
    }

    results = preflight.load_attestation_results_directory(results_path)
    validation = preflight.validate_provider_policy(
        output_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert validation["ok"] is True
    assert validation["issues"] == []


def test_generate_ppq_policy_refuses_runtime_unloadable_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_build_policy_document = generator.build_policy_document

    def build_invalid_policy_document(*args: object, **kwargs: object) -> dict[str, object]:
        document = original_build_policy_document(*args, **kwargs)
        document["confidentiality"]["mode"] = "future-ppq"
        return document

    monkeypatch.setattr(
        generator,
        "build_policy_document",
        build_invalid_policy_document,
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        _results([_verified_ppq_check()]),
    )
    output_path = tmp_path / "ppq-policy.json"

    exit_code = generator.main(
        [
            "--attestation-results-json",
            str(results_path),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert not output_path.exists()


def test_generate_ppq_policy_rejects_stale_attestation_results(
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
                    _results([_verified_ppq_check()], last_run=stale_timestamp),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
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
        raise AssertionError("expected stale PPQ attestation results to fail")


def test_generate_ppq_policy_rejects_stale_selected_model_row(
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
                    _results([_verified_ppq_check(ts=stale_timestamp)]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
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
            "selected PPQ model private/gpt-oss-120b attestation result is stale"
            in str(exc)
        )
    else:
        raise AssertionError("expected stale PPQ model row to fail")


def test_generate_ppq_policy_rejects_malformed_selected_model_row_timestamp(
    tmp_path: Path,
) -> None:
    args = generator.parse_args(
        [
            "--attestation-results-json",
            str(
                _write_json(
                    tmp_path / "attestation-results.json",
                    _results([_verified_ppq_check(ts="not-a-time")]),
                )
            ),
            "--attestation-targets-json",
            str(_targets(tmp_path / "attestation-targets.json")),
            "--model-id",
            "private/gpt-oss-120b",
            "--proxy-binary-path",
            str(_proxy_binary(tmp_path / "ppq-proxy")),
            "--verifier-manifest-json",
            str(_manifest(tmp_path / "manifest.json")),
        ]
    )

    try:
        generator.generate_policy(args)
    except generator.PolicyGenerationError as exc:
        assert (
            "selected PPQ model private/gpt-oss-120b attestation result timestamp "
            "is invalid"
        ) in str(exc)
    else:
        raise AssertionError("expected malformed PPQ model row timestamp to fail")
