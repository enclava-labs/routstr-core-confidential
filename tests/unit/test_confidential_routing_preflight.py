from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib import util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "confidential_routing_preflight.py"
)
SPEC = util.spec_from_file_location("confidential_routing_preflight", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
preflight = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)

VALID_VERIFIER_DIGEST = "sha256:" + ("b" * 64)
VALID_ATTESTATION_DIGEST = "sha256:" + ("c" * 64)
VALID_VERIFIER_COMMAND = "/opt/routstr/bin/confidential-verifier"
VALID_EHBP_CODE_MEASUREMENT = "sha256:" + ("f" * 64)
VALID_ROUTSTR_CODE_MEASUREMENT = "sha256:" + ("a" * 64)
VALID_ENV_VERIFIER_DIGEST = (
    "sha256:" + hashlib.sha256(b"routstr-tee-go-verifier").hexdigest()
)
VALID_ENV_ATTESTATION_DIGEST = (
    "sha256:" + hashlib.sha256(b"routstr-tee-attest-go").hexdigest()
)
VALID_ENV_ROUTSTR_CODE_MEASUREMENT = (
    "sha256:" + hashlib.sha256(b"routstr-code-measurement").hexdigest()
)
VALID_PRIVATEMODE_MANIFEST_DIGEST = "sha256:" + ("d" * 64)
VALID_PRIVATEMODE_PROXY_DIGEST = "sha256:" + ("e" * 64)
VALID_PRIVATEMODE_GPU_POLICY = "nvidia-ocsp-good-only"
VALID_PPQ_PROXY_PATH = "/opt/ppq/ppq-private-mode-proxy"
VALID_ROUTSTR_HPKE_KEY_CONFIG_B64 = (
    "AAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAABAAI="
)
VALID_ROUTSTR_PUBLIC_KEY = "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"
VALID_ROUTSTR_PUBLIC_KEY_DIGEST = (
    "sha256:" + hashlib.sha256(VALID_ROUTSTR_PUBLIC_KEY.encode("utf-8")).hexdigest()
)
EXAMPLE_POLICY_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "confidential-routing"
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _privatemode_full_result_details(
    model_id: str = "privatemode/model",
) -> dict[str, object]:
    return {
        "transport": "privatemode-proxy",
        "trust_tier": "app-e2ee",
        "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
        "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
        "coordinator_measurement": "sha256:" + ("1" * 64),
        "secret_service_measurement": "sha256:" + ("2" * 64),
        "ai_worker_measurement": "sha256:" + ("3" * 64),
        "key_release_binding": "sha256:" + ("4" * 64),
        "gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
        "selected_model_ids": [model_id],
        "coordinator_attestation_doc_digest": "sha256:" + ("5" * 64),
        "mesh_ca_digest": "sha256:" + ("6" * 64),
        "secret_service_certificate_digest": "sha256:" + ("7" * 64),
        "ai_worker_manifest_digest": "sha256:" + ("8" * 64),
        "attested_workload_identity_digest": "sha256:" + ("9" * 64),
        "attested_workload_policy_digest": "sha256:" + ("a" * 64),
        "expected_workload_identity_digest": preflight.sha256_json_digest(
            {"ids": [], "sans": ["workload-model"]}
        ),
        "model_workload_binding_digest": preflight.sha256_json_digest(
            {
                model_id: {
                    "workload_ids": [],
                    "workload_sans": ["workload-model"],
                }
            }
        ),
        "nvidia_ocsp_policy_header_digest": "sha256:" + ("b" * 64),
        "nvidia_ocsp_policy_mac_digest": "sha256:" + ("c" * 64),
        "prompt_encryption_ciphertext_digest": "sha256:" + ("d" * 64),
        "inference_secret_id_digest": "sha256:" + ("e" * 64),
        "verification_steps": {
            "contrast_manifest": True,
            "coordinator_attestation": True,
            "mesh_ca_binding": True,
            "secret_service_tls": True,
            "ai_worker_attestation": True,
            "gpu_attestation": True,
            "key_release_binding": True,
            "prompt_encryption": True,
            "nvidia_ocsp_revocation": True,
        },
    }


def test_preflight_cli_defaults_to_daily_attestation_result_freshness() -> None:
    args = preflight.parse_args(["--attestation-results-json", "attestation.json"])

    assert args.max_attestation_result_age_seconds == 86_400


def _ppq_private_backend_attestation_policy(model_id: str) -> dict[str, object]:
    return {
        "require_model_attestations": True,
        "model_attestation_targets": {
            model_id: {
                "host": f"{model_id.removeprefix('private/')}.tinfoil.example",
                "repo": f"tinfoilsh/confidential-{model_id.removeprefix('private/')}",
                "expected_release_digest": VALID_ATTESTATION_DIGEST,
            }
        },
    }


@pytest.mark.parametrize(
    "filename",
    [
        "tinfoil-provider-settings.example.json",
        "ppq-private-provider-settings.example.json",
        "privatemode-provider-settings.example.json",
    ],
)
def test_checked_in_confidential_provider_policy_examples_validate(
    filename: str,
) -> None:
    path = EXAMPLE_POLICY_DIR / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    result = preflight.validate_provider_policy(path)

    assert result["ok"] is True
    assert result["issues"] == []
    assert document["confidentiality"]["enabled"] is True


def test_privatemode_example_pins_verifier_emitted_gpu_policy() -> None:
    path = EXAMPLE_POLICY_DIR / "privatemode-provider-settings.example.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    policy = document["confidentiality"]["policy"]

    assert policy["expected_gpu_attestation_policy"] == "nvidia-ocsp-good-only"


def test_checked_in_routstr_tee_env_example_surfaces_strict_env_requirements() -> None:
    path = EXAMPLE_POLICY_DIR / "routstr-tee-environment.example"
    contents = path.read_text(encoding="utf-8")

    for env_var in (
        "CONFIDENTIAL_ROUTING_MODE",
        "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
        "ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT",
        "ROUTSTR_TEE_VERIFIER_COMMAND",
        "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
        "ROUTSTR_TEE_ATTESTATION_COMMAND",
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        "ROUTSTR_TEE_PUBLIC_KEY_PATH",
        "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_PATH",
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
    ):
        assert env_var in contents
    assert "attested-tls-termination" in contents


def test_preflight_strict_deployment_help_describes_full_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        preflight.parse_args(["--help"])
    assert exc_info.value.code == 0

    help_text = capsys.readouterr().out
    assert "local TEE readiness" in help_text
    assert "provider policy" in help_text
    assert "guardrail files" in help_text
    assert "routable selected model" in help_text


def test_preflight_json_output_is_machine_parseable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_dir = tmp_path / "verifier"
    module_dir.mkdir()
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/go")
    monkeypatch.setattr(
        preflight,
        "TARGETS",
        (
            preflight.VerifierTarget(
                name="dummy",
                module_dir=module_dir,
                output_name="dummy-verifier",
                test_args=("go", "test", "./..."),
                build_args=("go", "build"),
                purpose="dummy verifier",
                modes=("tinfoil",),
                verifier_names=(("tinfoil", "dummy-verifier"),),
            ),
        ),
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    exit_code = preflight.main(
        ["--json", "--skip-build", "--output-dir", str(tmp_path / "out")]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["deployment_ready"] is False
    assert report["local_tee_ready"] is False
    assert report["provider_policies_ready"] is False
    assert report["provider_policy_count"] == 0
    assert report["verifiers"][0]["name"] == "dummy"
    assert report["verifiers"][0]["modes"] == ["tinfoil"]
    assert report["verifiers"][0]["verifier_names"] == {
        "tinfoil": "dummy-verifier"
    }
    assert "$ go test ./..." in captured.err


def test_preflight_json_env_output_is_machine_parseable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PREFLIGHT_JSON", "1")
    monkeypatch.setattr(preflight, "build_verifiers", lambda **_kwargs: [])
    monkeypatch.setattr(preflight, "env_checks", lambda: [])

    exit_code = preflight.main(
        ["--skip-build", "--skip-tests", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["deployment_ready"] is False
    assert report["provider_policy_count"] == 0


def test_preflight_strict_deployment_ready_env_rejects_missing_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STRICT_DEPLOYMENT_READY", "1")
    monkeypatch.setattr(preflight, "build_verifiers", lambda **_kwargs: [])
    monkeypatch.setattr(
        preflight,
        "env_checks",
        lambda: [
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
    )

    exit_code = preflight.main(
        ["--skip-build", "--skip-tests", "--output-dir", str(tmp_path)]
    )

    assert exit_code == 1


def test_preflight_json_output_refuses_non_standard_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(preflight, "build_verifiers", lambda **_kwargs: [])
    monkeypatch.setattr(preflight, "env_checks", lambda: [])
    monkeypatch.setattr(
        preflight,
        "preflight_json_report",
        lambda **_kwargs: {"bad": float("nan")},
    )

    exit_code = preflight.main(
        ["--json", "--skip-build", "--skip-tests", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "failed to serialize preflight JSON report" in captured.err


def test_preflight_strict_deployment_ready_rejects_missing_provider_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "build_verifiers",
        lambda **kwargs: [
            {
                "name": "dummy",
                "sha256": VALID_VERIFIER_DIGEST,
                "artifact_path": str(tmp_path / "dummy"),
                "purpose": "dummy verifier",
            }
        ],
    )
    monkeypatch.setattr(
        preflight,
        "env_checks",
        lambda: [
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
    )

    exit_code = preflight.main(
        [
            "--json",
            "--skip-build",
            "--strict-deployment-ready",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


def test_preflight_strict_deployment_ready_rejects_missing_guardrail_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "build_verifiers", lambda **kwargs: [])
    monkeypatch.setattr(
        preflight,
        "env_checks",
        lambda: [
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
    )
    monkeypatch.setattr(
        preflight,
        "validate_provider_policy",
        lambda *args, **kwargs: {
            "path": "provider.json",
            "mode": "tinfoil",
            "model_ids": ["tinfoil/secure-model"],
            "ok": True,
            "issues": [],
            "warnings": [],
        },
    )

    exit_code = preflight.main(
        [
            "--json",
            "--skip-build",
            "--strict-deployment-ready",
            "--policy-json",
            str(tmp_path / "provider.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


def test_preflight_strict_deployment_json_marks_missing_guardrails_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy_path = tmp_path / "provider.json"
    policy_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(preflight, "build_verifiers", lambda **kwargs: [])
    monkeypatch.setattr(
        preflight,
        "env_checks",
        lambda: [
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
    )
    monkeypatch.setattr(
        preflight,
        "validate_provider_policy",
        lambda *args, **kwargs: {
            "path": "provider.json",
            "mode": "tinfoil",
            "model_ids": ["tinfoil/secure-model"],
            "ok": True,
            "issues": [],
            "warnings": [],
        },
    )

    exit_code = preflight.main(
        [
            "--json",
            "--skip-build",
            "--strict-deployment-ready",
            "--policy-json",
            str(policy_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["local_tee_ready"] is True
    assert report["provider_policies_ready"] is True
    assert report["guardrail_inputs_ready"] is False
    assert report["deployment_ready"] is False


def test_preflight_strict_deployment_ready_accepts_complete_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "provider.json"
    targets_path = tmp_path / "attestation-targets.json"
    results_path = tmp_path / "attestation-results.json"
    catalog_path = tmp_path / "providers.json"
    policy_path.write_text("{}", encoding="utf-8")
    targets_path.write_text("{}", encoding="utf-8")
    results_path.write_text("{}", encoding="utf-8")
    catalog_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(preflight, "build_verifiers", lambda **kwargs: [])
    monkeypatch.setattr(
        preflight,
        "env_checks",
        lambda: [
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
    )
    monkeypatch.setattr(
        preflight,
        "load_attestation_targets_directory",
        lambda path: object(),
    )
    monkeypatch.setattr(
        preflight,
        "load_attestation_results_directory",
        lambda path, **kwargs: object(),
    )
    monkeypatch.setattr(
        preflight,
        "load_provider_model_catalog",
        lambda path: object(),
    )
    monkeypatch.setattr(
        preflight,
        "validate_provider_policy",
        lambda *args, **kwargs: {
            "path": "provider.json",
            "mode": "tinfoil",
            "model_ids": ["tinfoil/secure-model"],
            "ok": True,
            "issues": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        preflight,
        "provider_model_support_summary",
        lambda **kwargs: {
            "tinfoil": {
                "routable_with_full_attestation": ["tinfoil/secure-model"]
            }
        },
    )

    exit_code = preflight.main(
        [
            "--json",
            "--skip-build",
            "--strict-deployment-ready",
            "--policy-json",
            str(policy_path),
            "--provider-catalog-json",
            str(catalog_path),
            "--attestation-targets-json",
            str(targets_path),
            "--attestation-results-json",
            str(results_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0


def test_preflight_json_report_exposes_deployment_readiness(
    tmp_path: Path,
) -> None:
    report = preflight.preflight_json_report(
        verifier_manifest=[
            {
                "name": "dummy",
                "sha256": VALID_VERIFIER_DIGEST,
                "artifact_path": str(tmp_path / "dummy"),
                "purpose": "dummy verifier",
            }
        ],
        checks=[
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": True,
                "issues": [],
                "warnings": [],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["deployment_ready"] is True
    assert report["local_tee_ready"] is True
    assert report["provider_policies_ready"] is True
    assert report["provider_policy_count"] == 1


def test_preflight_json_report_requires_provider_model_support_when_supplied(
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
                        },
                        {
                            "id": "tinfoil-secure-model",
                            "type": "tinfoil",
                            "host": "secure-model.tinfoil.example",
                            "model": "tinfoil/secure-model",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "host": "inference.tinfoil.sh",
                    "status": "failed",
                    "error": preflight.TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-secure-model",
                    "host": "secure-model.tinfoil.example",
                    "status": "verified",
                },
            ]
        },
    )
    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": True,
                "issues": [],
                "warnings": [],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    assert report["local_tee_ready"] is True
    assert report["provider_policies_ready"] is True
    assert report["provider_model_support_ready"] is False
    assert report["deployment_ready"] is False
    assert report["provider_model_support"]["tinfoil"][
        "routable_with_full_attestation"
    ] == []


def test_preflight_json_report_fails_closed_without_provider_policies(
    tmp_path: Path,
) -> None:
    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["deployment_ready"] is False
    assert report["local_tee_ready"] is True
    assert report["provider_policies_ready"] is False
    assert report["provider_policy_count"] == 0


def test_preflight_json_report_fails_closed_for_missing_local_tee(
    tmp_path: Path,
) -> None:
    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[
            {
                "status": "missing",
                "key": "ROUTSTR_TEE_ATTESTATION_COMMAND",
                "detail": "missing command",
            }
        ],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": True,
                "issues": [],
                "warnings": [],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["deployment_ready"] is False
    assert report["local_tee_ready"] is False
    assert report["provider_policies_ready"] is True


def test_preflight_json_report_fails_closed_without_local_tee_checks(
    tmp_path: Path,
) -> None:
    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": True,
                "issues": [],
                "warnings": [],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["deployment_ready"] is False
    assert report["local_tee_ready"] is False
    assert report["provider_policies_ready"] is True


def test_preflight_json_report_fails_closed_for_provider_policy_issue(
    tmp_path: Path,
) -> None:
    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": False,
                "issues": ["missing model attestation target"],
                "warnings": [],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["deployment_ready"] is False
    assert report["local_tee_ready"] is True
    assert report["provider_policies_ready"] is False


def test_preflight_json_report_fails_closed_for_empty_provider_guardrails(
    tmp_path: Path,
) -> None:
    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": True,
                "model_ids": ["tinfoil/secure-model"],
                "issues": [],
                "warnings": [],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
        provider_model_catalog=preflight.ProviderModelCatalog(provider_models={}),
        attestation_targets_directory=preflight.AttestationTargetsDirectory(
            provider_hosts={},
            provider_host_model_ids={},
            provider_host_repos={},
            provider_check_hosts={},
        ),
        attestation_results_directory=preflight.AttestationResultsDirectory(
            provider_host_statuses={},
            provider_host_errors={},
        ),
    )

    assert report["deployment_ready"] is False
    assert report["local_tee_ready"] is True
    assert report["provider_policies_ready"] is True
    assert report["provider_model_support_ready"] is False
    assert report["provider_model_support"] == {}


def test_preflight_provider_support_requires_selected_policy_model() -> None:
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "provider.json",
                "mode": "ppq-private-tee",
                "ok": True,
                "model_ids": ["private/kimi-k2-6"],
                "issues": [],
                "warnings": [],
            }
        ],
        {
            "ppq": {
                "routable_with_full_attestation": ["private/gpt-oss-120b"],
            }
        },
    )

    assert ready is False


def test_preflight_provider_support_requires_every_selected_policy_model() -> None:
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "ready-ppq.json",
                "mode": "ppq-private-tee",
                "ok": True,
                "model_ids": ["private/gpt-oss-120b"],
                "issues": [],
                "warnings": [],
            },
            {
                "path": "stale-tinfoil.json",
                "mode": "tinfoil",
                "ok": True,
                "model_ids": ["gpt-oss-120b"],
                "issues": [],
                "warnings": [],
            },
        ],
        {
            "ppq": {
                "routable_with_full_attestation": ["private/gpt-oss-120b"],
            },
            "tinfoil": {
                "routable_with_full_attestation": [],
            },
        },
    )

    assert ready is False


def test_preflight_provider_support_rejects_prefix_selectors_even_with_routable_exact_model() -> (
    None
):
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": True,
                "model_ids": ["tinfoil/gpt-secure"],
                "model_id_prefixes": ["tinfoil/"],
                "issues": [],
                "warnings": [],
            }
        ],
        {
            "tinfoil": {
                "routable_with_full_attestation": ["tinfoil/gpt-secure"],
            }
        },
    )

    assert ready is False


def test_preflight_provider_support_accepts_tinfoil_ehbp_policy_with_verified_models_when_router_tls_binding_failed() -> (
    None
):
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "tinfoil-ehbp.json",
                "mode": "tinfoil",
                "transport_security": "ehbp",
                "ok": True,
                "model_ids": ["tinfoil/llama-3.3-70b"],
                "model_id_prefixes": [],
                "issues": [],
                "warnings": [],
            }
        ],
        {
            "tinfoil": {
                "provider_level_ready": False,
                "provider_level_results": [
                    {
                        "host": "inference.tinfoil.sh",
                        "status": "failed",
                        "error": preflight.TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR,
                    }
                ],
                "routable_with_full_attestation": [],
                "attestation_target_models": ["tinfoil/llama3-3-70b"],
                "verified_attestation_models": ["tinfoil/llama-3.3-70b"],
            }
        },
    )

    assert ready is True


def test_preflight_provider_support_rejects_tinfoil_ehbp_policy_without_selected_target() -> (
    None
):
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "tinfoil-ehbp.json",
                "mode": "tinfoil",
                "transport_security": "ehbp",
                "ok": True,
                "model_ids": ["tinfoil/llama-3.3-70b"],
                "model_id_prefixes": [],
                "issues": [],
                "warnings": [],
            }
        ],
        {
            "tinfoil": {
                "provider_level_ready": False,
                "provider_level_results": [
                    {
                        "host": "inference.tinfoil.sh",
                        "status": "failed",
                        "error": preflight.TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR,
                    }
                ],
                "routable_with_full_attestation": [],
                "attestation_target_models": ["tinfoil/other-model"],
                "verified_attestation_models": ["tinfoil/llama-3.3-70b"],
            }
        },
    )

    assert ready is False


def test_preflight_provider_support_rejects_tinfoil_ehbp_policy_with_non_tls_router_failure() -> (
    None
):
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "tinfoil-ehbp.json",
                "mode": "tinfoil",
                "transport_security": "ehbp",
                "ok": True,
                "model_ids": ["tinfoil/llama-3.3-70b"],
                "model_id_prefixes": [],
                "issues": [],
                "warnings": [],
            }
        ],
        {
            "tinfoil": {
                "provider_level_ready": False,
                "provider_level_results": [
                    {
                        "host": "inference.tinfoil.sh",
                        "status": "failed",
                        "error": "hardware attestation report was not verified",
                    }
                ],
                "routable_with_full_attestation": [],
                "verified_attestation_models": ["tinfoil/llama-3.3-70b"],
            }
        },
    )

    assert ready is False


def test_preflight_provider_support_rejects_malformed_selectors_even_with_routable_model() -> (
    None
):
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "provider.json",
                "mode": "ppq-private-tee",
                "ok": True,
                "model_ids": ["private/gpt-oss-120b", {"unexpected": "object"}],
                "model_id_prefixes": [{"prefix": "private/"}],
                "issues": [],
                "warnings": [],
            }
        ],
        {
            "ppq": {
                "routable_with_full_attestation": ["private/gpt-oss-120b"],
            }
        },
    )

    assert ready is False


def test_preflight_provider_support_rejects_case_variant_selectors_even_with_routable_model() -> (
    None
):
    ready = preflight.provider_model_support_ready_for_policies(
        [
            {
                "path": "provider.json",
                "mode": "ppq-private-tee",
                "ok": True,
                "model_ids": ["private/gpt-oss-120b", "PRIVATE/gpt-oss-120b"],
                "model_id_prefixes": [],
                "issues": [],
                "warnings": [],
            }
        ],
        {
            "ppq": {
                "routable_with_full_attestation": ["private/gpt-oss-120b"],
            }
        },
    )

    assert ready is False


def test_preflight_json_report_fails_closed_for_provider_guardrail_warnings(
    tmp_path: Path,
) -> None:
    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[
            {
                "status": "ok",
                "key": "CONFIDENTIAL_ROUTING_MODE",
                "detail": "required",
            }
        ],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": True,
                "issues": [],
                "warnings": [
                    "Tinfoil provider host inference.tinfoil.sh is not verified "
                    "in confidential-inference attestation results"
                ],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["deployment_ready"] is False
    assert report["local_tee_ready"] is True
    assert report["provider_policies_ready"] is False


def test_preflight_json_report_exposes_guardrail_inputs(
    tmp_path: Path,
) -> None:
    providers_path = tmp_path / "providers.json"
    targets_path = tmp_path / "attestation-targets.json"
    results_path = tmp_path / "attestation-results.json"
    providers_path.write_text('{"providers":[]}', encoding="utf-8")
    targets_path.write_text('{"providers":[]}', encoding="utf-8")
    results_path.write_text('{"checks":[]}', encoding="utf-8")

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        provider_catalog_json=providers_path,
        attestation_targets_json=targets_path,
        attestation_results_json=results_path,
        max_attestation_result_age_seconds=86_400,
    )

    assert report["guardrail_inputs"] == {
        "max_attestation_result_age_seconds": 86_400,
        "provider_catalog_path": str(providers_path),
        "provider_catalog_sha256": (
            "sha256:" + hashlib.sha256(b'{"providers":[]}').hexdigest()
        ),
        "attestation_targets_path": str(targets_path),
        "attestation_targets_sha256": (
            "sha256:" + hashlib.sha256(b'{"providers":[]}').hexdigest()
        ),
        "attestation_results_path": str(results_path),
        "attestation_results_sha256": (
            "sha256:" + hashlib.sha256(b'{"checks":[]}').hexdigest()
        ),
    }


def test_preflight_json_report_exposes_provider_model_support(
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
                        },
                        {
                            "id": "ppq-kimi-k2-6",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/kimi-k2-6",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-gpt-oss-120b",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": "sha256:" + ("1" * 64),
                    "backend_release_digest": "sha256:" + ("2" * 64),
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [
            {
                "slug": "ppq",
                "models": [
                    {"slug": "gpt-oss-120b"},
                    {"slug": "kimi-k2.6"},
                    {"slug": "non-confidential-catalog-entry"},
                ],
            }
        ],
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )
    catalog = preflight.load_provider_model_catalog(catalog_path)

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        provider_model_catalog=catalog,
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    assert report["provider_model_support"]["ppq"] == {
        "catalog_models": [
            "gpt-oss-120b",
            "kimi-k2-6",
            "non-confidential-catalog-entry",
        ],
        "attestation_target_models": [
            "private/gpt-oss-120b",
            "private/kimi-k2-6",
        ],
        "verified_attestation_models": ["private/gpt-oss-120b"],
        "provider_level_required": True,
        "provider_level_ready": False,
        "provider_level_results": [
            {
                "host": "api.ppq.ai",
                "status": "missing",
                "error": (
                    "PPQ private provider host api.ppq.ai is missing verified "
                    "results for selected models: private/kimi-k2-6"
                ),
            }
        ],
        "routable_with_full_attestation": [],
        "attestation_targets_without_verified_results": [
            {
                "model": "private/kimi-k2-6",
                "status": "missing",
                "error": "",
            }
        ],
        "catalog_models_without_attestation_targets": [
            "non-confidential-catalog-entry"
        ],
    }


def test_preflight_model_support_does_not_treat_ppq_host_only_result_as_routable(
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
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "status": "verified",
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    ppq_support = report["provider_model_support"]["ppq"]
    assert ppq_support["attestation_target_models"] == ["private/gpt-oss-120b"]
    assert ppq_support["verified_attestation_models"] == []
    assert ppq_support["routable_with_full_attestation"] == []
    assert ppq_support["attestation_targets_without_verified_results"] == [
        {
            "model": "private/gpt-oss-120b",
            "status": "failed",
            "error": (
                "release_digest is missing or invalid in "
                "confidential-inference attestation results"
            ),
        }
    ]


def test_preflight_model_support_requires_ppq_backend_evidence(
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
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-gpt-oss-120b",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": "sha256:" + ("1" * 64),
                    "backend_release_digest": "sha256:" + ("2" * 64),
                    "backend_tls_matches": True,
                    "backend_sigstore_match": False,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    ppq_support = report["provider_model_support"]["ppq"]
    assert ppq_support["verified_attestation_models"] == []
    assert ppq_support["routable_with_full_attestation"] == []
    assert ppq_support["attestation_targets_without_verified_results"] == [
        {
            "model": "private/gpt-oss-120b",
            "status": "failed",
            "error": (
                "backend_sigstore_match was not verified in "
                "confidential-inference attestation results"
            ),
        }
    ]


def test_preflight_provider_model_support_rejects_ppq_wrong_attestation_target_model(
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
                            "id": "ppq-llama",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/llama3-3-70b",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-gpt-oss-120b",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": "sha256:" + ("1" * 64),
                    "backend_release_digest": "sha256:" + ("2" * 64),
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    support = preflight.provider_model_support_summary(
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )["ppq"]

    assert support["verified_attestation_models"] == []
    assert support["routable_with_full_attestation"] == []
    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "selected model is missing from confidential-inference attestation targets"
    )


def test_preflight_model_support_requires_provider_level_attestation(
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
                        },
                        {
                            "id": "tinfoil-gpt-oss-120b",
                            "type": "tinfoil",
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "host": "inference.tinfoil.sh",
                    "status": "failed",
                    "error": "TLS certificate binding was not verified",
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-gpt-oss-120b",
                    "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "model": "gpt-oss-120b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    tinfoil_support = report["provider_model_support"]["tinfoil"]
    assert tinfoil_support["provider_level_required"] is True
    assert tinfoil_support["provider_level_ready"] is False
    assert tinfoil_support["provider_level_results"] == [
        {
            "host": "inference.tinfoil.sh",
            "status": "failed",
            "error": "TLS certificate binding was not verified",
        }
    ]
    assert tinfoil_support["verified_attestation_models"] == [
        "gpt-oss-120b",
        "gpt-oss-120b-1",
    ]
    assert tinfoil_support["routable_with_full_attestation"] == []


def test_preflight_ignores_tinfoil_router_model_in_target_directory(
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
                            "label": "Router (inference.tinfoil.sh)",
                            "type": "tinfoil",
                            "host": "inference.tinfoil.sh",
                            "model": "tinfoil/gpt-secure",
                        }
                    ],
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)

    assert targets.provider_check_hosts["tinfoil"]["tinfoil-router"] == (
        "inference.tinfoil.sh"
    )
    assert targets.provider_check_models.get("tinfoil", {}) == {}
    assert targets.provider_target_models.get("tinfoil", set()) == set()


def test_preflight_ignores_tinfoil_router_model_in_result_directory(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "label": "Router (inference.tinfoil.sh)",
                    "host": "inference.tinfoil.sh",
                    "model": "tinfoil/gpt-secure",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                }
            ]
        },
    )

    results = preflight.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["tinfoil"]["inference.tinfoil.sh"] == (
        "verified"
    )
    assert results.provider_model_statuses.get("tinfoil", {}) == {}


def test_preflight_ignores_duplicate_attestation_target_check_ids(
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
                            "id": "ppq-private-model",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                        },
                        {
                            "id": "ppq-private-model",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/llama3-3-70b",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-private-model",
                    "status": "verified",
                }
            ]
        },
    )

    targets_directory = preflight.load_attestation_targets_directory(targets_path)
    results_directory = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert targets_directory.provider_check_hosts.get("ppq", {}) == {}
    assert targets_directory.provider_check_models.get("ppq", {}) == {}
    assert targets_directory.provider_target_models.get("ppq", set()) == set()
    assert results_directory.provider_host_statuses == {}
    assert results_directory.provider_model_statuses == {}


def test_preflight_model_support_requires_tinfoil_router_target_for_routable_models(
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
                            "id": "tinfoil-gpt-oss-120b",
                            "type": "tinfoil",
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-gpt-oss-120b",
                    "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "model": "gpt-oss-120b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    tinfoil_support = report["provider_model_support"]["tinfoil"]
    assert tinfoil_support["provider_level_required"] is True
    assert tinfoil_support["provider_level_ready"] is False
    assert tinfoil_support["provider_level_results"] == [
        {
            "host": "inference.tinfoil.sh",
            "status": "missing",
            "error": "Tinfoil provider-level router attestation target is missing",
        }
    ]
    assert tinfoil_support["verified_attestation_models"] == [
        "gpt-oss-120b",
        "gpt-oss-120b-1",
    ]
    assert tinfoil_support["routable_with_full_attestation"] == []


def test_preflight_model_support_rejects_tinfoil_router_without_release_digest(
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
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "host": "inference.tinfoil.sh",
                    "status": "verified",
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    assert report["provider_model_support"]["tinfoil"]["provider_level_results"] == [
        {
            "host": "inference.tinfoil.sh",
            "status": "failed",
            "error": (
                "release_digest is missing or invalid in "
                "confidential-inference attestation results"
            ),
        }
    ]


def test_preflight_provider_level_results_use_check_id_when_source_host_differs(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "privatemode",
                    "checks": [
                        {
                            "id": "privatemode-api",
                            "type": "privatemode",
                            "url": "https://docs.privatemode.ai",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "id": "privatemode-api",
                    "source_url": (
                        "https://cdn.confidential.cloud/privatemode/v2/manifest.json"
                    ),
                    "status": "failed",
                    "error": "reference values only",
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    report = preflight.preflight_json_report(
        verifier_manifest=[],
        checks=[],
        policies=[],
        manifest_path=tmp_path / "manifest.json",
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )

    privatemode_support = report["provider_model_support"]["privatemode"]
    assert privatemode_support["provider_level_results"] == [
        {
            "host": "docs.privatemode.ai",
            "status": "failed",
            "error": "reference values only",
        }
    ]


def test_preflight_provider_model_support_rejects_bare_privatemode_verified_model(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "privatemode",
                    "models": ["privatemode/model"],
                    "checks": [
                        {
                            "id": "privatemode-model",
                            "type": "privatemode",
                            "host": "cdn.confidential.cloud",
                            "model": "privatemode/model",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "id": "privatemode-model",
                    "host": "cdn.confidential.cloud",
                    "model": "privatemode/model",
                    "status": "verified",
                    "trust_tier": "app-e2ee",
                },
            ]
        },
    )
    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    support = preflight.provider_model_support_summary(
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )["privatemode"]

    assert support["verified_attestation_models"] == []
    assert support["routable_with_full_attestation"] == []
    assert support["attestation_targets_without_verified_results"] == [
        {
            "model": "privatemode/model",
            "status": "failed",
            "error": (
                "Privatemode selected model evidence is missing policy-bound "
                "proof details"
            ),
        }
    ]
    assert results.provider_model_statuses["privatemode"]["privatemode/model"] == (
        "failed"
    )
    assert (
        results.provider_model_errors["privatemode"]["privatemode/model"]
        == "Privatemode selected model evidence is missing policy-bound "
        "proof details"
    )


def test_preflight_provider_model_support_rejects_privatemode_duplicate_selected_model_ids(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "privatemode",
                    "models": ["privatemode/model"],
                    "checks": [
                        {
                            "id": "privatemode-model",
                            "type": "privatemode",
                            "host": "cdn.confidential.cloud",
                            "model": "privatemode/model",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "id": "privatemode-model",
                    "host": "cdn.confidential.cloud",
                    "model": "privatemode/model",
                    "status": "verified",
                    **{
                        **_privatemode_full_result_details(),
                        "selected_model_ids": [
                            "privatemode/model",
                            "PRIVATEMODE/model",
                        ],
                    },
                },
            ]
        },
    )
    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    support = preflight.provider_model_support_summary(
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )["privatemode"]

    assert support["verified_attestation_models"] == []
    assert support["routable_with_full_attestation"] == []
    assert support["attestation_targets_without_verified_results"] == [
        {
            "model": "privatemode/model",
            "status": "failed",
            "error": (
                "Privatemode selected model evidence selected_model_ids must "
                "exactly match the selected model"
            ),
        }
    ]


def test_preflight_provider_model_support_rejects_privatemode_wrong_attestation_target_model(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "privatemode",
                    "models": ["privatemode/other-model"],
                    "checks": [
                        {
                            "id": "privatemode-other",
                            "type": "privatemode",
                            "host": "cdn.confidential.cloud",
                            "model": "privatemode/other-model",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "id": "privatemode-model",
                    "host": "cdn.confidential.cloud",
                    "model": "privatemode/model",
                    "status": "verified",
                    **_privatemode_full_result_details(),
                },
            ]
        },
    )
    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    support = preflight.provider_model_support_summary(
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )["privatemode"]

    assert support["verified_attestation_models"] == []
    assert support["routable_with_full_attestation"] == []
    assert results.provider_model_statuses["privatemode"]["privatemode/model"] == (
        "failed"
    )
    assert (
        results.provider_model_errors["privatemode"]["privatemode/model"]
        == "selected model is missing from confidential-inference attestation targets"
    )


def test_preflight_provider_model_support_accepts_full_privatemode_model_proof(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "privatemode",
                    "models": ["privatemode/model"],
                    "checks": [
                        {
                            "id": "privatemode-model",
                            "type": "privatemode",
                            "host": "cdn.confidential.cloud",
                            "model": "privatemode/model",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "id": "privatemode-model",
                    "host": "cdn.confidential.cloud",
                    "model": "privatemode/model",
                    "status": "verified",
                    **_privatemode_full_result_details(),
                },
            ]
        },
    )
    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    support = preflight.provider_model_support_summary(
        attestation_targets_directory=targets,
        attestation_results_directory=results,
    )["privatemode"]

    assert support["provider_level_required"] is True
    assert support["provider_level_ready"] is True
    assert support["provider_level_results"] == [
        {
            "host": "cdn.confidential.cloud",
            "status": "verified",
            "error": "",
        }
    ]
    assert support["verified_attestation_models"] == ["privatemode/model"]
    assert support["routable_with_full_attestation"] == ["privatemode/model"]
    assert support["attestation_targets_without_verified_results"] == []


def test_preflight_text_report_labels_failed_provider_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preflight.print_report(
        verifier_manifest=[
            {
                "name": "dummy",
                "sha256": VALID_VERIFIER_DIGEST,
                "artifact_path": str(tmp_path / "dummy"),
                "purpose": "dummy verifier",
            }
        ],
        checks=[],
        policies=[
            {
                "path": "provider.json",
                "mode": "tinfoil",
                "ok": False,
                "issues": ["Tinfoil provider host inference.tinfoil.sh failed"],
                "warnings": [],
            }
        ],
        manifest_path=tmp_path / "manifest.json",
    )

    captured = capsys.readouterr()
    assert "- [failed] provider.json (tinfoil)" in captured.out
    assert "- [missing] provider.json" not in captured.out


def test_preflight_verifier_targets_cover_all_confidential_modes() -> None:
    covered_modes = {
        mode
        for target in preflight.TARGETS
        for mode in target.modes
    }

    assert covered_modes == preflight.SUPPORTED_CONFIDENTIALITY_MODES

    ppq_targets = [
        target
        for target in preflight.TARGETS
        if "ppq-private-tee" in target.modes
    ]
    assert len(ppq_targets) == 1
    assert ppq_targets[0].output_name == "routstr-tinfoil-go-verifier"
    assert dict(ppq_targets[0].verifier_names) == {
        "tinfoil": "routstr-tinfoil-go-verifier",
        "ppq-private-tee": "routstr-ppq-private-go-verifier",
    }


def test_preflight_rejects_tinfoil_policy_without_model_attestations(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "Tinfoil require_model_attestations must be true for selected model routing"
        in result["issues"]
    )


def test_preflight_rejects_tinfoil_required_model_attestation_without_targets(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "model_attestation_targets is required" in result["issues"]


def test_preflight_rejects_tinfoil_model_target_without_release_digest(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_code_measurement_fingerprint": (
                                "sha256:" + ("b" * 64)
                            ),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6: "
        "expected_release_digest or allowed_release_digests is required"
    ) in result["issues"]


def test_preflight_accepts_tinfoil_required_model_attestation_targets(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_rejects_tinfoil_model_target_conflicting_identity_aliases(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "expected_enclave_host": "other.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_repo": "attacker/conflicting-kimi",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6: "
        "enclave host aliases must match"
    ) in result["issues"]
    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6: repo aliases must match"
    ) in result["issues"]


def test_preflight_rejects_tinfoil_model_attestation_target_on_router_origin(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil-router-target.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": (
                                "https://inference.tinfoil.sh/.well-known/"
                                "tinfoil-attestation"
                            ),
                            "repo": "tinfoilsh/confidential-model-router",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6 must use a distinct "
        "model enclave origin, not the Tinfoil router origin"
    ) in result["issues"]


@pytest.mark.parametrize(
    ("target", "expected_issue"),
    [
        (
            {"host": "localhost"},
            "model_attestation_targets for tinfoil/kimi-k2-6: "
            "host must not use localhost or a non-global IP address",
        ),
        (
            {"attestation_url": "https://192.168.1.10/.well-known/tinfoil-attestation"},
            "model_attestation_targets for tinfoil/kimi-k2-6: "
            "attestation_url must not use localhost or a non-global IP address",
        ),
    ],
)
def test_preflight_rejects_tinfoil_private_model_target_hosts(
    tmp_path: Path,
    target: dict[str, object],
    expected_issue: str,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil-private-target.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            **target,
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert expected_issue in result["issues"]


def test_preflight_rejects_tinfoil_model_target_missing_from_attestation_directory(
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
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                        },
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-unknown-model-target.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/nomic-embed-text"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/nomic-embed-text": {
                            "host": "nomic-embed-text.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-nomic-embed-text",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/nomic-embed-text host "
        "nomic-embed-text.inf10.tinfoil.sh is not present in attestation "
        "targets for provider tinfoil"
    ) in result["issues"]


def test_preflight_accepts_tinfoil_model_target_from_attestation_directory(
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
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                        },
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_rejects_tinfoil_model_target_for_different_directory_model(
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
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6 enclave",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                        },
                        {
                            "id": "tinfoil-llama",
                            "label": "Llama 3.3 70B enclave",
                            "type": "tinfoil",
                            "host": "llama3-3-70b.tinfoil.containers.tinfoil.dev",
                        },
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-llama-wrong-target.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/llama-3.3-70b"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/llama-3.3-70b": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/llama-3.3-70b host "
        "kimi-k2-6.inf13.tinfoil.sh is listed for a different model in "
        "attestation targets for provider tinfoil"
    ) in result["issues"]


def test_preflight_rejects_tinfoil_selected_model_that_broadens_directory_identity(
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
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6 enclave",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi-preview-wrong-target.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6-preview"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6-preview": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6-preview host "
        "kimi-k2-6.inf13.tinfoil.sh is listed for a different model in "
        "attestation targets for provider tinfoil"
    ) in result["issues"]


def test_preflight_rejects_tinfoil_selected_model_prefix_of_directory_identity(
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
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6 enclave",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi-prefix-wrong-target.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/kimi-k2 host "
        "kimi-k2-6.inf13.tinfoil.sh is listed for a different model in "
        "attestation targets for provider tinfoil"
    ) in result["issues"]


def test_preflight_rejects_tinfoil_model_target_with_directory_repo_mismatch(
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
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6 enclave",
                            "type": "tinfoil",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi-wrong-repo.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-5",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6 host "
        "kimi-k2-6.inf13.tinfoil.sh repo tinfoilsh/confidential-kimi-k2-5 "
        "does not match attestation targets for provider tinfoil"
    ) in result["issues"]


def test_preflight_rejects_tinfoil_model_target_with_known_host_repo_mismatch_when_directory_omits_repo(
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
                            "id": "tinfoil-llama",
                            "label": "Llama 3.3 70B enclave",
                            "type": "tinfoil",
                            "host": "llama3-3-70b.tinfoil.containers.tinfoil.dev",
                        },
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-llama-wrong-repo.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/llama-3.3-70b"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/llama-3.3-70b": {
                            "host": "llama3-3-70b.tinfoil.containers.tinfoil.dev",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is False
    assert (
        "model_attestation_targets for tinfoil/llama-3.3-70b host "
        "llama3-3-70b.tinfoil.containers.tinfoil.dev repo "
        "tinfoilsh/confidential-kimi-k2-6-b200 does not match attestation "
        "targets for provider tinfoil"
    ) in result["issues"]


def test_preflight_rejects_tinfoil_model_target_failed_in_attestation_results(
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
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6 enclave",
                            "type": "tinfoil",
                            "url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "failed",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                    "error": "Trust decision failed: TLS certificate binding failed",
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "model_attestation_targets for tinfoil/kimi-k2-6 host "
        "kimi-k2-6.inf13.tinfoil.sh is not verified in "
        "confidential-inference attestation results" in issue
        for issue in result["issues"]
    )


def test_preflight_rejects_tinfoil_model_target_release_digest_mismatch(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "verified",
                    "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": "sha256:" + ("d" * 64),
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "model_attestation_targets for tinfoil/kimi-k2-6 release digest "
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd "
        "does not match policy release digests" in issue
        for issue in result["issues"]
    )


def test_preflight_maps_attestation_result_ids_to_target_hosts(
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
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6 enclave",
                            "type": "tinfoil",
                            "url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi.json",
        {
            "base_url": "https://direct.tinfoil.example/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "attestation_url": "https://direct.tinfoil.example/.well-known/tinfoil-attestation",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=targets,
        strict_attestation_targets=True,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert (
        "attestation results for provider tinfoil are not present"
        not in result["issues"]
    )
    assert not any(
        "model_attestation_targets for tinfoil/kimi-k2-6" in issue
        for issue in result["issues"]
    )


def test_preflight_maps_router_attestation_result_id_to_target_host(
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
                            "label": "Router (inference.tinfoil.sh)",
                            "type": "tinfoil",
                            "url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                            "host": "inference.tinfoil.sh",
                        },
                        {
                            "id": "tinfoil-kimi-k2-6",
                            "label": "Kimi K2.6 enclave",
                            "type": "tinfoil",
                            "url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "verified",
                    "tls_matches": False,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-kimi.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=targets,
        strict_attestation_targets=True,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "Tinfoil provider host inference.tinfoil.sh is not verified in "
        "confidential-inference attestation results" in issue
        for issue in result["issues"]
    )
    assert not any(
        "Tinfoil provider host inference.tinfoil.sh is not present" in issue
        for issue in result["issues"]
    )


def test_preflight_rejects_tinfoil_router_failed_in_attestation_results(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "failed",
                    "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "error": "Trust decision failed: TLS certificate binding failed",
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-router.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "Tinfoil provider host inference.tinfoil.sh is not verified in "
        "confidential-inference attestation results" in issue
        for issue in result["issues"]
    )


def test_preflight_rejects_tinfoil_router_release_digest_mismatch(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "verified",
                    "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": "sha256:" + ("d" * 64),
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-router.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "Tinfoil provider host inference.tinfoil.sh release digest "
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd "
        "does not match policy release digests" in issue
        for issue in result["issues"]
    )


def test_preflight_rejects_tinfoil_router_without_tls_binding_in_attestation_results(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "verified",
                    "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "tls_matches": False,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                    "tls_matches": True,
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-router.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "Tinfoil provider host inference.tinfoil.sh is not verified in "
        "confidential-inference attestation results" in issue
        and "TLS certificate binding was not verified" in issue
        for issue in result["issues"]
    )


def test_preflight_allows_tinfoil_ehbp_policy_without_tls_binding_in_attestation_results(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "verified",
                    "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": False,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": False,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-router.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "transport_security": "ehbp",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is True
    assert not any(
        "TLS certificate binding was not verified" in issue
        for issue in result["issues"]
    )


def test_preflight_rejects_malformed_tinfoil_transport_security_policy(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil-router.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "transport_security": "ehbp-only",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "Tinfoil transport_security must be tls or ehbp" in result["issues"]


def test_preflight_rejects_tinfoil_model_without_code_transparency_in_attestation_results(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "verified",
                    "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": False,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-router.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "model_attestation_targets for tinfoil/kimi-k2-6 host "
        "kimi-k2-6.inf13.tinfoil.sh is not verified in "
        "confidential-inference attestation results" in issue
        and "sigstore_match was not verified" in issue
        for issue in result["issues"]
    )


def test_preflight_rejects_tinfoil_model_missing_transparency_fields_in_attestation_results(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-router",
                    "status": "verified",
                    "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_match": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
                {
                    "provider": "tinfoil",
                    "id": "tinfoil-kimi-k2-6",
                    "status": "verified",
                    "source_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "tls_matches": True,
                    "sigstore_bundle_verified": True,
                    "images_verified": True,
                },
            ]
        },
    )
    path = _write_json(
        tmp_path / "tinfoil-router.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_ATTESTATION_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "attestation_url": "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": VALID_ATTESTATION_DIGEST,
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert any(
        "model_attestation_targets for tinfoil/kimi-k2-6 host "
        "kimi-k2-6.inf13.tinfoil.sh is not verified in "
        "confidential-inference attestation results" in issue
        and "sigstore_match is missing" in issue
        for issue in result["issues"]
    )


def test_preflight_maps_ppq_private_attestation_result_id_to_target_host(
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
                            "id": "ppq-private-router",
                            "label": "Private Mode Router",
                            "type": "ppq",
                            "url": "https://api.ppq.ai/private",
                            "host": "api.ppq.ai",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-private-router",
                    "status": "verified",
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "verified"


def test_preflight_downgrades_stale_attestation_result_rows(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "status": "verified",
                    "ts": "2026-01-01T00:00:00+00:00",
                }
            ]
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        max_result_age_seconds=60,
        now=datetime(2026, 1, 1, 0, 2, 1, tzinfo=timezone.utc),
    )

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "attestation result is stale: age_seconds=121 max_age_seconds=60"
    )


def test_preflight_accepts_fresh_attestation_result_last_run_fallback(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "last_run": "2026-01-01T00:00:00.123456789+00:00",
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "status": "verified",
                }
            ],
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        max_result_age_seconds=60,
        now=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
    )

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "verified"
    assert "ppq" not in results.provider_host_errors


def test_preflight_downgrades_malformed_attestation_result_row_timestamp(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "last_run": "2026-01-01T00:00:00+00:00",
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "status": "verified",
                    "ts": "not-a-timestamp",
                }
            ],
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        max_result_age_seconds=60,
        now=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
    )

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "attestation result timestamp is missing or invalid"
    )


def test_preflight_downgrades_duplicate_attestation_result_host_rows(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {"provider": "ppq", "host": "api.ppq.ai", "status": "failed"},
                {"provider": "ppq", "host": "api.ppq.ai", "status": "verified"},
            ],
        },
    )

    results = preflight.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "duplicate attestation result rows for provider ppq host api.ppq.ai"
    )


def test_preflight_downgrades_duplicate_attestation_result_model_rows(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "failed",
                },
                {
                    "provider": "ppq",
                    "host": "api2.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                },
            ],
        },
    )

    results = preflight.load_attestation_results_directory(results_path)

    assert (
        results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    )
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "duplicate attestation result rows for provider ppq model private/gpt-oss-120b"
    )


def test_preflight_downgrades_duplicate_attestation_result_check_ids(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-private-model",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "backend_release_digest": VALID_VERIFIER_DIGEST,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                },
                {
                    "provider": "ppq",
                    "id": "ppq-private-model",
                    "host": "api.ppq.ai",
                    "model": "private/llama3-3-70b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "backend_release_digest": VALID_VERIFIER_DIGEST,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                },
            ]
        },
    )

    results = preflight.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "duplicate attestation result rows for provider ppq check ppq-private-model"
    )
    assert results.provider_model_statuses["ppq"] == {
        "private/gpt-oss-120b": "failed",
        "private/llama3-3-70b": "failed",
    }
    assert results.provider_model_errors["ppq"] == {
        "private/gpt-oss-120b": (
            "duplicate attestation result rows for provider ppq check "
            "ppq-private-model"
        ),
        "private/llama3-3-70b": (
            "duplicate attestation result rows for provider ppq check "
            "ppq-private-model"
        ),
    }


def test_preflight_accepts_ppq_private_model_rows_on_same_provider_host_with_bare_digests(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST.removeprefix(
                        "sha256:"
                    ),
                    "backend_release_digest": VALID_VERIFIER_DIGEST.removeprefix(
                        "sha256:"
                    ),
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                },
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/llama3-3-70b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST.removeprefix(
                        "sha256:"
                    ),
                    "backend_release_digest": VALID_VERIFIER_DIGEST.removeprefix(
                        "sha256:"
                    ),
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                },
            ],
        },
    )

    results = preflight.load_attestation_results_directory(results_path)

    assert (
        results.provider_model_statuses["ppq"]["private/gpt-oss-120b"]
        == "verified"
    )
    assert (
        results.provider_model_statuses["ppq"]["private/llama3-3-70b"]
        == "verified"
    )
    assert "ppq" not in results.provider_model_errors


def test_preflight_maps_ppq_private_attestation_result_id_to_model(
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
                            "label": "GPT-OSS 120B",
                            "type": "ppq",
                            "url": "https://api.ppq.ai/private",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                        }
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-gpt-oss-120b",
                    "status": "verified",
                }
            ]
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "release_digest is missing or invalid in "
        "confidential-inference attestation results"
    )


def test_preflight_rejects_attestation_result_model_mismatch_with_target_id(
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
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-gpt-oss-120b",
                    "host": "api.ppq.ai",
                    "model": "private/llama3-3-70b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "backend_release_digest": VALID_VERIFIER_DIGEST,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    targets_directory = preflight.load_attestation_targets_directory(targets_path)
    results_directory = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert results_directory.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results_directory.provider_host_errors["ppq"]["api.ppq.ai"]
        == "model does not match confidential-inference attestation target"
    )
    assert (
        results_directory.provider_model_statuses["ppq"]["private/llama3-3-70b"]
        == "failed"
    )
    assert (
        results_directory.provider_model_errors["ppq"]["private/llama3-3-70b"]
        == "model does not match confidential-inference attestation target"
    )


def test_preflight_rejects_attestation_result_host_mismatch_with_target_id(
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
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-gpt-oss-120b",
                    "host": "api2.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": VALID_ATTESTATION_DIGEST,
                    "backend_release_digest": VALID_VERIFIER_DIGEST,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    targets_directory = preflight.load_attestation_targets_directory(targets_path)
    results_directory = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert results_directory.provider_host_statuses["ppq"]["api2.ppq.ai"] == "failed"
    assert (
        results_directory.provider_host_errors["ppq"]["api2.ppq.ai"]
        == "host does not match confidential-inference attestation target"
    )
    assert (
        results_directory.provider_model_statuses["ppq"]["private/gpt-oss-120b"]
        == "failed"
    )
    assert (
        results_directory.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "host does not match confidential-inference attestation target"
    )


def test_preflight_strict_results_rejects_wrong_ppq_private_model_result(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/llama3-3-70b",
                    "status": "verified",
                }
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "release_digest": VALID_VERIFIER_DIGEST,
                    "measurement_digest": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        provider_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert any(
        "PPQ private model private/gpt-oss-120b is not verified in "
        "confidential-inference attestation results"
        in issue
        or "PPQ private model private/gpt-oss-120b is not present in "
        "confidential-inference attestation results"
        in issue
        for issue in result["issues"]
    )


def test_preflight_strict_results_require_ppq_private_model_row_even_without_targets(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "status": "verified",
                }
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_VERIFIER_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "private/gpt-oss-120b": {
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                            "expected_release_digest": VALID_VERIFIER_DIGEST,
                        }
                    },
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        provider_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert any(
        "PPQ private model private/gpt-oss-120b is not present in "
        "confidential-inference attestation results"
        in issue
        for issue in result["issues"]
    )


def test_preflight_ppq_results_reject_router_only_evidence_without_selected_models() -> (
    None
):
    issues, warnings = preflight.ppq_private_attestation_results_issues(
        {
            "attestation_bundle_url": "https://api.ppq.ai/private",
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_VERIFIER_DIGEST,
            "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
        },
        provider_base_url="https://api.ppq.ai/private/v1",
        model_ids=[],
        attestation_targets_directory=None,
        attestation_results_directory=preflight.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
        strict_attestation_results=True,
    )

    assert warnings == []
    assert (
        "PPQ private selected model evidence is required in "
        "confidential-inference attestation results"
    ) in issues


def test_preflight_strict_results_rejects_ppq_private_release_digest_mismatch(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": "sha256:" + ("1" * 64),
                    "backend_release_digest": "sha256:" + ("2" * 64),
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("3" * 64),
                    "measurement_digest": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "private/gpt-oss-120b": {
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                            "expected_release_digest": "sha256:" + ("4" * 64),
                        }
                    },
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        provider_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert (
        "PPQ private model private/gpt-oss-120b router release digest "
        "sha256:1111111111111111111111111111111111111111111111111111111111111111 "
        "does not match policy release digests"
    ) in result["issues"]
    assert (
        "PPQ private model private/gpt-oss-120b backend release digest "
        "sha256:2222222222222222222222222222222222222222222222222222222222222222 "
        "does not match model_attestation_targets release digests"
    ) in result["issues"]


def test_preflight_strict_results_accepts_ppq_private_matching_release_digests(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": router_digest,
                    "measurement_digest": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "private/gpt-oss-120b": {
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                            "expected_release_digest": backend_digest,
                        }
                    },
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        provider_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_strict_results_rejects_ppq_private_missing_backend_evidence(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": False,
                    "backend_sigstore_bundle_verified": True,
                }
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": router_digest,
                    "measurement_digest": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "private/gpt-oss-120b": {
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                            "expected_release_digest": backend_digest,
                        }
                    },
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        provider_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert (
        "PPQ private model private/gpt-oss-120b is not verified in "
        "confidential-inference attestation results (status=failed): "
        "backend_sigstore_match was not verified in confidential-inference "
        "attestation results"
    ) in result["issues"]


def test_preflight_results_reject_ppq_backend_host_mismatch_with_targets(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
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
        {
            "checks": [
                {
                    "provider": "ppq",
                    "id": "ppq-gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_host": "wrong-backend.tinfoil.example",
                    "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=preflight.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "backend_host does not match confidential-inference attestation target"
    )


def test_preflight_results_reject_ppq_backend_host_mismatch_without_check_id(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
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
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_host": "wrong-backend.tinfoil.example",
                    "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=preflight.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "backend_host does not match confidential-inference attestation target"
    )


def test_preflight_results_reject_ppq_backend_repo_mismatch_without_check_id(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
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
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "backend_repo": "tinfoilsh/confidential-other",
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=preflight.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "backend_repo does not match confidential-inference attestation target"
    )


def test_preflight_results_reject_ppq_host_mismatch_without_check_id(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
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
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api2.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=preflight.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results.provider_host_statuses["ppq"]["api2.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api2.ppq.ai"]
        == "host does not match confidential-inference attestation target"
    )
    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "host does not match confidential-inference attestation target"
    )


def test_preflight_rejects_ppq_result_without_check_id_for_duplicate_model_targets(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "ppq",
                    "checks": [
                        {
                            "id": "ppq-gpt-oss-120b-a",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                            "backend_host": "gpt-oss-120b-a.inf10.tinfoil.sh",
                        },
                        {
                            "id": "ppq-gpt-oss-120b-b",
                            "type": "ppq",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                            "backend_host": "gpt-oss-120b-b.inf10.tinfoil.sh",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_host": "gpt-oss-120b-a.inf10.tinfoil.sh",
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                }
            ]
        },
    )

    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=preflight.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "model has multiple confidential-inference attestation targets; "
        "result row must include a check id"
    )


def test_preflight_strict_results_uses_ppq_model_rows_not_shared_host_status(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                },
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/kimi-k2-6",
                    "status": "unreachable",
                    "error": "GitHub rate limit",
                },
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": router_digest,
                    "measurement_digest": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "private/gpt-oss-120b": {
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                            "expected_release_digest": backend_digest,
                        }
                    },
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        provider_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_ppq_shared_host_uses_any_verified_model_row(
    tmp_path: Path,
) -> None:
    router_digest = "sha256:" + ("1" * 64)
    backend_digest = "sha256:" + ("2" * 64)
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/kimi-k2-6",
                    "status": "unreachable",
                    "error": "GitHub rate limit",
                },
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "model": "private/gpt-oss-120b",
                    "status": "verified",
                    "release_digest": router_digest,
                    "backend_release_digest": backend_digest,
                    "backend_tls_matches": True,
                    "backend_sigstore_match": True,
                    "backend_sigstore_bundle_verified": True,
                    "backend_images_verified": True,
                },
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": router_digest,
                    "measurement_digest": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "private/gpt-oss-120b": {
                            "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-gpt-oss-120b",
                            "expected_release_digest": backend_digest,
                        }
                    },
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        provider_path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_strict_results_require_ppq_private_model_result_when_targets_are_model_aware(
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
                            "label": "GPT-OSS 120B",
                            "type": "ppq",
                            "url": "https://api.ppq.ai/private",
                            "host": "api.ppq.ai",
                            "model": "private/gpt-oss-120b",
                        },
                        {
                            "id": "ppq-llama3-70b",
                            "label": "Llama 3.3 70B",
                            "type": "ppq",
                            "url": "https://api.ppq.ai/private",
                            "host": "api.ppq.ai",
                            "model": "private/llama3-3-70b",
                        },
                    ],
                }
            ]
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "ppq",
                    "host": "api.ppq.ai",
                    "status": "verified",
                }
            ]
        },
    )
    provider_path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "release_digest": VALID_VERIFIER_DIGEST,
                    "measurement_digest": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": ["./verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    targets = preflight.load_attestation_targets_directory(targets_path)
    results = preflight.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )
    result = preflight.validate_provider_policy(
        provider_path,
        attestation_targets_directory=targets,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert any(
        "PPQ private model private/gpt-oss-120b is not present in "
        "confidential-inference attestation results"
        in issue
        for issue in result["issues"]
    )


def test_preflight_strict_results_rejects_missing_ppq_private_directory_result(
    tmp_path: Path,
) -> None:
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "id": "privatemode-api",
                    "status": "failed",
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_private_backend_attestation_policy(
                        "private/gpt-oss-120b"
                    ),
                },
            },
        },
    )

    results = preflight.load_attestation_results_directory(results_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert (
        "attestation results for provider ppq are not present"
        in result["issues"]
    )


def test_preflight_does_not_warn_for_ppq_private_attestation_target_metadata(
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
                            "id": "ppq-api",
                            "type": "health",
                            "url": "https://ppq.ai/docs",
                            "note": "TEE attestation not publicly documented",
                        }
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_private_backend_attestation_policy(
                        "private/gpt-oss-120b"
                    ),
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is True
    assert result["warnings"] == []


def test_preflight_rejects_ppq_private_model_absent_from_provider_catalog(
    tmp_path: Path,
) -> None:
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [
            {
                "slug": "ppq",
                "models": [
                    {"slug": "gpt-oss-120b", "name": "GPT-OSS 120B"},
                    {"slug": "kimi-k2.6", "name": "Kimi K2.6"},
                ],
            }
        ],
    )
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/not-in-catalog"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_private_backend_attestation_policy("private/kimi-k2-6"),
                },
            },
        },
    )

    catalog = preflight.load_provider_model_catalog(catalog_path)
    result = preflight.validate_provider_policy(
        path,
        provider_model_catalog=catalog,
        strict_provider_catalog=True,
    )

    assert result["ok"] is False
    assert (
        "selected model private/not-in-catalog is not present in "
        "confidential-inference provider catalog for ppq"
    ) in result["issues"]


def test_preflight_accepts_ppq_private_dot_dash_catalog_alias(
    tmp_path: Path,
) -> None:
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [{"slug": "ppq", "models": [{"slug": "kimi-k2.6", "name": "Kimi K2.6"}]}],
    )
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/kimi-k2-6"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_private_backend_attestation_policy("private/kimi-k2-6"),
                },
            },
        },
    )

    catalog = preflight.load_provider_model_catalog(catalog_path)
    result = preflight.validate_provider_policy(
        path,
        provider_model_catalog=catalog,
        strict_provider_catalog=True,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_accepts_ppq_private_catalog_descriptive_aliases(
    tmp_path: Path,
) -> None:
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [
            {
                "slug": "ppq",
                "models": [
                    {"slug": "gemma-4-31b", "name": "Gemma 4 31B"},
                    {"slug": "llama-3.3-70b", "name": "Llama 3.3 70B"},
                    {"slug": "qwen3-vl-30b-a3b", "name": "Qwen3 VL 30B A3B"},
                ],
            }
        ],
    )
    catalog = preflight.load_provider_model_catalog(catalog_path)

    assert "gemma4-31b" in catalog.provider_models["ppq"]
    assert "llama3-3-70b" in catalog.provider_models["ppq"]
    assert "qwen3-vl-30b" in catalog.provider_models["ppq"]

    issues, warnings = preflight.provider_catalog_model_issues(
        mode="ppq-private-tee",
        model_ids=[
            "private/gemma4-31b",
            "private/llama3-3-70b",
            "private/qwen3-vl-30b",
        ],
        provider_model_catalog=catalog,
        strict_provider_catalog=True,
    )

    assert issues == []
    assert warnings == []


def test_preflight_does_not_apply_ppq_descriptive_aliases_to_privatemode_catalog(
    tmp_path: Path,
) -> None:
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [
            {
                "slug": "privatemode",
                "models": [{"slug": "qwen3-coder-30b-a3b"}],
            }
        ],
    )
    catalog = preflight.load_provider_model_catalog(catalog_path)

    issues, warnings = preflight.provider_catalog_model_issues(
        mode="privatemode",
        model_ids=["privatemode/qwen3-coder-30b"],
        provider_model_catalog=catalog,
        strict_provider_catalog=True,
    )

    assert warnings == []
    assert (
        "selected model privatemode/qwen3-coder-30b is not present in "
        "confidential-inference provider catalog for privatemode"
    ) in issues


def test_provider_model_support_summary_uses_catalog_descriptive_aliases(
    tmp_path: Path,
) -> None:
    catalog_path = _write_json(
        tmp_path / "providers.json",
        [
            {
                "slug": "ppq",
                "models": [
                    {"slug": "gemma-4-31b"},
                    {"slug": "llama-3.3-70b"},
                    {"slug": "qwen3-vl-30b-a3b"},
                ],
            }
        ],
    )
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "ppq",
                    "checks": [
                        {"id": "ppq-gemma4", "model": "private/gemma4-31b"},
                        {"id": "ppq-llama", "model": "private/llama3-3-70b"},
                        {"id": "ppq-qwen3-vl", "model": "private/qwen3-vl-30b"},
                    ],
                }
            ]
        },
    )

    summary = preflight.provider_model_support_summary(
        provider_model_catalog=preflight.load_provider_model_catalog(catalog_path),
        attestation_targets_directory=preflight.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert summary["ppq"]["catalog_models_without_attestation_targets"] == []


def test_preflight_warns_privatemode_attestation_directory_is_reference_only(
    tmp_path: Path,
) -> None:
    targets_path = _write_json(
        tmp_path / "attestation-targets.json",
        {
            "providers": [
                {
                    "slug": "privatemode",
                    "checks": [
                        {
                            "id": "privatemode-api",
                            "type": "privatemode",
                            "url": "https://docs.privatemode.ai",
                            "note": "Public manifest reference values only",
                        }
                    ],
                }
            ]
        },
    )
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_trust_tier": "app-e2ee",
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    directory = preflight.load_attestation_targets_directory(targets_path)
    result = preflight.validate_provider_policy(
        path,
        attestation_targets_directory=directory,
        strict_attestation_targets=True,
    )

    assert result["ok"] is True
    assert (
        "confidential-inference attestation targets contain Privatemode "
        "reference metadata only; Privatemode still requires live proxy "
        "Contrast attestation evidence"
    ) in result["warnings"]


def test_preflight_strict_results_require_privatemode_public_policy_host(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_url": "https://cdn.confidential.cloud/privatemode/v2/manifest.json",
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_trust_tier": "app-e2ee",
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "host": "docs.privatemode.ai",
                    "status": "verified",
                }
            ]
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert (
        "Privatemode provider host cdn.confidential.cloud is not present in "
        "confidential-inference attestation results"
    ) in result["issues"]


def test_preflight_strict_results_reject_privatemode_result_without_policy_bound_proof(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_url": "https://cdn.confidential.cloud/privatemode/v2/manifest.json",
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_trust_tier": "app-e2ee",
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "host": "cdn.confidential.cloud",
                    "model": "privatemode/model",
                    "status": "verified",
                    "trust_tier": "app-e2ee",
                }
            ]
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert (
        "Privatemode provider host cdn.confidential.cloud is not verified in "
        "confidential-inference attestation results (status=failed): "
        "Privatemode selected model evidence is missing policy-bound proof details"
    ) in result["issues"]


def test_preflight_strict_results_accept_privatemode_policy_bound_proof(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_url": "https://cdn.confidential.cloud/privatemode/v2/manifest.json",
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_trust_tier": "app-e2ee",
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "host": "cdn.confidential.cloud",
                    "model": "privatemode/model",
                    "status": "verified",
                    **_privatemode_full_result_details(),
                }
            ]
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_strict_results_accept_privatemode_policy_bound_proof_without_public_host(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_trust_tier": "app-e2ee",
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "model": "privatemode/model",
                    "host": "local-proxy.privatemode.internal",
                    "status": "verified",
                    **_privatemode_full_result_details(),
                }
            ]
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_strict_results_reject_privatemode_duplicate_selected_model_ids(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_url": "https://cdn.confidential.cloud/privatemode/v2/manifest.json",
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_trust_tier": "app-e2ee",
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "host": "cdn.confidential.cloud",
                    "model": "privatemode/model",
                    "status": "verified",
                    **{
                        **_privatemode_full_result_details(),
                        "selected_model_ids": [
                            "privatemode/model",
                            "PRIVATEMODE/model",
                        ],
                    },
                }
            ]
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert (
        "Privatemode provider host cdn.confidential.cloud is not verified in "
        "confidential-inference attestation results (status=failed): "
        "Privatemode selected model evidence selected_model_ids must exactly "
        "match the selected model"
    ) in result["issues"]


def test_preflight_strict_results_require_privatemode_model_row(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_url": "https://cdn.confidential.cloud/privatemode/v2/manifest.json",
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "host": "cdn.confidential.cloud",
                    "status": "verified",
                    "trust_tier": "app-e2ee",
                }
            ]
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert (
        "Privatemode model privatemode/model is not present in "
        "confidential-inference attestation results"
    ) in result["issues"]


def test_preflight_strict_results_reject_privatemode_verified_without_app_e2ee_tier(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_url": "https://cdn.confidential.cloud/privatemode/v2/manifest.json",
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_coordinator_measurement": "sha256:" + ("1" * 64),
                    "expected_secret_service_measurement": "sha256:" + ("2" * 64),
                    "expected_ai_worker_measurement": "sha256:" + ("3" * 64),
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_key_release_binding": "sha256:" + ("4" * 64),
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )
    results_path = _write_json(
        tmp_path / "attestation-results.json",
        {
            "checks": [
                {
                    "provider": "privatemode",
                    "host": "cdn.confidential.cloud",
                    "status": "verified",
                    "trust_tier": "none",
                }
            ]
        },
    )
    results = preflight.load_attestation_results_directory(results_path)

    result = preflight.validate_provider_policy(
        path,
        attestation_results_directory=results,
        strict_attestation_results=True,
    )

    assert result["ok"] is False
    assert (
        "Privatemode provider host cdn.confidential.cloud is not verified in "
        "confidential-inference attestation results (status=failed): "
        "Privatemode verified attestation results must have trust_tier=app-e2ee"
    ) in result["issues"]


def test_preflight_accepts_tinfoil_native_sev_snp_measurement_pin(
    tmp_path: Path,
) -> None:
    native_measurement = "b1" * 48
    normalized_measurement = hashlib.sha256(
        bytes.fromhex(native_measurement)
    ).hexdigest()
    path = _write_json(
        tmp_path / "tinfoil-native-measurement.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "expected_code_measurement_fingerprint": native_measurement,
                    "allowed_code_measurement_fingerprints": [
                        "sha256:" + normalized_measurement
                    ],
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_code_measurement_fingerprint": (
                                VALID_EHBP_CODE_MEASUREMENT
                            ),
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_rejects_embedded_secret_in_verifier_command_argument(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil-secret-arg.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": [
                        VALID_VERIFIER_COMMAND,
                        "--token=sk-inline-secret",
                    ],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "inline secret-like policy value is not allowed: $.verifier_command[1]"
    ) in result["issues"]


def test_preflight_verify_artifacts_rejects_missing_verifier_artifact(
    tmp_path: Path,
) -> None:
    verifier_path = tmp_path / "missing-verifier"
    path = _write_json(
        tmp_path / "tinfoil-missing-verifier.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": [str(verifier_path)],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path, verify_artifacts=True)

    assert result["ok"] is False
    assert any(
        issue.startswith("failed to hash verifier artifact:")
        for issue in result["issues"]
    )


def test_preflight_verify_artifacts_rejects_verifier_digest_mismatch(
    tmp_path: Path,
) -> None:
    verifier_path = tmp_path / "verifier"
    verifier_path.write_bytes(b"test verifier binary")
    path = _write_json(
        tmp_path / "tinfoil-wrong-verifier-digest.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": [str(verifier_path)],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path, verify_artifacts=True)

    assert result["ok"] is False
    assert any(
        issue.startswith("verifier command digest mismatch:")
        for issue in result["issues"]
    )


def test_preflight_verify_artifacts_accepts_matching_verifier_digest(
    tmp_path: Path,
) -> None:
    verifier_path = tmp_path / "verifier"
    verifier_path.write_bytes(b"test verifier binary")
    verifier_digest = "sha256:" + hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    path = _write_json(
        tmp_path / "tinfoil-matching-verifier-digest.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": [str(verifier_path)],
                    "verifier_command_digest": verifier_digest,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path, verify_artifacts=True)

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_verify_artifacts_accepts_repo_root_relative_verifier_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_path = tmp_path / "build" / "confidential-verifiers" / "verifier"
    verifier_path.parent.mkdir(parents=True)
    verifier_path.write_bytes(b"repo root verifier binary")
    verifier_digest = "sha256:" + hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    policy_dir = tmp_path / "examples" / "confidential-routing"
    policy_dir.mkdir(parents=True)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    path = _write_json(
        policy_dir / "tinfoil-matching-verifier-digest.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": [
                        "build/confidential-verifiers/verifier"
                    ],
                    "verifier_command_digest": verifier_digest,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path, verify_artifacts=True)

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_rejects_secret_query_parameter_in_policy_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil-secret-query.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "attestation_url": (
                        "https://inference.tinfoil.sh/.well-known/"
                        "tinfoil-attestation?api_key=secret-token"
                    ),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "credential-bearing policy URL query is not allowed: $.attestation_url"
    ) in result["issues"]


def test_preflight_rejects_policy_without_confidentiality_mode(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "missing-mode.json",
        {
            "confidentiality": {
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "mode is required; expected tinfoil, ppq-private-tee, or privatemode"
        in result["issues"]
    )


def test_preflight_rejects_unknown_confidentiality_mode(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "unknown-mode.json",
        {
            "confidentiality": {
                "mode": "plain-openai",
                "model_ids": ["secure-model"],
                "policy": {
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "unknown confidentiality mode: plain-openai; expected tinfoil, ppq-private-tee, or privatemode"
        in result["issues"]
    )


def test_preflight_rejects_tinfoil_policy_without_artifact_identity(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "expected_release_digest or allowed_release_digests is required"
        in result["issues"]
    )


def test_preflight_rejects_tinfoil_policy_with_code_measurement_only(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "expected_release_digest or allowed_release_digests is required"
        in result["issues"]
    )


def test_preflight_rejects_tinfoil_policy_with_malformed_release_digest(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "not-a-sha256-digest",
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "release digest policy values must be sha256 digests" in result["issues"]


def test_preflight_rejects_malformed_release_digest_alias_when_expected_is_valid(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "allowed_release_digests": [{"unexpected": "object"}],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "release digest policy values must be sha256 digests" in result["issues"]


def test_preflight_rejects_blank_release_digest_alias_when_expected_is_valid(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "allowed_release_digest": "",
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "release digest policy values must be sha256 digests" in result["issues"]


def test_preflight_rejects_tinfoil_policy_with_conflicting_release_digest_allowlist(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "allowed_release_digests": ["sha256:" + ("b" * 64)],
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "release_digest expected value must be allowed" in result["issues"]


def test_preflight_rejects_tinfoil_policy_with_conflicting_code_measurement_allowlist(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "allowed_code_measurement_fingerprints": ["sha256:" + ("e" * 64)],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "code_measurement_fingerprint expected value must be allowed"
        in result["issues"]
    )


def test_preflight_rejects_tinfoil_policy_with_non_string_repo_alias(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_repo": {"name": "tinfoilsh/confidential-model-router"},
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "expected_repo must be a non-empty string" in result["issues"]


def test_preflight_rejects_tinfoil_policy_with_malformed_measurement_pin(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "allowed_enclave_measurement_fingerprints": ["enclave-fp"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "measurement fingerprint policy values must be sha256 digests "
        "or native SEV-SNP measurements" in (result["issues"])
    )


def test_preflight_rejects_tinfoil_credential_bearing_urls(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://user:pass@inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "attestation_url": "https://token@inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "hpke_keys_url": "https://token@inference.tinfoil.sh/.well-known/hpke-keys",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "provider base_url must not include credentials" in result["issues"]
    assert "attestation_url must not include credentials" in result["issues"]
    assert "hpke_keys_url must not include credentials" in result["issues"]


def test_preflight_rejects_tinfoil_non_https_provider_base_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "http://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "provider base_url must use https" in result["issues"]


def test_preflight_rejects_tinfoil_provider_base_url_without_host(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://:443/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "provider base_url must include a host" in result["issues"]


def test_preflight_rejects_tinfoil_policy_host_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "expected_enclave_host": "other.tinfoil.sh",
                    "attestation_url": "https://other.tinfoil.sh/.well-known/tinfoil-attestation",
                    "hpke_keys_url": "https://other.tinfoil.sh/.well-known/hpke-keys",
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "expected_enclave_host must match provider base_url host" in result["issues"]
    assert "attestation_url host must match provider base_url host" in result["issues"]
    assert "hpke_keys_url host must match provider base_url host" in result["issues"]


def test_preflight_rejects_tinfoil_conflicting_identity_aliases(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_repo": "attacker/conflicting-router",
                    "enclave_host": "inference.tinfoil.sh",
                    "expected_enclave_host": "other.tinfoil.sh",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "repo aliases must match" in result["issues"]
    assert "enclave host aliases must match" in result["issues"]


def test_preflight_rejects_tinfoil_non_string_policy_urls(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "expected_enclave_host": {"host": "inference.tinfoil.sh"},
                    "attestation_url": {
                        "url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation"
                    },
                    "hpke_keys_url": 123,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "expected_enclave_host must be a string" in result["issues"]
    assert "attestation_url must be a string" in result["issues"]
    assert "hpke_keys_url must be a string" in result["issues"]


def test_preflight_parses_provider_settings_json_string_for_ppq_private(
    tmp_path: Path,
) -> None:
    provider_settings = {
        "confidentiality": {
            "mode": "ppq-private-tee",
            "model_ids": ["private/gpt-oss-120b"],
            "policy": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "repo": "ppq-ai/private-tee",
                "allowed_release_digests": ["sha256:" + ("a" * 64)],
                "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                "proxy_binary_path": VALID_PPQ_PROXY_PATH,
                "verifier_command": VALID_VERIFIER_COMMAND,
                "verifier_command_digest": VALID_VERIFIER_DIGEST,
                **_ppq_private_backend_attestation_policy("private/gpt-oss-120b"),
            },
        }
    }
    path = _write_json(
        tmp_path / "ppq.json",
        {"provider_settings": json.dumps(provider_settings)},
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is True
    assert result["mode"] == "ppq-private-tee"


def test_preflight_accepts_ppq_private_without_proxy_binary_digest(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_private_backend_attestation_policy("private/gpt-oss-120b"),
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_accepts_ppq_private_proxy_digest_without_proxy_path(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_private_backend_attestation_policy("private/gpt-oss-120b"),
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_rejects_malformed_provider_settings_json_string(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq.json",
        {"provider_settings": '{"confidentiality":'},
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert any(
        issue.startswith("provider_settings JSON is invalid:")
        for issue in result["issues"]
    )


def test_preflight_rejects_ppq_private_prefix_selectors(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_id_prefixes": ["private/"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "PPQ private model_id_prefixes are not supported; use exact model_ids"
        in result["issues"]
    )
    assert "PPQ private model_ids are required" in result["issues"]


def test_preflight_rejects_ppq_private_non_private_model_ids(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["openai/gpt-5-mini"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "PPQ private model_ids must start with private/" in result["issues"]


def test_preflight_rejects_ppq_private_credential_bearing_urls(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://user:pass@api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://token@api.ppq.ai/private",
                    "hpke_keys_url": "https://token@api.ppq.ai/private/.well-known/hpke-keys",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "provider base_url must not include credentials" in result["issues"]
    assert "attestation_bundle_url must not include credentials" in result["issues"]
    assert "hpke_keys_url must not include credentials" in result["issues"]


def test_preflight_rejects_ppq_private_non_https_provider_base_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "http://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "provider base_url must use https" in result["issues"]


def test_preflight_rejects_ppq_private_provider_base_url_invalid_port(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai:bad/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "provider base_url must include a valid port" in result["issues"]


def test_preflight_rejects_ppq_private_non_private_attestation_bundle_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/attestation",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "attestation_bundle_url must include a private path segment" in result["issues"]
    )


def test_preflight_rejects_ppq_private_non_private_hpke_keys_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "hpke_keys_url": "https://api.ppq.ai/.well-known/hpke-keys",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "hpke_keys_url must include a private path segment" in result["issues"]


def test_preflight_rejects_ppq_private_policy_host_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://attest.ppq.ai/private",
                    "hpke_keys_url": "https://attest.ppq.ai/private/.well-known/hpke-keys",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "attestation_bundle_url host must match provider base_url host"
        in result["issues"]
    )
    assert "hpke_keys_url host must match provider base_url host" in result["issues"]


def test_preflight_rejects_ppq_private_policy_origin_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai:8443/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "hpke_keys_url": (
                        "https://api.ppq.ai/private/.well-known/hpke-keys"
                    ),
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "attestation_bundle_url origin must match provider base_url origin"
        in result["issues"]
    )
    assert (
        "hpke_keys_url origin must match provider base_url origin" in result["issues"]
    )


def test_preflight_rejects_ppq_private_enclave_host_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "expected_enclave_host": "attest.ppq.ai",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "expected_enclave_host must match provider base_url host" in result["issues"]


def test_preflight_rejects_ppq_private_non_string_policy_urls(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": {"url": "https://api.ppq.ai/private"},
                    "hpke_keys_url": 123,
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "attestation_bundle_url must be a string" in result["issues"]
    assert "hpke_keys_url must be a string" in result["issues"]


def test_preflight_rejects_ppq_private_non_private_base_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://api.ppq.ai/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "PPQ private base_url must include a private path segment" in result["issues"]
    )


def test_preflight_rejects_ppq_private_unowned_base_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "base_url": "https://attacker.example/private/v1",
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": ["sha256:" + ("a" * 64)],
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "PPQ private base_url host must be ppq.ai or a ppq.ai subdomain"
        in result["issues"]
    )


def test_preflight_rejects_ppq_private_policy_with_malformed_release_digest(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "allowed_release_digests": [{"unexpected": "object"}],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "release digest policy values must be sha256 digests" in result["issues"]


def test_preflight_rejects_ppq_private_policy_with_malformed_measurement_pin(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "ppq-private.json",
        {
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": "code-fp",
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "measurement fingerprint policy values must be sha256 digests "
        "or native SEV-SNP measurements" in (result["issues"])
    )


@pytest.mark.parametrize(
    ("filename", "document", "expected_issue"),
    [
        (
            "tinfoil.json",
            {
                "confidentiality": {
                    "mode": "tinfoil",
                    "model_ids": ["tinfoil/gpt-secure"],
                    "policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": "sha256:" + ("a" * 64),
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            },
            "verifier_command or tinfoil_verifier_command is required",
        ),
        (
            "ppq-private.json",
            {
                "confidentiality": {
                    "mode": "ppq-private-tee",
                    "model_ids": ["private/gpt-oss-120b"],
                    "policy": {
                        "attestation_bundle_url": "https://api.ppq.ai/private",
                        "repo": "ppq-ai/private-tee",
                        "allowed_release_digests": ["sha256:" + ("a" * 64)],
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            },
            "verifier_command or ppq_private_verifier_command is required",
        ),
        (
            "privatemode.json",
            {
                "confidentiality": {
                    "mode": "privatemode",
                    "model_ids": ["privatemode/model"],
                    "policy": {
                        "dump_requests": False,
                        "shared_prompt_cache": False,
                        "nvidia_ocsp_allow_unknown": False,
                        "nvidia_ocsp_revoked_grace_period_hours": 0,
                        "manifest_digest": "sha256:manifest",
                        "proxy_image_digest": "sha256:proxy",
                        "expected_workload_sans": ["workload-kimi-k2-6"],
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            },
            "verifier_command or privatemode_verifier_command is required",
        ),
    ],
)
def test_preflight_rejects_provider_policy_without_verifier_command(
    tmp_path: Path,
    filename: str,
    document: dict[str, object],
    expected_issue: str,
) -> None:
    path = _write_json(tmp_path / filename, document)

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert expected_issue in result["issues"]


def test_preflight_rejects_malformed_present_verifier_command_alias(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/model"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": {"path": "/tmp/ambiguous-verifier"},
                    "tinfoil_verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "verifier_command must be a command string or array" in result["issues"]


def test_preflight_rejects_conflicting_privatemode_privacy_knob_aliases(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "dumpRequests": True,
                    "shared_prompt_cache": False,
                    "sharedPromptCache": True,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidiaOCSPAllowUnknown": True,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "nvidiaOCSPRevokedGracePeriod": 48,
                    "manifest_digest": "sha256:" + ("a" * 64),
                    "proxy_image_digest": "sha256:" + ("b" * 64),
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "dump_requests must be explicitly false" in result["issues"]
    assert "shared_prompt_cache must be explicitly false" in result["issues"]
    assert "nvidia_ocsp_allow_unknown must be explicitly false" in result["issues"]
    assert "nvidia_ocsp_revoked_grace_period_hours must be 0" in result["issues"]


def test_preflight_rejects_conflicting_privatemode_artifact_digest_aliases(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": "sha256:" + ("a" * 64),
                    "manifestDigest": "sha256:" + ("b" * 64),
                    "proxy_image_digest": "sha256:" + ("c" * 64),
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "proxyImageDigest": "sha256:" + ("d" * 64),
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "manifest_digest aliases must match" in result["issues"]
    assert "proxy_image_digest aliases must match" in result["issues"]


def test_preflight_rejects_privatemode_proxy_image_digest_without_binary_digest(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "proxy_binary_digest is required for full Privatemode proxy artifact "
        "verification"
    ) in result["issues"]


@pytest.mark.parametrize(
    ("filename", "document"),
    [
        (
            "tinfoil.json",
            {
                "confidentiality": {
                    "mode": "tinfoil",
                    "model_ids": ["tinfoil/gpt-secure"],
                    "policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": "sha256:" + ("a" * 64),
                        "verifier_command_digest": "not-a-sha256-digest",
                    },
                }
            },
        ),
        (
            "ppq-private.json",
            {
                "confidentiality": {
                    "mode": "ppq-private-tee",
                    "model_ids": ["private/gpt-oss-120b"],
                    "policy": {
                        "attestation_bundle_url": "https://api.ppq.ai/private",
                        "repo": "ppq-ai/private-tee",
                        "allowed_release_digests": ["sha256:" + ("a" * 64)],
                        "verifier_command_digest": "not-a-sha256-digest",
                    },
                }
            },
        ),
        (
            "privatemode.json",
            {
                "confidentiality": {
                    "mode": "privatemode",
                    "model_ids": ["privatemode/model"],
                    "policy": {
                        "dump_requests": False,
                        "shared_prompt_cache": False,
                        "nvidia_ocsp_allow_unknown": False,
                        "nvidia_ocsp_revoked_grace_period_hours": 0,
                        "manifest_digest": "sha256:manifest",
                        "proxy_image_digest": "sha256:proxy",
                        "expected_workload_sans": ["workload-kimi-k2-6"],
                        "verifier_command_digest": "not-a-sha256-digest",
                    },
                }
            },
        ),
    ],
)
def test_preflight_rejects_malformed_verifier_command_digest(
    tmp_path: Path,
    filename: str,
    document: dict[str, object],
) -> None:
    path = _write_json(tmp_path / filename, document)

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "verifier_command_digest must be a sha256 digest" in result["issues"]


def test_preflight_rejects_malformed_verifier_digest_alias_when_other_alias_is_valid(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": 123,
                    "verifier_binary_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "verifier_command_digest must be a sha256 digest" in result["issues"]


def test_preflight_rejects_unbound_verifier_artifact_path(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": ["/usr/bin/python3", "/tmp/evil-verifier.py"],
                    "verifier_artifact_path": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "verifier_artifact_path must match verifier command executable or argument"
        in result["issues"]
    )


def test_preflight_rejects_blank_verifier_artifact_path(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_artifact_path": "",
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "verifier_artifact_path must be a non-empty string" in result["issues"]


@pytest.mark.parametrize(
    ("policy_fragment", "expected_issue"),
    (
        (
            {"max_verifier_age_seconds": None},
            "max_verifier_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": 300, "max_evidence_age_seconds": None},
            "max_evidence_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": 300, "max_evidence_age_seconds": 301},
            "max_verifier_age_seconds and max_evidence_age_seconds aliases must match",
        ),
        (
            {"max_verifier_age_seconds": 0},
            "max_verifier_age_seconds must be positive",
        ),
    ),
)
def test_preflight_rejects_provider_max_age_policy_aliases(
    tmp_path: Path,
    policy_fragment: dict[str, object],
    expected_issue: str,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **policy_fragment,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert expected_issue in result["issues"]


def test_preflight_rejects_inline_policy_secrets(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "expected_trust_tier": "app-e2ee",
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "nested": {"clientSecret": "secret-value"},
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert any("clientSecret" in issue for issue in result["issues"])


def test_preflight_rejects_nested_policy_url_credentials(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "expected_trust_tier": "app-e2ee",
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "metadata": {
                        "evidence_url": "https://operator:secret-token@example.test"
                    },
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert any("credential-bearing policy URL" in issue for issue in result["issues"])
    assert not any("secret-token" in issue for issue in result["issues"])


def test_preflight_rejects_non_standard_json_policy_constants(tmp_path: Path) -> None:
    path = tmp_path / "tinfoil.json"
    path.write_text(
        """
        {
          "confidentiality": {
            "mode": "tinfoil",
            "model_ids": ["tinfoil/gpt-secure"],
            "policy": {
              "repo": "tinfoilsh/confidential-model-router",
              "expected_release_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "verifier_command": "/usr/local/bin/routstr-tinfoil-verifier",
              "verifier_command_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "non_finite": NaN
            }
          }
        }
        """,
        encoding="utf-8",
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "policy JSON must not contain NaN" in result["issues"]


def test_preflight_rejects_inline_refresh_token(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "credentials": {"refreshToken": "refresh-secret"},
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert any("refreshToken" in issue for issue in result["issues"])


def test_preflight_strict_deployment_rejects_template_placeholder_digests(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil-placeholder-digest.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("c" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("c" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    shape_result = preflight.validate_provider_policy(path)
    strict_result = preflight.validate_provider_policy(
        path,
        reject_placeholder_digests=True,
    )

    assert shape_result["ok"] is True
    assert strict_result["ok"] is False
    assert any(
        issue.startswith("placeholder digest policy value is not allowed:")
        for issue in strict_result["issues"]
    )


def test_preflight_accepts_valid_privatemode_policy(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "expected_trust_tier": "app-e2ee",
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-kimi-k2-6"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is True
    assert result["issues"] == []


def test_preflight_rejects_privatemode_without_expected_trust_tier(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "expected_trust_tier must be app-e2ee" in result["issues"]


def test_preflight_rejects_privatemode_gpu_policy_not_emitted_by_verifier(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "expected_trust_tier": "app-e2ee",
                    "expected_gpu_attestation_policy": "strict-ocsp",
                    "expected_workload_sans": ["workload-model"],
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-model"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "expected_gpu_attestation_policy must be nvidia-ocsp-good-only"
        in result["issues"]
    )


def test_preflight_rejects_privatemode_scalar_expected_workload_selectors(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": "workload-kimi-k2-6",
                    "model_workload_bindings": {
                        "privatemode/model": {"workload_sans": ["workload-kimi-k2-6"]}
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "expected_workload_sans must be a list of non-empty strings" in result["issues"]


def test_preflight_rejects_unprefixed_privatemode_model_ids(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["gpt-oss-120b"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "expected_workload_sans": [
                        "gpt-oss-120b.default.svc.cluster.local"
                    ],
                    "model_workload_bindings": {
                        "gpt-oss-120b": {
                            "workload_sans": [
                                "gpt-oss-120b.default.svc.cluster.local"
                            ]
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "Privatemode model_ids must start with privatemode/" in result["issues"]


def test_preflight_rejects_privatemode_without_model_workload_bindings(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/kimi-k2-6"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "model_workload_bindings is required" in result["issues"]


def test_preflight_rejects_privatemode_model_bound_to_unexpected_workload(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/kimi-k2-6"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "model_workload_bindings": {
                        "privatemode/kimi-k2-6": {
                            "workload_sans": ["workload-gpt-oss-120b"]
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "model_workload_bindings for privatemode/kimi-k2-6 references "
        "workload_sans not listed in expected_workload_sans"
    ) in result["issues"]


def test_preflight_rejects_privatemode_without_expected_workload_policy(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "expected_workload_sans or expected_workload_ids is required"
        in result["issues"]
    )


def test_preflight_rejects_malformed_privatemode_component_policy(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "allowed_secret_service_measurements": [
                        "sha256:" + ("1" * 64),
                        {"unexpected": "object"},
                    ],
                    "expected_gpu_attestation_policy": {"unexpected": "object"},
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "secret_service_measurement policy values must be sha256 digests"
        in result["issues"]
    )
    assert (
        "gpu_attestation_policy policy values must contain only non-empty strings"
        in result["issues"]
    )


def test_preflight_rejects_conflicting_privatemode_component_policy(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "expected_secret_service_measurement": "sha256:" + ("1" * 64),
                    "allowed_secret_service_measurements": ["sha256:" + ("2" * 64)],
                    "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
                    "allowed_gpu_attestation_policies": ["nvidia-ocsp-relaxed"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "secret_service_measurement expected value must be allowed" in result["issues"]
    )
    assert "gpu_attestation_policy expected value must be allowed" in result["issues"]


def test_preflight_rejects_privatemode_placeholder_artifact_digests(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": "sha256:manifest",
                    "proxy_image_digest": "sha256:proxy",
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "manifest_digest must be a sha256 digest" in result["issues"]
    assert "proxy_image_digest must be a sha256 digest" in result["issues"]


def test_preflight_rejects_boolean_privatemode_ocsp_grace_period(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": False,
                    "manifest_digest": "sha256:manifest",
                    "proxy_image_digest": "sha256:proxy",
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "nvidia_ocsp_revoked_grace_period_hours must be 0" in result["issues"]


def test_preflight_rejects_privatemode_remote_proxy_base_url(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://privatemode-proxy:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "Privatemode proxy base_url must use a loopback host" in result["issues"]


def test_preflight_rejects_privatemode_non_http_proxy_base_url(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "ws://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "Privatemode proxy base_url must use http or https" in result["issues"]


def test_preflight_rejects_privatemode_proxy_base_url_invalid_port(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:bad/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "Privatemode proxy base_url must include a valid port" in result["issues"]


def test_preflight_rejects_privatemode_credential_bearing_policy_urls(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": "sha256:manifest",
                    "proxy_image_digest": "sha256:proxy",
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_base_url": "https://token@api.privatemode.ai",
                    "cdn_base_url": "https://token@cdn.privatemode.ai",
                    "manifest_url": "https://token@cdn.privatemode.ai/manifest.json",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "api_base_url must not include credentials" in result["issues"]
    assert "cdn_base_url must not include credentials" in result["issues"]
    assert "manifest_url must not include credentials" in result["issues"]


def test_preflight_rejects_privatemode_non_string_policy_urls(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_base_url": {"url": "https://api.privatemode.ai"},
                    "cdn_base_url": 123,
                    "manifest_url": ["https://cdn.privatemode.ai/manifest.json"],
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "api_base_url must be a string" in result["issues"]
    assert "cdn_base_url must be a string" in result["issues"]
    assert "manifest_url must be a string" in result["issues"]


def test_preflight_rejects_privatemode_non_string_verifier_input_fields(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_key_env": {"env": "PRIVATEMODE_API_KEY"},
                    "api_key_file": 123,
                    "manifest_path": ["/var/lib/privatemode/manifest.json"],
                    "manifest_b64": {"value": "e30="},
                    "manifest_log_dir": ["/var/lib/privatemode/log"],
                    "proxy_binary_path": {"path": "/opt/privatemode/proxy"},
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "api_key_env must be a string" in result["issues"]
    assert "api_key_file must be a string" in result["issues"]
    assert "manifest_path must be a string" in result["issues"]
    assert "manifest_b64 must be a string" in result["issues"]
    assert "manifest_log_dir must be a string" in result["issues"]
    assert "proxy_binary_path must be a string" in result["issues"]


def test_preflight_rejects_privatemode_blank_verifier_input_fields(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/model"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "expected_workload_sans": ["workload-kimi-k2-6"],
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "api_base_url": " ",
                    "cdn_base_url": "",
                    "manifest_url": " ",
                    "api_key_env": " ",
                    "api_key_file": "",
                    "manifest_path": " ",
                    "manifest_b64": "",
                    "manifest_log_dir": " ",
                    "proxy_binary_path": "",
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    for field in (
        "api_base_url",
        "cdn_base_url",
        "manifest_url",
        "api_key_env",
        "api_key_file",
        "manifest_path",
        "manifest_b64",
        "manifest_log_dir",
        "proxy_binary_path",
    ):
        assert f"{field} must be a non-empty string" in result["issues"]


def test_preflight_rejects_confidential_policy_without_model_selectors(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": VALID_EHBP_CODE_MEASUREMENT,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "model_ids or model_id_prefixes is required" in result["issues"]


def test_preflight_rejects_runtime_unloadable_top_level_policy_shape(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "base_url": "https://inference.tinfoil.sh/v1",
            "mode": "tinfoil",
            "model_ids": ["tinfoil/kimi-k2-6"],
            "policy": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_release_digest": "sha256:" + ("a" * 64),
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "tinfoil/kimi-k2-6": {
                        "host": "kimi-k2-6.inf13.tinfoil.sh",
                        "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        "expected_release_digest": "sha256:" + ("b" * 64),
                    }
                },
                "verifier_command": VALID_VERIFIER_COMMAND,
                "verifier_command_digest": VALID_VERIFIER_DIGEST,
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert any("runtime-loadable" in issue for issue in result["issues"])


def test_preflight_rejects_malformed_confidential_model_selectors(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure", 123],
                "model_id_prefixes": [{"unexpected": "object"}],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "model_ids must be a list of non-empty strings" in result["issues"]
    assert "model_id_prefixes must be a list of non-empty strings" in result["issues"]


def test_preflight_rejects_duplicate_confidential_model_selectors(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6", "tinfoil/kimi-k2-6"],
                "model_id_prefixes": ["tinfoil/kimi-", "tinfoil/kimi-"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "model_ids must not contain duplicates" in result["issues"]
    assert "model_id_prefixes must not contain duplicates" in result["issues"]


def test_preflight_rejects_case_variant_duplicate_confidential_model_selectors(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["TINFOIL/kimi-k2-6", "tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        }
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "model_ids must not contain duplicates" in result["issues"]


def test_preflight_rejects_case_variant_duplicate_tinfoil_target_keys(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_release_digest": "sha256:" + ("b" * 64),
                        },
                        "TINFOIL/kimi-k2-6": {
                            "host": "other.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-other",
                            "expected_release_digest": "sha256:" + ("c" * 64),
                        },
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert (
        "model_attestation_targets contains duplicate model target TINFOIL/kimi-k2-6"
    ) in result["issues"]


def test_preflight_rejects_case_variant_duplicate_privatemode_workload_policy(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "privatemode.json",
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/kimi-k2-6"],
                "policy": {
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    "proxy_binary_path": "/opt/privatemode/privatemode-proxy",
                    "expected_workload_sans": [
                        "workload-kimi-k2-6",
                        "WORKLOAD-kimi-k2-6",
                    ],
                    "model_workload_bindings": {
                        "privatemode/kimi-k2-6": {
                            "workload_sans": [
                                "workload-kimi-k2-6",
                                "WORKLOAD-kimi-k2-6",
                            ]
                        },
                        "PRIVATEMODE/kimi-k2-6": {
                            "workload_sans": ["workload-kimi-k2-6"]
                        },
                    },
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "expected_workload_sans must not contain duplicates" in result["issues"]
    assert (
        "model_workload_bindings contains duplicate model binding "
        "PRIVATEMODE/kimi-k2-6"
    ) in result["issues"]
    assert (
        "model_workload_bindings for privatemode/kimi-k2-6: "
        "workload_sans must not contain duplicates"
    ) in result["issues"]


def test_preflight_rejects_scalar_confidential_model_selectors(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": "tinfoil/gpt-secure",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "model_ids must be a list of non-empty strings" in result["issues"]


def test_preflight_rejects_non_string_confidentiality_mode(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "tinfoil.json",
        {
            "mode": "tinfoil",
            "confidentiality": {
                "mode": 123,
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": VALID_VERIFIER_COMMAND,
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            },
        },
    )

    result = preflight.validate_provider_policy(path)

    assert result["ok"] is False
    assert "mode must be a non-empty string" in result["issues"]


def test_preflight_env_checks_pass_for_required_tee_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv(
        "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
        "attested-tls-termination",
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv("ROUTSTR_TEE_PUBLIC_KEY", VALID_ROUTSTR_PUBLIC_KEY)
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ENV_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
        VALID_ENV_VERIFIER_DIGEST,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": (
                    VALID_ENV_ROUTSTR_CODE_MEASUREMENT
                )
            }
        ),
    )

    checks = preflight.env_checks()

    assert not [check for check in checks if check["status"] == "missing"]
    assert {
        "status": "ok",
        "key": "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        "detail": VALID_ROUTSTR_PUBLIC_KEY_DIGEST,
    } in checks


def test_preflight_env_checks_derives_attested_tls_public_key_digest_from_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_key_path = tmp_path / "routstr-public.pem"
    public_key_path.write_text(f"\n{VALID_ROUTSTR_PUBLIC_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv(
        "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
        "attested-tls-termination",
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv("ROUTSTR_TEE_PUBLIC_KEY_PATH", str(public_key_path))
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ENV_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
        VALID_ENV_VERIFIER_DIGEST,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": (
                    VALID_ENV_ROUTSTR_CODE_MEASUREMENT
                )
            }
        ),
    )

    checks = preflight.env_checks()

    assert not [check for check in checks if check["status"] == "missing"]
    assert {
        "status": "ok",
        "key": "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        "detail": VALID_ROUTSTR_PUBLIC_KEY_DIGEST,
    } in checks


def test_preflight_env_checks_rejects_attested_tls_public_key_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv(
        "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
        "attested-tls-termination",
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv("ROUTSTR_TEE_PUBLIC_KEY", VALID_ROUTSTR_PUBLIC_KEY)
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        VALID_ENV_VERIFIER_DIGEST,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ENV_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
        VALID_ENV_VERIFIER_DIGEST,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": (
                    VALID_ENV_ROUTSTR_CODE_MEASUREMENT
                )
            }
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        "detail": (
            "configured digest does not match ROUTSTR_TEE_PUBLIC_KEY; "
            f"expected {VALID_ROUTSTR_PUBLIC_KEY_DIGEST}"
        ),
    } in checks


def test_preflight_env_checks_require_client_to_routstr_confidentiality_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv("ROUTSTR_TEE_PUBLIC_KEY", VALID_ROUTSTR_PUBLIC_KEY)
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
        "detail": (
            "set to attested-tls-termination; inbound EHBP/OHTTP request "
            "decryption is not implemented"
        ),
    } in checks


def test_preflight_env_checks_require_routstr_public_key_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_PUBLIC_KEY",
        "detail": (
            "set ROUTSTR_TEE_PUBLIC_KEY or ROUTSTR_TEE_PUBLIC_KEY_PATH so "
            "local TEE proof binds Routstr's public key"
        ),
    } in checks
    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        "detail": (
            "set ROUTSTR_TEE_PUBLIC_KEY or ROUTSTR_TEE_PUBLIC_KEY_PATH so "
            "attested TLS boundary can publish Routstr's public key digest"
        ),
    } in checks


def test_preflight_env_checks_reject_unsupported_attestation_document_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "raw_report")
    monkeypatch.setenv("ROUTSTR_TEE_PUBLIC_KEY", VALID_ROUTSTR_PUBLIC_KEY)
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT"
        and check["status"] == "missing"
    ]
    assert (
        "unsupported ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT; expected TDX or SEV-SNP evidence format"
        in missing_details
    )


def test_preflight_env_checks_reject_secret_in_verifier_command_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND",
        "/opt/routstr/bin/verify --token=sk-local-verifier-secret",
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_VERIFIER_COMMAND",
        "detail": (
            "inline secret-like policy value is not allowed: "
            "ROUTSTR_TEE_VERIFIER_COMMAND[1]"
        ),
    } in checks


def test_preflight_env_checks_reject_non_standard_verifier_command_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND",
        '["/opt/routstr/bin/verify", NaN]',
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_VERIFIER_COMMAND",
        "detail": "ROUTSTR_TEE_VERIFIER_COMMAND JSON must not contain NaN",
    } in checks


def test_preflight_env_checks_reject_secret_in_attestation_command_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND",
        "/opt/routstr/bin/attest --token=sk-local-attestation-secret",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_ATTESTATION_COMMAND",
        "detail": (
            "inline secret-like policy value is not allowed: "
            "ROUTSTR_TEE_ATTESTATION_COMMAND[1]"
        ),
    } in checks


def test_preflight_env_checks_reject_static_tee_evidence_for_live_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"static quote")
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.delenv("ROUTSTR_TEE_ATTESTATION_COMMAND", raising=False)
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_PATH", str(evidence_path))
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_ATTESTATION_COMMAND",
        "detail": (
            "static evidence is only suitable for fixtures or offline validation; "
            "live deployments should configure a digest-pinned "
            "ROUTSTR_TEE_ATTESTATION_COMMAND"
        ),
    } in checks


def test_preflight_env_checks_reject_secret_in_tdx_quote_command_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TDX_QUOTE_COMMAND",
        "/opt/routstr/bin/tdx-quote --token=sk-local-tdx-secret",
    )
    monkeypatch.setenv("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    assert {
        "status": "missing",
        "key": "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        "detail": (
            "inline secret-like policy value is not allowed: tdx_quote_command[1]"
        ),
    } in checks


@pytest.mark.parametrize(
    ("hpke_key_config_b64", "expected_detail"),
    (
        (
            "not-base64",
            "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64 is not valid base64",
        ),
        (
            base64.b64encode(b"bad").decode("ascii"),
            "invalid Routstr TEE HPKE key config",
        ),
    ),
)
def test_preflight_env_checks_reject_malformed_hpke_key_config_b64(
    hpke_key_config_b64: str,
    expected_detail: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv("ROUTSTR_TEE_HPKE_KEY_CONFIG_B64", hpke_key_config_b64)
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_HPKE_KEY_CONFIG"
        and check["status"] == "missing"
    ]
    assert any(expected_detail in detail for detail in missing_details)


def test_preflight_env_checks_reject_unreadable_hpke_key_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_PATH",
        "/definitely/missing/hpke-key-config",
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_HPKE_KEY_CONFIG"
        and check["status"] == "missing"
    ]
    assert any(
        "failed to read Routstr TEE HPKE key config" in detail
        for detail in missing_details
    )


def test_preflight_env_checks_reject_malformed_tee_command_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        "not-a-sha256-digest",
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
        "not-a-sha256-digest",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = {
        check["key"]: check["detail"]
        for check in checks
        if check["status"] == "missing"
    }
    assert (
        missing_details["ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST"]
        == "must be a sha256 digest"
    )
    assert (
        missing_details["ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST"]
        == "must be a sha256 digest"
    )


def test_preflight_env_checks_reject_placeholder_tee_command_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        "sha256:" + ("b" * 64),
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
        "sha256:" + ("b" * 64),
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = {
        check["key"]: check["detail"]
        for check in checks
        if check["status"] == "missing"
    }
    assert (
        missing_details["ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST"]
        == "placeholder digest is not allowed"
    )
    assert (
        missing_details["ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST"]
        == "placeholder digest is not allowed"
    )


def test_preflight_env_checks_reject_unbound_tee_verifier_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND",
        json.dumps(["/usr/bin/python3", "/tmp/evil-verifier.py"]),
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_ARTIFACT_PATH",
        "/opt/routstr/bin/routstr-tee-go-verifier",
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_ARTIFACT_PATH"
        and check["status"] == "missing"
    ]
    assert (
        "must match ROUTSTR_TEE_VERIFIER_COMMAND executable or argument"
        in missing_details
    )


def test_preflight_env_checks_reject_unbound_tee_attestation_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND",
        json.dumps(["/usr/bin/python3", "/tmp/evil-attester.py"]),
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_ARTIFACT_PATH",
        "/opt/routstr/bin/routstr-tee-attest",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_ATTESTATION_ARTIFACT_PATH"
        and check["status"] == "missing"
    ]
    assert (
        "must match ROUTSTR_TEE_ATTESTATION_COMMAND executable or argument"
        in missing_details
    )


def test_preflight_env_checks_reject_malformed_tee_code_measurement_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps({"expected_routstr_code_measurement": "sha256:routstr-code"}),
    )

    checks = preflight.env_checks()

    missing_details = {
        check["key"]: check["detail"]
        for check in checks
        if check["status"] == "missing"
    }
    assert (
        missing_details["ROUTSTR_TEE_VERIFIER_POLICY_JSON"]
        == "routstr code measurement policy values must be sha256 digests"
    )


def test_preflight_env_checks_reject_placeholder_tee_code_measurement_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps({"expected_routstr_code_measurement": "sha256:" + ("a" * 64)}),
    )

    checks = preflight.env_checks()

    missing_details = {
        check["key"]: check["detail"]
        for check in checks
        if check["status"] == "missing"
    }
    assert (
        missing_details["ROUTSTR_TEE_VERIFIER_POLICY_JSON"]
        == "placeholder digest policy value is not allowed: $.expected_routstr_code_measurement"
    )


def test_preflight_env_checks_reject_non_string_tee_code_measurement_policy_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "allowed_routstr_code_measurements": [
                    VALID_ROUTSTR_CODE_MEASUREMENT,
                    {"unexpected": "object"},
                ]
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = {
        check["key"]: check["detail"]
        for check in checks
        if check["status"] == "missing"
    }
    assert (
        missing_details["ROUTSTR_TEE_VERIFIER_POLICY_JSON"]
        == "routstr code measurement policy values must be sha256 digests"
    )


def test_preflight_env_checks_reject_empty_tee_code_measurement_policy_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "allowed_routstr_code_measurements": [],
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = {
        check["key"]: check["detail"]
        for check in checks
        if check["status"] == "missing"
    }
    assert (
        missing_details["ROUTSTR_TEE_VERIFIER_POLICY_JSON"]
        == "routstr code measurement policy values must be sha256 digests"
    )


@pytest.mark.parametrize(
    ("policy_fragment", "expected_issue"),
    (
        (
            {"max_verifier_age_seconds": "300"},
            "max_verifier_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": None},
            "max_verifier_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": 300, "max_evidence_age_seconds": None},
            "max_evidence_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": 300, "max_evidence_age_seconds": 301},
            "max_verifier_age_seconds and max_evidence_age_seconds aliases must match",
        ),
        (
            {"max_verifier_age_seconds": 0},
            "max_verifier_age_seconds must be positive",
        ),
    ),
)
def test_preflight_env_checks_reject_malformed_tee_max_age_policy(
    policy_fragment: dict[str, object],
    expected_issue: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                **policy_fragment,
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert expected_issue in missing_details


def test_preflight_env_checks_reject_non_object_tee_verifier_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            [{"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}]
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "must be a JSON object" in missing_details


def test_preflight_env_checks_reject_non_string_sev_guest_device_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "sev_guest_device_path": 123,
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "sev_guest_device_path must be a non-empty string" in missing_details


def test_preflight_env_checks_reject_blank_sev_guest_device_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "sev_guest_device_path": "",
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "sev_guest_device_path must be a non-empty string" in missing_details


@pytest.mark.parametrize(
    ("policy_fragment", "expected_issue"),
    (
        (
            {"sev_snp_vmpl": "1"},
            "sev_snp_vmpl must be an integer from 0 to 3",
        ),
        (
            {"sev_snp_vmpl": None},
            "sev_snp_vmpl must be an integer from 0 to 3",
        ),
        (
            {"sev_snp_vmpl": 0, "vmpl": "1"},
            "vmpl must be an integer from 0 to 3",
        ),
        (
            {"sev_snp_vmpl": 0, "vmpl": None},
            "vmpl must be an integer from 0 to 3",
        ),
        (
            {"sev_snp_vmpl": 0, "vmpl": 1},
            "sev_snp_vmpl and vmpl aliases must match",
        ),
    ),
)
def test_preflight_env_checks_reject_malformed_sev_vmpl_policy(
    policy_fragment: dict[str, object],
    expected_issue: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "sev_snp_report")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                **policy_fragment,
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert expected_issue in missing_details


def test_preflight_env_checks_reject_malformed_tdx_quote_command_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "tdx_quote_command": ["/opt/routstr/bin/tdx-quote"],
                "tdx_quote_command_digest": "sha256:tdx",
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "tdx_quote_command_digest must be a sha256 digest" in missing_details


def test_preflight_env_checks_reject_blank_tdx_quote_command_before_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv("ROUTSTR_TDX_QUOTE_COMMAND", "/opt/routstr/bin/tdx-quote")
    monkeypatch.setenv("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "tdx_quote_command": "",
                "tdx_quote_command_digest": VALID_VERIFIER_DIGEST,
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "tdx_quote_command is empty" in missing_details


def test_preflight_env_checks_reject_non_string_tdx_quote_digest_before_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "tdx_quote_command": ["/opt/routstr/bin/tdx-quote"],
                "tdx_quote_command_digest": 123,
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "tdx_quote_command_digest must be a sha256 digest" in missing_details


def test_preflight_env_checks_reject_blank_tdx_quote_digest_before_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "tdx_quote_command": ["/opt/routstr/bin/tdx-quote"],
                "tdx_quote_command_digest": "",
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "tdx_quote_command_digest must be a sha256 digest" in missing_details


def test_preflight_env_checks_reject_unbound_tdx_quote_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "tdx_quote_command": ["/usr/bin/python3", "/tmp/evil-tdx.py"],
                "tdx_quote_artifact_path": "/opt/routstr/bin/tdx-quote-wrapper",
                "tdx_quote_command_digest": VALID_VERIFIER_DIGEST,
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert (
        "tdx_quote_artifact_path must match tdx quote command executable or argument"
        in missing_details
    )


def test_preflight_env_checks_reject_non_string_tdx_quote_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "tdx_quote_command": ["/opt/routstr/bin/tdx-quote"],
                "tdx_quote_command_digest": VALID_VERIFIER_DIGEST,
                "tdx_quote_artifact_path": 123,
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "tdx_quote_artifact_path must be a non-empty string" in missing_details


def test_preflight_env_checks_reject_blank_tdx_quote_artifact_path_before_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv("ROUTSTR_TDX_QUOTE_ARTIFACT_PATH", "/opt/routstr/bin/tdx-quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
                "tdx_quote_command": ["/opt/routstr/bin/tdx-quote"],
                "tdx_quote_command_digest": VALID_VERIFIER_DIGEST,
                "tdx_quote_artifact_path": "",
            }
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert "tdx_quote_artifact_path must be a non-empty string" in missing_details


def test_preflight_env_checks_require_tdx_quote_command_for_tdx_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "tdx_quote")
    monkeypatch.setenv(
        "ROUTSTR_TEE_HPKE_KEY_CONFIG_B64",
        VALID_ROUTSTR_HPKE_KEY_CONFIG_B64,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_COMMAND", "/opt/routstr/bin/attest")
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
        VALID_ATTESTATION_DIGEST,
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/bin/verify")
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST", VALID_VERIFIER_DIGEST)
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
        json.dumps(
            {"expected_routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT}
        ),
    )

    checks = preflight.env_checks()

    missing_details = [
        check["detail"]
        for check in checks
        if check["key"] == "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
        and check["status"] == "missing"
    ]
    assert (
        "tdx_quote_command or ROUTSTR_TDX_QUOTE_COMMAND is required for TDX quote generation"
        in missing_details
    )
    assert (
        "tdx_quote_command_digest or ROUTSTR_TDX_QUOTE_COMMAND_DIGEST is required for TDX quote generation"
        in missing_details
    )
