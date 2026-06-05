from __future__ import annotations

import base64
import json
import os
import sys


def main() -> int:
    behavior = os.environ.get("ROUTSTR_TEE_ATTESTATION_STUB_BEHAVIOR", "verified")
    if behavior == "failure_secret":
        sys.stderr.write(
            "fixture attestation failure SECRET_PROMPT=hello "
            "sk-live-secret report_data=raw"
        )
        return 9

    payload = json.load(sys.stdin)
    if payload.get("schema_version") != "routstr-tee-attestation-request-v1":
        raise ValueError("unexpected schema_version")

    quote = {
        "schema_version": "fixture-routstr-tee-quote-v1",
        "attestation_document_format": payload.get("attestation_document_format"),
        "report_data_digest": payload.get("tee_report_data_digest"),
        "report_data_hex": payload.get("tee_report_data_hex"),
        "verification_nonce": payload.get("verification_nonce"),
    }
    quote_bytes = json.dumps(
        quote, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    result = {
        "schema_version": "routstr-tee-attestation-document-v1",
        "attestation_document_b64": base64.b64encode(quote_bytes).decode("ascii"),
        "attestation_document_format": payload.get("attestation_document_format"),
        "tee_report_data_digest": payload.get("tee_report_data_digest"),
        "tee_report_data_hex": payload.get("tee_report_data_hex"),
    }
    if behavior == "numeric_document_b64":
        result["attestation_document_b64"] = 1234
    if behavior == "numeric_document_format":
        result["attestation_document_format"] = 0
    json.dump(result, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
