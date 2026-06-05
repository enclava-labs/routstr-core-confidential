from routstr.upstream.ehbp import (
    AEAD_AES_256_GCM,
    KDF_HKDF_SHA256,
    KEM_X25519_HKDF_SHA256,
    parse_ehbp_key_config,
)


def _valid_config(public_key: bytes = b"\x22" * 32) -> bytes:
    return (
        b"\x00"
        + KEM_X25519_HKDF_SHA256.to_bytes(2, "big")
        + public_key
        + (4).to_bytes(2, "big")
        + KDF_HKDF_SHA256.to_bytes(2, "big")
        + AEAD_AES_256_GCM.to_bytes(2, "big")
    )


def test_parse_ehbp_key_config_accepts_reference_suite() -> None:
    key = parse_ehbp_key_config(_valid_config())

    assert key.key_id == 0
    assert key.kem_id == KEM_X25519_HKDF_SHA256
    assert key.public_key == b"\x22" * 32
    assert key.public_key_digest.startswith("sha256:")
    assert key.cipher_suites == ((KDF_HKDF_SHA256, AEAD_AES_256_GCM),)


def test_parse_ehbp_key_config_rejects_unsupported_first_suite() -> None:
    invalid = (
        b"\x00"
        + KEM_X25519_HKDF_SHA256.to_bytes(2, "big")
        + (b"\x22" * 32)
        + (4).to_bytes(2, "big")
        + (0x0002).to_bytes(2, "big")
        + AEAD_AES_256_GCM.to_bytes(2, "big")
    )

    try:
        parse_ehbp_key_config(invalid)
    except ValueError as exc:
        assert "unsupported EHBP cipher suite" in str(exc)
    else:
        raise AssertionError("expected parser to reject unsupported suite")


def test_parse_ehbp_key_config_rejects_trailing_bytes() -> None:
    try:
        parse_ehbp_key_config(_valid_config() + b"extra")
    except ValueError as exc:
        assert "trailing bytes" in str(exc)
    else:
        raise AssertionError("expected parser to reject trailing bytes")
