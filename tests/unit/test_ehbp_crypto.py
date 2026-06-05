from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from routstr.upstream.ehbp import (
    AEAD_AES_256_GCM,
    KDF_HKDF_SHA256,
    KEM_X25519_HKDF_SHA256,
    decrypt_ehbp_request_body_for_test,
    decrypt_ehbp_response_body,
    derive_ehbp_response_keys,
    encrypt_ehbp_request_body,
)


def _key_config(public_key: bytes) -> bytes:
    return (
        b"\x00"
        + KEM_X25519_HKDF_SHA256.to_bytes(2, "big")
        + public_key
        + (4).to_bytes(2, "big")
        + KDF_HKDF_SHA256.to_bytes(2, "big")
        + AEAD_AES_256_GCM.to_bytes(2, "big")
    )


def _raw_public_key(private_key: x25519.X25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def test_ehbp_request_encryption_round_trips_with_fixed_keys() -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    ephemeral_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32, 64)))
    plaintext = b'{"model":"private/gpt-oss-120b","messages":[{"content":"secret"}]}'

    encrypted = encrypt_ehbp_request_body(
        _key_config(_raw_public_key(recipient_private)),
        plaintext,
        ephemeral_private_key=ephemeral_private,
    )

    assert encrypted.encapsulated_key_hex == _raw_public_key(ephemeral_private).hex()
    assert encrypted.body != plaintext
    assert plaintext not in encrypted.body
    assert encrypted.body[:4] == (len(encrypted.body) - 4).to_bytes(4, "big")
    assert encrypted.exported_secret_hex

    decrypted = decrypt_ehbp_request_body_for_test(
        recipient_private,
        encrypted.encapsulated_key,
        encrypted.body,
    )
    assert decrypted == plaintext


def test_derive_ehbp_response_keys_matches_reference_vector() -> None:
    exported_secret = bytes.fromhex(
        "000102030405060708090a0b0c0d0e0f"
        "101112131415161718191a1b1c1d1e1f"
    )
    request_enc = bytes.fromhex(
        "202122232425262728292a2b2c2d2e2f"
        "303132333435363738393a3b3c3d3e3f"
    )
    response_nonce = bytes.fromhex(
        "404142434445464748494a4b4c4d4e4f"
        "505152535455565758595a5b5c5d5e5f"
    )

    keys = derive_ehbp_response_keys(exported_secret, request_enc, response_nonce)

    assert keys.key == bytes.fromhex(
        "40ec528847cd4e928449f2ed1a70a7d1"
        "e8ee317d5e900424fc1dd5b0475b97f7"
    )
    assert keys.nonce_base == bytes.fromhex("f8b0ce9466f27aa6243c65f9")


def test_decrypt_ehbp_response_body_matches_reference_vector() -> None:
    exported_secret = bytes.fromhex(
        "000102030405060708090a0b0c0d0e0f"
        "101112131415161718191a1b1c1d1e1f"
    )
    request_enc = bytes.fromhex(
        "202122232425262728292a2b2c2d2e2f"
        "303132333435363738393a3b3c3d3e3f"
    )
    response_nonce = bytes.fromhex(
        "404142434445464748494a4b4c4d4e4f"
        "505152535455565758595a5b5c5d5e5f"
    )
    encrypted_response = bytes.fromhex(
        "0000002647e74d9a561b60ff42ac7ffbb4caf6b6d5ad0bb7621e41840d2ab7de"
        "7208ff7a59d0a2a482cc"
    )

    plaintext = decrypt_ehbp_response_body(
        exported_secret,
        request_enc,
        response_nonce,
        encrypted_response,
    )

    assert plaintext == b"hello from test vector"


def test_ehbp_request_encryption_rejects_empty_body() -> None:
    recipient_private = x25519.X25519PrivateKey.generate()

    try:
        encrypt_ehbp_request_body(_key_config(_raw_public_key(recipient_private)), b"")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected EHBP encryption to reject empty body")
