from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

KEM_X25519_HKDF_SHA256 = 0x0020
KDF_HKDF_SHA256 = 0x0001
AEAD_AES_256_GCM = 0x0002
HPKE_REQUEST_INFO = b"ehbp request"
HPKE_RESPONSE_EXPORT_LABEL = b"ehbp response"
HPKE_EXPORT_LENGTH = 32
EHBP_RESPONSE_NONCE_LENGTH = 32
AES256_KEY_LENGTH = 32
AES_GCM_NONCE_LENGTH = 12
X25519_KEY_LENGTH = 32
_HASH_LENGTH = 32

_PUBLIC_KEY_SIZES = {
    0x0010: 65,   # DHKEM(P-256, HKDF-SHA256)
    0x0011: 97,   # DHKEM(P-384, HKDF-SHA384)
    0x0012: 133,  # DHKEM(P-521, HKDF-SHA512)
    KEM_X25519_HKDF_SHA256: 32,
    0x0021: 56,   # DHKEM(X448, HKDF-SHA512)
}


@dataclass(frozen=True)
class EhbpKeyConfig:
    key_id: int
    kem_id: int
    public_key: bytes
    cipher_suites: tuple[tuple[int, int], ...]

    @property
    def public_key_digest(self) -> str:
        return f"sha256:{hashlib.sha256(self.public_key).hexdigest()}"

    @property
    def first_cipher_suite(self) -> tuple[int, int]:
        return self.cipher_suites[0]

    def verified_claims(self) -> dict[str, object]:
        return {
            "ehbp_key_id": self.key_id,
            "ehbp_kem_id": f"0x{self.kem_id:04x}",
            "ehbp_public_key_digest": self.public_key_digest,
            "ehbp_cipher_suites": [
                {"kdf_id": f"0x{kdf:04x}", "aead_id": f"0x{aead:04x}"}
                for kdf, aead in self.cipher_suites
            ],
        }


@dataclass(frozen=True)
class EhbpEncryptedRequest:
    encapsulated_key: bytes
    body: bytes
    exported_secret: bytes

    @property
    def encapsulated_key_hex(self) -> str:
        return self.encapsulated_key.hex()

    @property
    def exported_secret_hex(self) -> str:
        return self.exported_secret.hex()


@dataclass(frozen=True)
class EhbpResponseKeys:
    key: bytes
    nonce_base: bytes


@dataclass(frozen=True)
class EhbpDecryptedRequestForTest:
    plaintext: bytes
    exported_secret: bytes


def _read_u16(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ValueError("invalid EHBP HPKE key config: truncated uint16")
    return int.from_bytes(data[offset : offset + 2], "big"), offset + 2


def parse_ehbp_key_config(data: bytes) -> EhbpKeyConfig:
    """Parse the EHBP/OHTTP HPKE key config emitted by Tinfoil-compatible servers."""
    if len(data) < 1 + 2 + 2:
        raise ValueError("invalid EHBP HPKE key config: too short")

    offset = 0
    key_id = data[offset]
    offset += 1

    kem_id, offset = _read_u16(data, offset)
    public_key_size = _PUBLIC_KEY_SIZES.get(kem_id)
    if public_key_size is None:
        raise ValueError(f"invalid EHBP HPKE key config: unsupported KEM 0x{kem_id:04x}")
    if offset + public_key_size > len(data):
        raise ValueError("invalid EHBP HPKE key config: truncated public key")

    public_key = data[offset : offset + public_key_size]
    offset += public_key_size

    suites_len, offset = _read_u16(data, offset)
    if suites_len == 0:
        raise ValueError("invalid EHBP HPKE key config: no cipher suites")
    if suites_len % 4 != 0:
        raise ValueError("invalid EHBP HPKE key config: malformed cipher suite list")
    if offset + suites_len > len(data):
        raise ValueError("invalid EHBP HPKE key config: truncated cipher suite list")

    suites_end = offset + suites_len
    suites: list[tuple[int, int]] = []
    while offset < suites_end:
        kdf_id, offset = _read_u16(data, offset)
        aead_id, offset = _read_u16(data, offset)
        suites.append((kdf_id, aead_id))

    if offset != len(data):
        raise ValueError("invalid EHBP HPKE key config: trailing bytes")

    config = EhbpKeyConfig(
        key_id=key_id,
        kem_id=kem_id,
        public_key=public_key,
        cipher_suites=tuple(suites),
    )
    if config.key_id != 0:
        raise ValueError(
            f"invalid EHBP HPKE key config: unsupported key id {config.key_id}"
        )
    if config.kem_id != KEM_X25519_HKDF_SHA256:
        raise ValueError(
            f"invalid EHBP HPKE key config: unsupported KEM 0x{config.kem_id:04x}"
        )
    if config.first_cipher_suite != (KDF_HKDF_SHA256, AEAD_AES_256_GCM):
        kdf_id, aead_id = config.first_cipher_suite
        raise ValueError(
            "invalid EHBP HPKE key config: unsupported EHBP cipher suite "
            f"KDF=0x{kdf_id:04x} AEAD=0x{aead_id:04x}"
        )
    return config


def _i2osp(value: int, length: int) -> bytes:
    return value.to_bytes(length, "big")


def _xor_nonce(base_nonce: bytes, seq: int) -> bytes:
    if len(base_nonce) != AES_GCM_NONCE_LENGTH:
        raise ValueError("nonce base must be 12 bytes")
    if seq < 0 or seq >= 2**64:
        raise ValueError("sequence number out of range")
    seq_bytes = seq.to_bytes(8, "big")
    prefix = base_nonce[:4]
    suffix = bytes(a ^ b for a, b in zip(base_nonce[4:], seq_bytes))
    return prefix + suffix


def _hkdf_extract(salt: bytes | None, ikm: bytes) -> bytes:
    if salt is None or salt == b"":
        salt = b"\x00" * _HASH_LENGTH
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    return HKDFExpand(
        algorithm=hashes.SHA256(),
        length=length,
        info=info,
    ).derive(prk)


def _labeled_extract(
    *,
    suite_id: bytes,
    salt: bytes | None,
    label: bytes,
    ikm: bytes,
) -> bytes:
    labeled_ikm = b"HPKE-v1" + suite_id + label + ikm
    return _hkdf_extract(salt, labeled_ikm)


def _labeled_expand(
    *,
    suite_id: bytes,
    prk: bytes,
    label: bytes,
    info: bytes,
    length: int,
) -> bytes:
    labeled_info = _i2osp(length, 2) + b"HPKE-v1" + suite_id + label + info
    return _hkdf_expand(prk, labeled_info, length)


def _kem_suite_id() -> bytes:
    return b"KEM" + _i2osp(KEM_X25519_HKDF_SHA256, 2)


def _hpke_suite_id() -> bytes:
    return (
        b"HPKE"
        + _i2osp(KEM_X25519_HKDF_SHA256, 2)
        + _i2osp(KDF_HKDF_SHA256, 2)
        + _i2osp(AEAD_AES_256_GCM, 2)
    )


def _raw_public_key(public_key: x25519.X25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _derive_public_key(private_key: x25519.X25519PrivateKey) -> bytes:
    return _raw_public_key(private_key.public_key())


def _dhkem_extract_and_expand(
    *,
    dh: bytes,
    enc: bytes,
    recipient_public_key: bytes,
) -> bytes:
    suite_id = _kem_suite_id()
    eae_prk = _labeled_extract(
        suite_id=suite_id,
        salt=None,
        label=b"eae_prk",
        ikm=dh,
    )
    kem_context = enc + recipient_public_key
    return _labeled_expand(
        suite_id=suite_id,
        prk=eae_prk,
        label=b"shared_secret",
        info=kem_context,
        length=32,
    )


def _setup_base_sender_context(
    *,
    recipient_public_key: bytes,
    ephemeral_private_key: x25519.X25519PrivateKey,
    info: bytes = HPKE_REQUEST_INFO,
) -> tuple[bytes, bytes, bytes, bytes]:
    recipient_public = x25519.X25519PublicKey.from_public_bytes(recipient_public_key)
    enc = _derive_public_key(ephemeral_private_key)
    dh = ephemeral_private_key.exchange(recipient_public)
    shared_secret = _dhkem_extract_and_expand(
        dh=dh,
        enc=enc,
        recipient_public_key=recipient_public_key,
    )
    key, base_nonce, exporter_secret = _key_schedule_base(shared_secret, info)
    return enc, key, base_nonce, exporter_secret


def _setup_base_recipient_context_for_test(
    *,
    recipient_private_key: x25519.X25519PrivateKey,
    enc: bytes,
    info: bytes = HPKE_REQUEST_INFO,
) -> tuple[bytes, bytes, bytes]:
    if len(enc) != X25519_KEY_LENGTH:
        raise ValueError("encapsulated key must be 32 bytes")
    ephemeral_public = x25519.X25519PublicKey.from_public_bytes(enc)
    recipient_public_key = _derive_public_key(recipient_private_key)
    dh = recipient_private_key.exchange(ephemeral_public)
    shared_secret = _dhkem_extract_and_expand(
        dh=dh,
        enc=enc,
        recipient_public_key=recipient_public_key,
    )
    return _key_schedule_base(shared_secret, info)


def _key_schedule_base(shared_secret: bytes, info: bytes) -> tuple[bytes, bytes, bytes]:
    suite_id = _hpke_suite_id()
    psk_id_hash = _labeled_extract(
        suite_id=suite_id,
        salt=None,
        label=b"psk_id_hash",
        ikm=b"",
    )
    info_hash = _labeled_extract(
        suite_id=suite_id,
        salt=None,
        label=b"info_hash",
        ikm=info,
    )
    key_schedule_context = b"\x00" + psk_id_hash + info_hash
    secret = _labeled_extract(
        suite_id=suite_id,
        salt=shared_secret,
        label=b"secret",
        ikm=b"",
    )
    key = _labeled_expand(
        suite_id=suite_id,
        prk=secret,
        label=b"key",
        info=key_schedule_context,
        length=AES256_KEY_LENGTH,
    )
    base_nonce = _labeled_expand(
        suite_id=suite_id,
        prk=secret,
        label=b"base_nonce",
        info=key_schedule_context,
        length=AES_GCM_NONCE_LENGTH,
    )
    exporter_secret = _labeled_expand(
        suite_id=suite_id,
        prk=secret,
        label=b"exp",
        info=key_schedule_context,
        length=_HASH_LENGTH,
    )
    return key, base_nonce, exporter_secret


def _hpke_export(exporter_secret: bytes, label: bytes, length: int) -> bytes:
    return _labeled_expand(
        suite_id=_hpke_suite_id(),
        prk=exporter_secret,
        label=b"sec",
        info=label,
        length=length,
    )


def _frame_ciphertexts(ciphertexts: list[bytes]) -> bytes:
    framed = bytearray()
    for ciphertext in ciphertexts:
        framed.extend(len(ciphertext).to_bytes(4, "big"))
        framed.extend(ciphertext)
    return bytes(framed)


def _open_framed_body(aead: AESGCM, nonce_base: bytes, body: bytes) -> bytes:
    offset = 0
    seq = 0
    plaintext = bytearray()
    while offset < len(body):
        if offset + 4 > len(body):
            raise ValueError("invalid EHBP body: truncated chunk length")
        chunk_len = int.from_bytes(body[offset : offset + 4], "big")
        offset += 4
        if chunk_len == 0:
            continue
        if offset + chunk_len > len(body):
            raise ValueError("invalid EHBP body: truncated ciphertext chunk")
        ciphertext = body[offset : offset + chunk_len]
        offset += chunk_len
        plaintext.extend(aead.decrypt(_xor_nonce(nonce_base, seq), ciphertext, None))
        seq += 1
    return bytes(plaintext)


def encrypt_ehbp_request_body(
    key_config_bytes: bytes,
    plaintext: bytes,
    *,
    ephemeral_private_key: x25519.X25519PrivateKey | None = None,
) -> EhbpEncryptedRequest:
    """Encrypt a non-empty request body using EHBP's HPKE request framing."""
    if not plaintext:
        raise ValueError("EHBP request encryption requires a non-empty body")
    key_config = parse_ehbp_key_config(key_config_bytes)
    if ephemeral_private_key is None:
        ephemeral_private_key = x25519.X25519PrivateKey.from_private_bytes(
            os.urandom(X25519_KEY_LENGTH)
        )
    enc, key, base_nonce, exporter_secret = _setup_base_sender_context(
        recipient_public_key=key_config.public_key,
        ephemeral_private_key=ephemeral_private_key,
    )
    aead = AESGCM(key)
    ciphertext = aead.encrypt(_xor_nonce(base_nonce, 0), plaintext, None)
    exported_secret = _hpke_export(
        exporter_secret,
        HPKE_RESPONSE_EXPORT_LABEL,
        HPKE_EXPORT_LENGTH,
    )
    return EhbpEncryptedRequest(
        encapsulated_key=enc,
        body=_frame_ciphertexts([ciphertext]),
        exported_secret=exported_secret,
    )


def decrypt_ehbp_request_body_for_test(
    recipient_private_key: x25519.X25519PrivateKey,
    encapsulated_key: bytes,
    body: bytes,
) -> bytes:
    """Decrypt an EHBP request body for tests and interoperability fixtures."""
    return decrypt_ehbp_request_for_test(
        recipient_private_key,
        encapsulated_key,
        body,
    ).plaintext


def decrypt_ehbp_request_for_test(
    recipient_private_key: x25519.X25519PrivateKey,
    encapsulated_key: bytes,
    body: bytes,
) -> EhbpDecryptedRequestForTest:
    """Decrypt an EHBP request body and return server-side context for tests."""
    key, base_nonce, _exporter_secret = _setup_base_recipient_context_for_test(
        recipient_private_key=recipient_private_key,
        enc=encapsulated_key,
    )
    plaintext = _open_framed_body(AESGCM(key), base_nonce, body)
    exported_secret = _hpke_export(
        _exporter_secret,
        HPKE_RESPONSE_EXPORT_LABEL,
        HPKE_EXPORT_LENGTH,
    )
    return EhbpDecryptedRequestForTest(
        plaintext=plaintext,
        exported_secret=exported_secret,
    )


def derive_ehbp_response_keys(
    exported_secret: bytes,
    request_enc: bytes,
    response_nonce: bytes,
) -> EhbpResponseKeys:
    if len(exported_secret) != HPKE_EXPORT_LENGTH:
        raise ValueError("exported secret must be 32 bytes")
    if len(request_enc) != X25519_KEY_LENGTH:
        raise ValueError("request enc must be 32 bytes")
    if len(response_nonce) != EHBP_RESPONSE_NONCE_LENGTH:
        raise ValueError("response nonce must be 32 bytes")
    salt = request_enc + response_nonce
    prk = _hkdf_extract(salt, exported_secret)
    key = _hkdf_expand(prk, b"key", AES256_KEY_LENGTH)
    nonce_base = _hkdf_expand(prk, b"nonce", AES_GCM_NONCE_LENGTH)
    return EhbpResponseKeys(key=key, nonce_base=nonce_base)


def decrypt_ehbp_response_body(
    exported_secret: bytes,
    request_enc: bytes,
    response_nonce: bytes,
    body: bytes,
) -> bytes:
    keys = derive_ehbp_response_keys(exported_secret, request_enc, response_nonce)
    return _open_framed_body(AESGCM(keys.key), keys.nonce_base, body)


def encrypt_ehbp_response_body_for_test(
    exported_secret: bytes,
    request_enc: bytes,
    response_nonce: bytes,
    plaintext: bytes,
) -> bytes:
    keys = derive_ehbp_response_keys(exported_secret, request_enc, response_nonce)
    aead = AESGCM(keys.key)
    ciphertext = aead.encrypt(_xor_nonce(keys.nonce_base, 0), plaintext, None)
    return _frame_ciphertexts([ciphertext])
