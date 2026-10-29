import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import secrets
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import Crypto
    from Crypto.Cipher import ChaCha20
    from Crypto.Hash import SHA256
    from Crypto.Protocol.KDF import HKDF
except ImportError as error:
    raise ImportError("Install dependencies with: pip install numpy pycryptodome") from error


BASE_DIR = Path(__file__).resolve().parent
PROTOCOL_VERSION = "CPIS-LRE-ChaCha20-v1"
HKDF_SALT = b"CPIS-LRE-HKDF-SHA256-v1"


def decode_key(value, key_bits=128):
    if key_bits not in (128, 192, 256):
        raise ValueError("key_bits must be 128, 192, or 256.")
    if isinstance(value, str):
        if len(value) != key_bits // 4:
            raise ValueError("The key must contain exactly %d hexadecimal characters." % (key_bits // 4))
        try:
            value = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError("The key must be hexadecimal, not a password or decimal seed.") from error
    if not isinstance(value, bytes) or len(value) * 8 != key_bits:
        raise ValueError("The decoded key length does not match key_bits.")
    return value


def generate_subkeys(key_bits=128):
    if key_bits not in (128, 192, 256):
        raise ValueError("key_bits must be 128, 192, or 256.")
    return secrets.token_bytes(key_bits // 8), secrets.token_bytes(key_bits // 8)


def hkdf_sha256(key, length=32, salt=HKDF_SALT, info=b""):
    return HKDF(key, length, salt, SHA256, context=info)


def validate_shape(shape):
    shape = tuple(int(value) for value in shape)
    if len(shape) != 4 or any(value < 1 for value in shape):
        raise ValueError("shape must be (batch, channels, height, width), with positive dimensions.")
    return shape


def derive_stream_key(key, purpose, shape, key_bits=128):
    key = decode_key(key, key_bits)
    if purpose not in ("LEP", "LEN"):
        raise ValueError("purpose must be LEP or LEN.")
    shape = validate_shape(shape)
    info = json.dumps([PROTOCOL_VERSION, purpose, list(shape)], separators=(",", ":")).encode("ascii")
    return hkdf_sha256(key, 32, info=info)


def uniform_below(stream, upper):
    if not 1 <= upper <= 2 ** 32:
        raise ValueError("upper must be in [1, 2**32].")
    if upper == 1:
        return 0
    limit = (2 ** 32 // upper) * upper
    while True:
        value = struct.unpack("<I", stream.read(4))[0]
        if value < limit:
            return value % upper


class ChaCha20Stream:
    def __init__(self, key, nonce, initial_counter=1):
        if len(key) != 32:
            raise ValueError("ChaCha20 requires a 32-byte key.")
        if len(nonce) != 12:
            raise ValueError("RFC 8439 requires a 12-byte nonce.")
        if not 0 <= initial_counter < 2 ** 32:
            raise ValueError("initial_counter must fit in 32 bits.")
        self.cipher = ChaCha20.new(key=key, nonce=nonce)
        self.cipher.seek(initial_counter * 64)
        self.remaining = (2 ** 32 - initial_counter) * 64
        self.generated = 0
        self.buffer = b""
        self.offset = 0

    def read(self, size):
        if size < 0 or size > self.remaining:
            raise ValueError("Invalid size or exhausted RFC 8439 counter space.")
        available = len(self.buffer) - self.offset
        if available < size:
            prefix = self.buffer[self.offset:]
            amount = min(max(4096, size - available), self.remaining - available)
            self.buffer = prefix + self.cipher.encrypt(bytes(amount))
            self.generated += amount
            self.offset = 0
        result = self.buffer[self.offset:self.offset + size]
        self.offset += size
        self.remaining -= size
        return result

    def randbelow(self, upper):
        return uniform_below(self, upper)


def make_stream(key, purpose, shape, nonce, key_bits=128):
    return ChaCha20Stream(derive_stream_key(key, purpose, shape, key_bits), nonce, initial_counter=1)


def fisher_yates(channels, draw):
    values = list(range(channels))
    for index in range(channels - 1, 0, -1):
        selected = draw(index + 1)
        values[index], values[selected] = values[selected], values[index]
    return values


def channel_permutation(shape, key, nonce, key_bits=128):
    batch, channels, height, width = validate_shape(shape)
    stream = make_stream(key, "LEP", shape, nonce, key_bits)
    values = np.empty((batch, height, width, channels), dtype=np.int64)
    flat = values.reshape(-1, channels)
    for index in range(flat.shape[0]):
        flat[index] = fisher_yates(channels, stream.randbelow)
    indices = values.transpose(0, 3, 1, 2).copy()
    return indices, np.argsort(indices, axis=1)


def broadcast_mask(mask, shape):
    shape = validate_shape(shape)
    mask = np.asarray(mask)
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("The mask must be binary.")
    try:
        return np.broadcast_to(mask.astype(bool), shape)
    except ValueError as error:
        raise ValueError("The mask cannot be broadcast to the latent shape.") from error


def apply_permutation(latent, indices, mask):
    if latent.shape != indices.shape:
        raise ValueError("Latent and permutation shapes must match.")
    selected = broadcast_mask(mask, latent.shape)
    transformed = np.take_along_axis(latent, indices, axis=1)
    return np.where(selected, transformed, latent)


def gaussian_perturbation(shape, key, nonce, scale=1.0, key_bits=128, dtype=np.float32):
    shape = validate_shape(shape)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative.")
    count = math.prod(shape)
    pairs = (count + 1) // 2
    stream = make_stream(key, "LEN", shape, nonce, key_bits)
    words = np.frombuffer(stream.read(pairs * 16), dtype="<u8").reshape(pairs, 2)
    uniforms = ((words >> np.uint64(12)).astype(np.float64) + 0.5) * (2.0 ** -52)
    radius = np.sqrt(-2.0 * np.log(uniforms[:, 0]))
    angle = 2.0 * np.pi * uniforms[:, 1]
    values = np.empty(pairs * 2, dtype=np.float64)
    values[0::2] = radius * np.cos(angle)
    values[1::2] = radius * np.sin(angle)
    return (scale * values[:count]).reshape(shape).astype(dtype)


def flip_bit(key, bit_index):
    if not 0 <= bit_index < 8 * len(key):
        raise ValueError("bit_index is outside the key.")
    changed = bytearray(key)
    changed[bit_index // 8] ^= 1 << (bit_index % 8)
    return bytes(changed)


def bit_difference(left, right):
    if len(left) != len(right):
        raise ValueError("Bit comparison requires equal lengths.")
    return sum(bin(a ^ b).count("1") for a, b in zip(left, right)) / (8.0 * len(left))


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def fingerprints(args):
    shape = (1, 4, 64, 64)
    lep = decode_key(args.lep_key_hex, args.key_bits)
    length = decode_key(args.len_key_hex, args.key_bits)
    nonce = bytes.fromhex(args.nonce_hex)
    indices, inverse = channel_permutation(shape, lep, nonce, args.key_bits)
    delta = gaussian_perturbation(shape, length, nonce, args.len_strength, args.key_bits)
    return {"lep_sha256": array_hash(indices.astype("<i8")),
            "inverse_lep_sha256": array_hash(inverse.astype("<i8")),
            "len_sha256": array_hash(delta.astype("<f4"))}


def coefficients(alpha_t, alpha_previous):
    a = math.sqrt(alpha_previous / alpha_t)
    b = math.sqrt(max(0.0, 1.0 - alpha_previous)) - a * math.sqrt(max(0.0, 1.0 - alpha_t))
    return a, b


class SyntheticDDIM:
    def __init__(self, steps=50, lep_step=1, len_step=25):
        if not 1 <= lep_step <= steps or not 1 <= len_step <= steps:
            raise ValueError("LRE steps must be one-based positions in the denoising loop.")
        self.steps = steps
        self.lep_step = lep_step
        self.len_step = len_step
        self.timesteps = (np.arange(steps) * (1000 // steps))[::-1] + 1
        betas = np.linspace(math.sqrt(0.00085), math.sqrt(0.012), 1000) ** 2
        cumulative = np.cumprod(1.0 - betas)
        self.alphas = list(cumulative[self.timesteps]) + [1.0]
        self.events = []
        self.residuals = []

    def epsilon(self, latent, index, branch):
        bias = 0.02 if branch == "private" else -0.03
        return 0.08 * np.tanh(latent) + 0.015 * np.roll(latent, 1, axis=1) + bias + 0.01 * math.sin(index)

    def step(self, latent, index, branch, mask, reference, delta=None):
        a, b = coefficients(self.alphas[index], self.alphas[index + 1])
        epsilon = self.epsilon(latent, index, branch)
        if delta is not None and index + 1 == self.len_step:
            epsilon = epsilon + delta * mask
        result = a * latent + b * epsilon
        return np.where(mask, result, reference)

    def denoise(self, latent, branch, mask, delta=None, permutation=None):
        current = latent.copy()
        reference = latent.copy()
        for index in range(self.steps):
            if permutation is not None and index + 1 == self.lep_step:
                current = apply_permutation(current, permutation, mask)
                self.events.append(["sender_LEP", index + 1])
            if delta is not None and index + 1 == self.len_step:
                self.events.append(["sender_LEN", index + 1])
            current = self.step(current, index, branch, mask, reference, delta)
        return current

    def invert(self, latent, branch, mask, delta=None, inverse_permutation=None):
        current = latent.copy()
        reference = latent.copy()
        for inverse_position, index in enumerate(range(self.steps - 1, -1, -1), 1):
            target = current.copy()
            estimate = current.copy()
            a, _ = coefficients(self.alphas[index], self.alphas[index + 1])
            for _ in range(12):
                residual = self.step(estimate, index, branch, mask, reference, delta) - target
                error = float(np.sqrt(np.mean(residual[broadcast_mask(mask, residual.shape)] ** 2)))
                if error <= 1e-10:
                    break
                estimate = np.where(mask, estimate - residual / a, reference)
            final = self.step(estimate, index, branch, mask, reference, delta) - target
            self.residuals.append(float(np.sqrt(np.mean(final[broadcast_mask(mask, final.shape)] ** 2))))
            current = estimate
            if delta is not None and index + 1 == self.len_step:
                self.events.append(["receiver_LEN", inverse_position, index + 1])
            if inverse_permutation is not None and index + 1 == self.lep_step:
                current = apply_permutation(current, inverse_permutation, mask)
                self.events.append(["receiver_LEP", inverse_position, index + 1])
        return current


class Verification:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, details=None):
        row = {"name": name, "passed": bool(passed), "details": details}
        self.checks.append(row)
        print("[%s] %s" % ("PASS" if passed else "FAIL", name), flush=True)


def standard_vectors(test):
    key = bytes.fromhex("0b" * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    expected = bytes.fromhex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")
    test.check("RFC 5869 SHA-256 test case 1", hkdf_sha256(key, 42, salt, info) == expected)
    expected = bytes.fromhex("8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8")
    test.check("RFC 5869 SHA-256 test case 3", hkdf_sha256(key, 42, b"", b"") == expected)
    expected = bytes.fromhex("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
    key = bytes(range(32))
    nonce = bytes.fromhex("000000090000004a00000000")
    stream = ChaCha20Stream(key, nonce, 1)
    test.check("RFC 8439 section 2.3.2 ChaCha20 block", stream.read(64) == expected)
    whole = ChaCha20Stream(key, nonce, 1).read(12000)
    chunks = ChaCha20Stream(key, nonce, 1)
    partitioned = b"".join(chunks.read(size) for size in (1, 3, 64, 1000, 4096, 6836))
    test.check("ChaCha20 stream is independent of read chunking", partitioned == whole)
    class FixedWords:
        def __init__(self):
            self.values = iter((2 ** 32 - 1, 4))
            self.calls = 0
        def read(self, size):
            self.calls += 1
            return struct.pack("<I", next(self.values))
    source = FixedWords()
    selected = uniform_below(source, 3)
    test.check("Rejection sampling discards out-of-range words", selected == 1 and source.calls == 2)
    for channels in (2, 3, 4):
        outputs = []
        for choices in itertools.product(*(range(index + 1) for index in range(channels - 1, 0, -1))):
            sequence = iter(choices)
            outputs.append(tuple(fisher_yates(channels, lambda upper: next(sequence))))
        test.check("Fisher-Yates exhaustively covers %d! outcomes" % channels,
                   len(outputs) == len(set(outputs)) == math.factorial(channels))


def run_verification(args):
    test = Verification()
    started = time.perf_counter()
    print("Public test fixtures are used. No diffusion model or real image is loaded.", flush=True)
    standard_vectors(test)
    shape = (1, 4, 64, 64)
    lep = decode_key(args.lep_key_hex, args.key_bits)
    length = decode_key(args.len_key_hex, args.key_bits)
    nonce = bytes.fromhex(args.nonce_hex)
    test.check("Strict hexadecimal key decoding", len(lep) * 8 == args.key_bits and len(length) * 8 == args.key_bits)
    rejected = False
    try:
        decode_key("1234", args.key_bits)
    except ValueError:
        rejected = True
    test.check("Short password-like inputs are rejected", rejected)
    generated = generate_subkeys(args.key_bits)
    test.check("Two separate CSPRNG key-generation calls produce valid subkeys",
               all(len(value) * 8 == args.key_bits for value in generated) and generated[0] != generated[1])
    test.check("HKDF produces two 256-bit ChaCha20 keys",
               len(derive_stream_key(lep, "LEP", shape, args.key_bits)) == 32
               and len(derive_stream_key(length, "LEN", shape, args.key_bits)) == 32)
    test.check("LEP and LEN use distinct HKDF domains",
               derive_stream_key(lep, "LEP", shape, args.key_bits) != derive_stream_key(lep, "LEN", shape, args.key_bits))
    indices, inverse = channel_permutation(shape, lep, nonce, args.key_bits)
    identity = np.broadcast_to(np.arange(4)[None, :, None, None], shape)
    test.check("Every spatial position has a bijective channel permutation", np.array_equal(np.sort(indices, axis=1), identity))
    test.check("Inverse permutation indices are correct", np.array_equal(np.take_along_axis(indices, inverse, axis=1), identity))
    repeated = [channel_permutation(shape, lep, nonce, args.key_bits)[0] for _ in range(args.repeats)]
    test.check("LEP reproduces exactly over %d runs" % args.repeats, all(np.array_equal(indices, value) for value in repeated))
    source = np.arange(math.prod(shape), dtype=np.float64).reshape(shape) / 97.0
    mask = np.zeros((1, 1, 64, 64), dtype=bool)
    mask[:, :, 5:59, 7:57] = True
    selected = broadcast_mask(mask, shape)
    encrypted = apply_permutation(source, indices, mask)
    recovered = apply_permutation(encrypted, inverse, mask)
    max_error = float(np.max(np.abs(recovered - source)))
    test.check("LEP correct-key round trip is exact", max_error == 0.0, {"max_absolute_error": max_error})
    test.check("LEP preserves all unmasked values", np.array_equal(encrypted[~selected], source[~selected]))
    delta = gaussian_perturbation(shape, length, nonce, args.len_strength, args.key_bits)
    repeated_delta = [gaussian_perturbation(shape, length, nonce, args.len_strength, args.key_bits) for _ in range(args.repeats)]
    test.check("LEN reproduces exactly over %d runs" % args.repeats, all(np.array_equal(delta, value) for value in repeated_delta))
    test.check("Box-Muller outputs are finite", bool(np.isfinite(delta).all()))
    test.check("LEN injection preserves all unmasked values", np.array_equal((source + delta * mask)[~selected], source[~selected]))
    half = gaussian_perturbation(shape, length, nonce, 0.5, args.key_bits, np.float64)
    full = gaussian_perturbation(shape, length, nonce, 1.0, args.key_bits, np.float64)
    test.check("LEN scale changes only perturbation amplitude", np.array_equal(half * 2.0, full))
    changed_nonce = bytearray(nonce)
    changed_nonce[0] ^= 1
    test.check("A changed nonce changes the LEN stream",
               not np.array_equal(delta, gaussian_perturbation(shape, length, bytes(changed_nonce), args.len_strength, args.key_bits)))
    statistical = gaussian_perturbation((1, 4, 512, 512), length, nonce, 1.0, args.key_bits, np.float64).ravel()
    statistics = {"n": statistical.size, "mean": float(np.mean(statistical)),
                  "variance": float(np.var(statistical)), "std": float(np.std(statistical)),
                  "skewness": float(np.mean(((statistical - statistical.mean()) / statistical.std()) ** 3)),
                  "excess_kurtosis": float(np.mean(((statistical - statistical.mean()) / statistical.std()) ** 4) - 3.0),
                  "fraction_abs_above_3": float(np.mean(np.abs(statistical) > 3.0)),
                  "adjacent_correlation": float(np.corrcoef(statistical[:-1], statistical[1:])[0, 1])}
    test.check("Gaussian sample moments are close to N(0,1)", abs(statistics["mean"]) < 0.01 and abs(statistics["variance"] - 1.0) < 0.02, statistics)
    rows = []
    trial_stream = ChaCha20Stream(hashlib.sha256(b"CPIS-public-verification-fixtures-v1").digest(), nonce)
    for index in range(args.wrong_trials):
        wrong_lep = trial_stream.read(args.key_bits // 8)
        wrong_len = trial_stream.read(args.key_bits // 8)
        if wrong_lep == lep or wrong_len == length:
            raise RuntimeError("A test key unexpectedly equals a correct key.")
        _, wrong_inverse = channel_permutation(shape, wrong_lep, nonce, args.key_bits)
        bad = apply_permutation(encrypted, wrong_inverse, mask)
        wrong_delta = gaussian_perturbation(shape, wrong_len, nonce, args.len_strength, args.key_bits)
        rows.append({"trial": index + 1, "condition": "pseudorandom_wrong_keys",
                     "lep_roundtrip_exact": bool(np.array_equal(bad, source)),
                     "lep_changed_position_fraction": float(np.mean(np.any(bad != source, axis=1)[mask[:, 0]])),
                     "len_exact": bool(np.array_equal(wrong_delta, delta)),
                     "len_correlation": float(np.corrcoef(delta.ravel(), wrong_delta.ravel())[0, 1])})
    test.check("Tested incorrect LEP keys do not recover the complete tensor", all(not row["lep_roundtrip_exact"] for row in rows))
    test.check("Tested incorrect LEN keys do not reproduce the complete perturbation", all(not row["len_exact"] for row in rows))
    bit_index = 17
    bit_lep = flip_bit(lep, bit_index)
    bit_len = flip_bit(length, bit_index)
    test.check("Single-bit test changes exactly one input bit", bit_difference(lep, bit_lep) == 1.0 / args.key_bits and bit_difference(length, bit_len) == 1.0 / args.key_bits)
    _, bit_inverse = channel_permutation(shape, bit_lep, nonce, args.key_bits)
    bit_recovered = apply_permutation(encrypted, bit_inverse, mask)
    bit_delta = gaussian_perturbation(shape, bit_len, nonce, args.len_strength, args.key_bits)
    test.check("Single-bit LEP change alters recovery", not np.array_equal(bit_recovered, source))
    test.check("Single-bit LEN change alters the perturbation", not np.array_equal(bit_delta, delta))
    single_bit = {"flipped_bit_index": bit_index,
                  "lep_changed_position_fraction": float(np.mean(np.any(bit_recovered != source, axis=1)[mask[:, 0]])),
                  "len_correlation": float(np.corrcoef(delta.ravel(), bit_delta.ravel())[0, 1]),
                  "derived_lep_key_bit_difference_fraction": bit_difference(derive_stream_key(lep, "LEP", shape, args.key_bits), derive_stream_key(bit_lep, "LEP", shape, args.key_bits))}
    for bits in (128, 192, 256):
        short_shape = (1, 4, 2, 2)
        generated_key = bytes(range(bits // 8))
        p, inv = channel_permutation(short_shape, generated_key, nonce, bits)
        toy = np.arange(16).reshape(short_shape)
        reconstructed = apply_permutation(apply_permutation(toy, p, True), inv, True)
        test.check("Configured %d-bit key supports exact LEP recovery" % bits, np.array_equal(reconstructed, toy))
    parent_fingerprints = fingerprints(args)
    child_results = []
    for _ in range(args.process_repeats):
        command = [sys.executable, str(Path(__file__).resolve()), "--fingerprint_only", "--lep_key_hex", args.lep_key_hex,
                   "--len_key_hex", args.len_key_hex, "--nonce_hex", args.nonce_hex, "--key_bits", str(args.key_bits),
                   "--len_strength", str(args.len_strength)]
        child_results.append(json.loads(subprocess.check_output(command, text=True)))
    test.check("Fresh Python processes reproduce LEP and LEN exactly", all(result == parent_fingerprints for result in child_results), {"independent_processes": len(child_results)})
    small_shape = (1, 4, 16, 16)
    small_mask = np.ones((1, 1, 16, 16), dtype=bool)
    small_mask[:, :, :2, :] = False
    small_indices, small_inverse = channel_permutation(small_shape, lep, nonce, args.key_bits)
    small_delta = gaussian_perturbation(small_shape, length, nonce, 1.0, args.key_bits, np.float64)
    original = np.sin(np.arange(math.prod(small_shape)).reshape(small_shape) * 0.07)
    solver = SyntheticDDIM(args.num_steps, args.lep_step, args.len_step)
    terminal = solver.invert(original, "private", small_mask)
    stego = solver.denoise(terminal, "public", small_mask, small_delta, small_indices)
    estimated_terminal = solver.invert(stego, "public", small_mask, small_delta, small_inverse)
    restored = solver.denoise(estimated_terminal, "private", small_mask)
    small_selected = broadcast_mask(small_mask, small_shape)
    numerical_rmse = float(np.sqrt(np.mean((restored[small_selected] - original[small_selected]) ** 2)))
    test.check("Synthetic 50-step DDIM round trip uses the correct keys", numerical_rmse < 1e-6, {"masked_latent_rmse": numerical_rmse})
    test.check("Synthetic DDIM chain preserves unmasked values exactly", np.array_equal(restored[~small_selected], original[~small_selected]))
    expected_events = [["sender_LEP", args.lep_step], ["sender_LEN", args.len_step],
                       ["receiver_LEN", args.num_steps - args.len_step + 1, args.len_step],
                       ["receiver_LEP", args.num_steps - args.lep_step + 1, args.lep_step]]
    test.check("LEP/LEN placement and reverse ordering are correct", solver.events == expected_events, {"actual_events": solver.events})
    wrong_numerical = []
    for condition, use_lep, use_len in (("wrong_LEP", bit_lep, length), ("wrong_LEN", lep, bit_len), ("wrong_both", bit_lep, bit_len)):
        _, use_inverse = channel_permutation(small_shape, use_lep, nonce, args.key_bits)
        use_delta = gaussian_perturbation(small_shape, use_len, nonce, 1.0, args.key_bits, np.float64)
        receiver = SyntheticDDIM(args.num_steps, args.lep_step, args.len_step)
        estimated = receiver.invert(stego, "public", small_mask, use_delta, use_inverse)
        wrong_restored = receiver.denoise(estimated, "private", small_mask)
        error = float(np.sqrt(np.mean((wrong_restored[small_selected] - original[small_selected]) ** 2)))
        wrong_numerical.append({"condition": condition, "masked_latent_rmse": error})
    test.check("Tested mismatched keys change synthetic DDIM recovery", all(value["masked_latent_rmse"] > 100 * numerical_rmse for value in wrong_numerical), wrong_numerical)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL_VERSION, "scope": "Cryptographic generator and synthetic latent-chain functional verification; no real-image diffusion or PSNR measurement.",
              "public_fixture_keys": True, "key_bits_per_subkey": args.key_bits, "derived_chacha20_key_bits": 256,
              "nominal_joint_key_space": "2^%d" % (2 * args.key_bits),
              "hkdf_salt_ascii": HKDF_SALT.decode("ascii"), "hkdf_info": "Compact ASCII JSON: [protocol_version, purpose, [B,C,H,W]]",
              "nonce_hex": args.nonce_hex, "initial_counter": 1,
              "uniform_conversion": "Open interval (0,1): ((uint64_le >> 12) + 0.5) * 2^-52",
              "fisher_yates_sampling": "Unbiased bounded integers by rejection sampling of little-endian uint32 words",
              "traversal_order": "Batch, row, column; descending Fisher-Yates channel index",
              "numpy_version": np.__version__, "pycryptodome_version": Crypto.__version__, "python_version": sys.version.split()[0],
              "checks": test.checks, "passed": sum(row["passed"] for row in test.checks), "total": len(test.checks),
              "correct_lep_max_absolute_error": max_error, "fingerprints": parent_fingerprints,
              "gaussian_statistics": statistics, "single_bit": single_bit,
              "wrong_key_trial_count": len(rows), "wrong_key_fixture_policy": "Reproducible pseudorandom public test keys; not a paper dataset experiment.",
              "wrong_lep_complete_recovery_count": sum(row["lep_roundtrip_exact"] for row in rows),
              "wrong_len_exact_perturbation_count": sum(row["len_exact"] for row in rows),
              "mean_wrong_lep_changed_position_fraction": float(np.mean([row["lep_changed_position_fraction"] for row in rows])),
              "mean_abs_wrong_len_correlation": float(np.mean([abs(row["len_correlation"]) for row in rows])),
              "synthetic_ddim": {"steps": args.num_steps, "lep_step": args.lep_step, "len_step": args.len_step,
                                 "timesteps_descending": solver.timesteps.tolist(), "events": solver.events,
                                 "correct_key_masked_latent_rmse": numerical_rmse, "wrong_key_cases": wrong_numerical,
                                 "worst_correct_inverse_residual_rms": max(solver.residuals)},
              "elapsed_seconds": time.perf_counter() - started,
              "limits": ["HKDF output length does not increase the entropy of a 128-bit input key.",
                         "Nominal joint key count is not a proof of 256-bit security for the diffusion pipeline.",
                         "The synthetic predictor is an explicitly defined numerical fixture, not Stable Diffusion.",
                         "The test nonce is public and fixed for reproducible fixtures. Use fresh image-pair subkeys or a distinct nonce for another image pair.",
                         "Receiver stream regeneration intentionally reuses the sender key and nonce for the same image pair."]}
    with (output / "verification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with (output / "wrong_key_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Passed: %d/%d" % (report["passed"], report["total"]))
    print("Correct LEP maximum absolute error: %.12g" % max_error)
    print("Gaussian mean: %.8f; variance: %.8f; std: %.8f" % (statistics["mean"], statistics["variance"], statistics["std"]))
    print("Incorrect LEP complete tensor recoveries: %d/%d" % (report["wrong_lep_complete_recovery_count"], len(rows)))
    print("Incorrect LEN exact perturbations: %d/%d" % (report["wrong_len_exact_perturbation_count"], len(rows)))
    print("Synthetic DDIM correct-key masked latent RMSE: %.12g" % numerical_rmse)
    print("Report directory: " + str(output.resolve()))
    return 0 if report["passed"] == report["total"] else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lep_key_hex", default="00112233445566778899aabbccddeeff")
    parser.add_argument("--len_key_hex", default="ffeeddccbbaa99887766554433221100")
    parser.add_argument("--key_bits", type=int, choices=(128, 192, 256), default=128)
    parser.add_argument("--nonce_hex", default="000102030405060708090a0b")
    parser.add_argument("--len_strength", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--lep_step", type=int, default=1)
    parser.add_argument("--len_step", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--process_repeats", type=int, default=2)
    parser.add_argument("--wrong_trials", type=int, default=1000)
    parser.add_argument("--output_dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--fingerprint_only", action="store_true")
    args = parser.parse_args()
    decode_key(args.lep_key_hex, args.key_bits)
    decode_key(args.len_key_hex, args.key_bits)
    try:
        nonce = bytes.fromhex(args.nonce_hex)
    except ValueError:
        parser.error("nonce_hex must contain hexadecimal characters.")
    if len(nonce) != 12:
        parser.error("nonce_hex must decode to exactly 12 bytes.")
    if args.repeats < 1 or args.process_repeats < 1 or args.wrong_trials < 1:
        parser.error("All trial and repeat counts must be positive.")
    if args.num_steps != 50 or not 1 <= args.lep_step < args.len_step <= args.num_steps:
        parser.error("This synthetic schedule uses 50 steps and requires 1 <= lep_step < len_step <= 50.")
    if not math.isfinite(args.len_strength) or args.len_strength <= 0:
        parser.error("len_strength must be finite and positive.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.fingerprint_only:
        print(json.dumps(fingerprints(arguments), sort_keys=True))
    else:
        raise SystemExit(run_verification(arguments))
