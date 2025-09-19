# 🎲⚡ 5FREΞ Lite RNG Engine (refined)
# Dual-chaos RNG with Von Neumann correction and optional SHA-256 conditioning
# Author: Steven Ray Britt (Mr. Khaotic) & The 5FREΞ Research Team
# License: Evaluation Only — see README & LICENSE

from __future__ import annotations
import hashlib, random, json
from typing import List, Optional

# --------------------------
# Core utilities
# --------------------------

def _unpack_bytes_to_bits(b: bytes) -> List[int]:
    out: List[int] = []
    for by in b:
        for k in range(7, -1, -1):
            out.append((by >> k) & 1)
    return out

def _pack_bits_to_bytes(bits: List[int]) -> bytes:
    full = (len(bits) // 8) * 8
    buf = bytearray()
    byte_val = 0
    for i, bit in enumerate(bits[:full], 1):
        byte_val = (byte_val << 1) | (1 if bit else 0)
        if i % 8 == 0:
            buf.append(byte_val & 0xFF)
            byte_val = 0
    return bytes(buf)

def _von_neumann(bits: List[int]) -> List[int]:
    """Von Neumann corrector"""
    out: List[int] = []
    i, n = 0, len(bits)
    while i < n - 1:
        b1, b2 = bits[i], bits[i + 1]
        if b1 != b2:
            out.append(b1)
        i += 2
    return out

def _sha256_expand(seed_bytes: bytes, target_bits: int) -> List[int]:
    """Simple expander: chain SHA-256 digests until reaching target_bits."""
    material = hashlib.sha256(seed_bytes).digest()
    out_bits: List[int] = []
    i = 0
    while len(out_bits) < target_bits:
        block = hashlib.sha256(material + bytes([i & 0xFF])).digest()
        out_bits.extend(_unpack_bytes_to_bits(block))
        i += 1
    return out_bits[:target_bits]

# --------------------------
# Public APIs
# --------------------------

def generate_target_bits(target_bits: int,
                         r1: float = 3.99,
                         r2: float = 3.90,
                         seed: Optional[float] = None,
                         use_hash: bool = False,
                         max_iters: int = 50,
                         chunk_raw: Optional[int] = None) -> List[int]:
    """
    Generate AT LEAST `target_bits` bits:
      - Without hashing: collect VN-corrected bits until enough, then truncate.
      - With hashing: collect a pool, then condition/expand via SHA-256.

    Params:
      * r1, r2 — chaotic map parameters
      * seed — optional reproducibility seed
      * use_hash — enable SHA-256 conditioner/expander
      * max_iters — safeguard for runaway loops
      * chunk_raw — raw sample chunk size (auto if None)
    """
    if target_bits <= 0:
        return []

    rnd = random.Random()
    if seed is not None:
        rnd.seed(seed)
    x1, x2 = rnd.random(), rnd.random()

    out_corrected: List[int] = []
    chunk_raw = chunk_raw or max(80_000, target_bits * 8)

    for _ in range(max_iters):
        raw: List[int] = []
        for _ in range(chunk_raw):
            x1 = r1 * x1 * (1 - x1)
            x2 = r2 * x2 * (1 - x2)
            raw.append((1 if x1 >= 0.5 else 0) ^ (1 if x2 >= 0.5 else 0))
        corr = _von_neumann(raw)
        out_corrected.extend(corr)

        if not use_hash and len(out_corrected) >= target_bits:
            return out_corrected[:target_bits]
        if use_hash and len(out_corrected) >= 8192:
            break

    if not use_hash:
        return out_corrected[:target_bits]

    seed_bytes = _pack_bits_to_bytes(out_corrected)
    return _sha256_expand(seed_bytes, target_bits)

# --------------------------
# Stats + CLI
# --------------------------

def _basic_stats(bits: List[int]) -> dict:
    n = len(bits)
    ones = sum(bits)
    zeros = n - ones
    ones_frac = ones / n if n else 0.0
    if n < 2:
        return {"n": n, "ones": ones, "zeros": zeros,
                "ones_frac": ones_frac, "runs": n,
                "longest_run": n, "flip_rate": 0.0}
    runs = 1
    longest = 1
    current = 1
    flips = 0
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            flips += 1
            runs += 1
            longest = max(longest, current)
            current = 1
        else:
            current += 1
    longest = max(longest, current)
    flip_rate = flips / (n - 1)
    return {"n": n, "ones": ones, "zeros": zeros,
            "ones_frac": ones_frac, "runs": runs,
            "longest_run": longest, "flip_rate": flip_rate}

def _save_bin(path: str, bits: List[int]) -> None:
    with open(path, "wb") as f:
        f.write(_pack_bits_to_bytes(bits))

def _save_hex(path: str, bits: List[int]) -> None:
    data = _pack_bits_to_bytes(bits)
    with open(path, "w") as f:
        f.write(data.hex())

if __name__ == "__main__":
    import argparse, time, matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description="5FREΞ Lite RNG Engine")
    p.add_argument("-n", "--target", type=int, default=100_000,
                   help="Output bits (after VN/conditioning)")
    p.add_argument("--nohash", action="store_true", help="Disable hash conditioning")
    p.add_argument("--seed", type=float, default=None, help="Optional reproducibility seed")
    p.add_argument("--r1", type=float, default=3.99, help="Logistic map r1")
    p.add_argument("--r2", type=float, default=3.90, help="Logistic map r2")
    p.add_argument("--chunk", type=int, default=None,
                   help="Raw chunk size (default auto)")
    p.add_argument("-o", "--output", type=str, default="lite_rng_output.bin",
                   help="Output .bin file")
    p.add_argument("--hex", action="store_true", help="Also save hex output file")
    p.add_argument("--stats", action="store_true", help="Write JSON stats")
    args = p.parse_args()

    use_hash = not args.nohash

    t0 = time.time()
    bits = generate_target_bits(
        target_bits=args.target,
        r1=args.r1,
        r2=args.r2,
        seed=args.seed,
        use_hash=use_hash,
        chunk_raw=args.chunk
    )
    elapsed = time.time() - t0

    _save_bin(args.output, bits)
    if args.hex:
        _save_hex(args.output + ".hex", bits)

    st = _basic_stats(bits)
    st["elapsed_sec"] = elapsed
    st["hash"] = use_hash

    print(f"[LiteRNG] Output bits: {st['n']} saved -> {args.output}")
    print(f"[LiteRNG] ones%={st['ones_frac']*100:.3f}%  "
          f"flip_rate={st['flip_rate']*100:.3f}%  "
          f"runs={st['runs']}  longest_run={st['longest_run']}  "
          f"hash={'ON' if use_hash else 'OFF'}  elapsed={elapsed:.3f}s")

    if args.stats:
        with open(args.output + ".json", "w") as f:
            json.dump(st, f, indent=2)
        print(f"[LiteRNG] Stats JSON -> {args.output}.json")

    # Plot
    cum, walk = 0, []
    for b in bits[:10_000]:
        cum += (1 if b else -1)
        walk.append(cum)
    plt.figure()
    plt.plot(walk)
    plt.title("Lite RNG - Cumulative Random Walk (first 10k bits)")
    plt.xlabel("Index"); plt.ylabel("Cumulative sum (+1/-1)")
    plt.tight_layout()
    plt.savefig("lite_rng_plot.png")
    print("[LiteRNG] Saved plot as lite_rng_plot.png")
