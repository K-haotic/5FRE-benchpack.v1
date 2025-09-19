# 🎲⚡ 5FREΞ Five-Map RNG Engine
# XOR of five chaotic logistic maps + Von Neumann + optional SHA-256 conditioning/expansion
# Author: Steven Ray Britt (Mr. Khaotic) & The 5FREΞ Research Team
# License: Evaluation Only — see README & LICENSE

from __future__ import annotations
import hashlib, random, json, argparse, time
from typing import List, Optional

# --------------------------
# Bit packing / unpacking
# --------------------------

def _pack_bits_to_bytes(bits: List[int]) -> bytes:
    full = (len(bits)//8)*8
    buf = bytearray()
    byte = 0
    for i, b in enumerate(bits[:full], 1):
        byte = (byte << 1) | (1 if b else 0)
        if i % 8 == 0:
            buf.append(byte & 0xFF)
            byte = 0
    return bytes(buf)

def _unpack_bytes_to_bits(b: bytes) -> List[int]:
    out: List[int] = []
    for by in b:
        for k in range(7, -1, -1):
            out.append((by >> k) & 1)
    return out

# --------------------------
# Extraction / Conditioning
# --------------------------

def _von_neumann(bits: List[int]) -> List[int]:
    """Von Neumann extractor: 01->0, 10->1, (00|11)->discard."""
    out: List[int] = []
    i, n = 0, len(bits)
    while i < n - 1:
        b1, b2 = bits[i], bits[i + 1]
        if b1 != b2:
            out.append(b1)
        i += 2
    return out

def _sha256_expand(seed_bytes: bytes, target_bits: int) -> List[int]:
    """Simple conditioner/expander chaining SHA-256 blocks until reaching target_bits."""
    material = hashlib.sha256(seed_bytes).digest()
    out_bits: List[int] = []
    i = 0
    while len(out_bits) < target_bits:
        block = hashlib.sha256(material + bytes([i & 0xFF])).digest()
        out_bits.extend(_unpack_bytes_to_bits(block))
        i += 1
    return out_bits[:target_bits]

# --------------------------
# Core generator (5 maps)
# --------------------------

def generate_target_bits(target_bits: int,
                         rs: Optional[List[float]] = None,
                         seed: Optional[float] = None,
                         use_hash: bool = False,
                         max_iters: int = 50,
                         chunk_raw: Optional[int] = None) -> List[int]:
    """
    Five logistic maps -> XOR combine per step -> Von Neumann -> (optional) SHA-256 expand to exact size.

    Args:
      target_bits: desired number of output bits (exact).
      rs: list of 5 logistic map parameters in (3.9, 4.0); default [3.99, 3.97, 3.95, 3.93, 3.91].
      seed: optional reproducibility seed.
      use_hash: if True, condition+expand corrected bits with SHA-256 to reach exact size.
      max_iters: safety bound on chunk loops.
      chunk_raw: raw sample chunk size (auto if None).

    Note:
      This implementation is for research/evaluation; it is NOT a certified DRBG.
    """
    if target_bits <= 0:
        return []

    rs = rs or [3.99, 3.97, 3.95, 3.93, 3.91]
    if len(rs) != 5:
        raise ValueError("rs must contain exactly 5 r-parameters")

    rnd = random.Random()
    if seed is not None:
        rnd.seed(seed)
    xs = [rnd.random() for _ in range(5)]  # initial states for the 5 maps

    out_corr: List[int] = []
    # VN discards ~50%; 5-map XOR yields strong raw entropy — pick a safe chunk
    chunk_raw = chunk_raw or max(100_000, target_bits * 6)

    for _ in range(max_iters):
        raw: List[int] = []
        for _ in range(chunk_raw):
            bit = 0
            for i in range(5):
                xs[i] = rs[i] * xs[i] * (1 - xs[i])
                bit ^= (1 if xs[i] >= 0.5 else 0)
            raw.append(bit)
        corr = _von_neumann(raw)
        out_corr.extend(corr)

        if not use_hash and len(out_corr) >= target_bits:
            return out_corr[:target_bits]
        if use_hash and len(out_corr) >= 8192:  # enough seed material for expander
            break

    if not use_hash:
        return out_corr[:target_bits]

    seed_bytes = _pack_bits_to_bytes(out_corr)
    return _sha256_expand(seed_bytes, target_bits)

# --------------------------
# Stats helpers
# --------------------------

def _basic_stats(bits: List[int]) -> dict:
    n = len(bits)
    ones = sum(bits)
    zeros = n - ones
    ones_frac = ones/n if n else 0.0
    if n < 2:
        return {"n": n, "ones": ones, "zeros": zeros,
                "ones_frac": ones_frac, "runs": n,
                "longest_run": n, "flip_rate": 0.0}
    runs = 1
    longest = 1
    current = 1
    flips = 0
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            flips += 1
            runs += 1
            if current > longest:
                longest = current
            current = 1
        else:
            current += 1
    if current > longest:
        longest = current
    flip_rate = flips/(n-1)
    return {"n": n, "ones": ones, "zeros": zeros,
            "ones_frac": ones_frac, "runs": runs,
            "longest_run": longest, "flip_rate": flip_rate}

# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="5FREΞ Five-Map RNG")
    p.add_argument("-n", "--target", type=int, default=100_000, help="Output bits after VN/conditioning")
    p.add_argument("--seed", type=float, default=None, help="Optional reproducibility seed")
    p.add_argument("--nohash", action="store_true", help="Disable SHA-256 conditioning/expansion")
    p.add_argument("--chunk", type=int, default=None, help="Raw chunk size (default auto)")
    p.add_argument("--r", type=float, nargs=5, metavar=("r1","r2","r3","r4","r5"),
                   default=[3.99, 3.97, 3.95, 3.93, 3.91], help="Five logistic map parameters")
    p.add_argument("-o", "--output", type=str, default="fivemap_rng_output.bin", help="Output .bin filename")
    p.add_argument("--stats", action="store_true", help="Write JSON stats alongside .bin")
    args = p.parse_args()

    use_hash = not args.nohash

    t0 = time.time()
    bits = generate_target_bits(target_bits=args.target,
                                rs=args.r,
                                seed=args.seed,
                                use_hash=use_hash,
                                chunk_raw=args.chunk)
    elapsed = time.time() - t0

    # Save binary
    with open(args.output, "wb") as f:
        f.write(_pack_bits_to_bytes(bits))

    # Stats
    st = _basic_stats(bits)
    st["elapsed_sec"] = elapsed
    st["hash"] = use_hash
    print(f"[FiveMapRNG] n={st['n']} ones%={st['ones_frac']*100:.3f}% "
          f"flip={st['flip_rate']*100:.3f}% runs={st['runs']} Lrun={st['longest_run']} "
          f"hash={'ON' if use_hash else 'OFF'} time={elapsed:.3f}s")

    if args.stats:
        with open(args.output + ".json", "w") as jf:
            json.dump(st, jf, indent=2)
        print(f"[FiveMapRNG] Stats -> {args.output}.json")

    # Optional plot (first 10k cumulative walk)
    try:
        import matplotlib.pyplot as plt
        cum = 0; walk = []
        for b in bits[:10_000]:
            cum += (1 if b else -1); walk.append(cum)
        plt.figure()
        plt.plot(walk)
        plt.title("Five-Map RNG - Cumulative Random Walk (first 10k)")
        plt.xlabel("Index"); plt.ylabel("Cumulative sum (+1/-1)")
        plt.tight_layout()
        plt.savefig("fivemap_rng_plot.png")
        print("[FiveMapRNG] Saved plot as fivemap_rng_plot.png")
    except Exception as e:
        print(f"[FiveMapRNG] Plot skipped: {e}")
