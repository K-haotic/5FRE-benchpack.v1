# 🎲⚡ 5FREΞ Toy RNG Engine
# Minimal Chaotic RNG for Education & Evaluation
# Author: Steven Ray Britt (Mr. Khaotic) & The 5FREΞ Research Team
# License: Demonstration/Evaluation Only — see README & LICENSE

import random

def generate_random_bits_raw(count: int, r: float = 3.9, seed: float | None = None):
    """
    Produce 'count' raw bits from a logistic map in the chaotic regime.
    This is PRE-whitening / PRE-Von Neumann.
    """
    rnd = random.Random()
    if seed is not None:
        rnd.seed(seed)
    x = rnd.random()
    raw_bits = []
    append = raw_bits.append
    for _ in range(count):
        x = r * x * (1 - x)
        append(1 if x >= 0.5 else 0)
    return raw_bits

def xor_whiten_adjacent(bits):
    """Simple adjacent-bit XOR whitening."""
    n = len(bits)
    if n < 2:
        return bits[:]
    return [bits[i] ^ bits[i + 1] for i in range(n - 1)]

def von_neumann_correct(bits):
    """
    Von Neumann extractor:
      01 -> 0
      10 -> 1
      00, 11 -> discard
    """
    out = []
    i, n = 0, len(bits)
    append = out.append
    while i < n - 1:
        b1, b2 = bits[i], bits[i + 1]
        if b1 != b2:
            append(b1)  # (0,1) -> 0  ;  (1,0) -> 1
        i += 2
    return out

def generate_random_bits(count: int, r: float = 3.9, seed: float | None = None):
    """
    Convenience: generate ~count corrected bits via fixed-steps pipeline.
    (Note: actual corrected length may be lower due to Von Neumann discards.)
    """
    raw = generate_random_bits_raw(count, r=r, seed=seed)
    white = xor_whiten_adjacent(raw)
    corrected = von_neumann_correct(white)
    return corrected

def generate_target_bits(target_bits: int, r: float = 3.9, seed: float | None = None, max_iters: int | None = None):
    """
    Generate AT LEAST 'target_bits' corrected bits (keeps iterating until enough).
    Useful because Von Neumann discards an unknown fraction of bits.

    max_iters guards against infinite loops (defaults to 50x target raw cycles).
    """
    if target_bits <= 0:
        return []

    rnd = random.Random()
    if seed is not None:
        rnd.seed(seed)
    x = rnd.random()

    out = []
    append_out = out.append
    produced = 0
    # Choose a chunk size to amortize overhead
    chunk_raw = max(50_000, target_bits * 5)
    # sanity max
    if max_iters is None:
        max_iters = 50

    iters = 0
    while len(out) < target_bits and iters < max_iters:
        # raw chunk
        raw = []
        append_raw = raw.append
        for _ in range(chunk_raw):
            x = r * x * (1 - x)
            append_raw(1 if x >= 0.5 else 0)
        # whiten + VN
        white = xor_whiten_adjacent(raw)
        i = 0
        n = len(white)
        while i < n - 1 and len(out) < target_bits:
            b1, b2 = white[i], white[i + 1]
            if b1 != b2:
                append_out(b1)
            i += 2
        iters += 1

    return out[:target_bits]

def _pack_bits_to_bytes(bits):
    """Pack bit list into bytes (big-endian within each byte)."""
    full = (len(bits) // 8) * 8
    buf = bytearray()
    byte_val = 0
    for i, bit in enumerate(bits[:full], 1):
        byte_val = (byte_val << 1) | (1 if bit else 0)
        if i % 8 == 0:
            buf.append(byte_val & 0xFF)
            byte_val = 0
    return bytes(buf)

def _basic_stats(bits):
    n = len(bits)
    ones = sum(bits)
    zeros = n - ones
    ones_frac = (ones / n) if n else 0.0
    # runs + flip rate + longest run
    if n < 2:
        return {
            "n": n, "ones": ones, "zeros": zeros, "ones_frac": ones_frac,
            "runs": n, "longest_run": n, "flip_rate": 0.0
        }
    runs = 1
    longest = 1
    current = 1
    flips = 0
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            flips += 1
            runs += 1
            if current > longest:
                longest = current
            current = 1
        else:
            current += 1
    if current > longest:
        longest = current
    flip_rate = flips / (n - 1)
    return {
        "n": n, "ones": ones, "zeros": zeros, "ones_frac": ones_frac,
        "runs": runs, "longest_run": longest, "flip_rate": flip_rate
    }

if __name__ == "__main__":
    # Example: request 100k corrected bits (keeps iterating until enough are produced)
    TARGET_BITS = 100_000
    bits = generate_target_bits(TARGET_BITS)

    # Save binary output
    out_filename = "toy_rng_output.bin"
    with open(out_filename, "wb") as f:
        f.write(_pack_bits_to_bytes(bits))
    print(f"[ToyRNG] Corrected bits: {len(bits)}. Saved to {out_filename}")

    # Print quick stats
    st = _basic_stats(bits)
    print(f"[ToyRNG] Stats: n={st['n']} ones={st['ones']} zeros={st['zeros']} ones%={st['ones_frac']*100:.2f}%")
    print(f"[ToyRNG] runs={st['runs']} longest_run={st['longest_run']} flip_rate={st['flip_rate']*100:.2f}%")

    # Optional: simple plot if matplotlib is available (no colors specified)
    try:
        import matplotlib.pyplot as plt
        # cumulative random walk of first 10k bits
        walk = []
        cum = 0
        for b in bits[:10_000]:
            cum += (1 if b else -1)
            walk.append(cum)
        plt.figure()
        plt.plot(walk)
        plt.title("Toy RNG - Cumulative Random Walk (first 10k bits)")
        plt.xlabel("Index"); plt.ylabel("Cumulative sum (+1/-1)")
        plt.tight_layout()
        plt.savefig("toy_rng_plot.png")
        print("[ToyRNG] Saved plot as toy_rng_plot.png")
        # plt.show()  # enable if running interactively
    except Exception as e:
        print(f"[ToyRNG] Plotting skipped: {e}")
