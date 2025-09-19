# 5FREΞ Toy Engine (benchpackToy.py)

**Project:** Phase-Aware Logic Engine — Toy Demonstration
**File:** `benchpackToy.py`
**Author:** Steven Britt
**License:** Demonstration-only (Do not reuse without written permission)

---

## 🔹 What This Is

This script is the **Toy version** of the 5FREΞ semantic recursion system — a lightweight standalone engine that shows how:

* Phase vector encoding
* χ-coherence (logic/memory agreement)
* Λ-gate (symbolic bifurcation check)

...can be used to outperform cosine similarity on semantically tricky examples like negation or tonal shifts ("not good", "yeah right", etc).

This is **not the full engine**, and **not the Lite version** either. This is the raw concept, proven with a few examples, and minimal moving parts.

---

## ✅ What It Demonstrates

* `phase_vector(text)`: Extracts symbolic/emotional phase (caps, punctuation, tone)
* `chi_coherence(p, a)`: Measures logic agreement between phase and amplitude
* `lambda_gate(p, a, chi)`: Decides true/false output based on phase + coherence
* `run_inline(text)`: Run an individual example and see what it returns

**Typical Results:**

* "happy": ✔ gate open
* "not happy": ✖ gate closed
* "lol yeah right": ✖ gate closed
* "sure! great." ✔ gate open

This proves that **phase-aware gating + coherence checking** outperforms cosine alone in handling "not-X" cases.

---

## 🧪 Use Case

Use this to:

* Prove the symbolic field concept works without training
* Benchmark against cosine similarity on tonal/negated inputs
* Show potential of symbolic/recursive logic to infra teams

It’s **safe to share** for technical demo purposes — no core recursion loop, no attractor, no memory field — just the logic gate concept.

---

## 🧱 Roadmap Position

| Tier           | Includes                                                                        |
| -------------- | ------------------------------------------------------------------------------- |
| **Toy**        | ✔ Phase logic, ✔ χ-coherence, ✔ Λ-gate                                          |
| **Lite**       | + Calibration stats, + JSON I/O, + seed batch runner                            |
| **Full 5FREΞ** | + Recursion loop, + Lyapunov shell, + attractor memory, + symbolic merge engine |

This Toy version lives at the bottom layer — pure logic. It’s your clean pitch and testbed.

---

## ▶️ How to Use

Run it like this:

```python
python benchpackToy.py
```

Or use inline:

```python
from benchpackToy import run_inline
run_inline("not happy")
```

---

## 🔐 Legal & IP

* This file is **evaluation/demo only** — do not copy, repackage, or reuse in any system without written permission.
* Covered under the 5FREΞ system patent disclosures (2025).
* For licensing discussions: stevenbrittdev and 5FREΞ dev teams---> khaoticgamingltd@gmail.com

---

## 📎 Summary

This file shows that **symbolic phase + logic coherence + pressure gate** outperforms cosine on tricky semantics. It proves the smallest useful unit of 5FREΞ.

**It is the seed — not the tree.**
