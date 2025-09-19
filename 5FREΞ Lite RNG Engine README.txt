🎲⚡ 5FREΞ Lite RNG Engine

by Steven Ray Britt aka Mr. Khaotic & The 5FREΞ Research Team

🚀 What Is This?

The Lite RNG Engine is a step-up chaos-based random number generator.
It’s more advanced than the Toy RNG — designed for evaluation and research only.

Not cryptographic, but it’s a solid way to see dual chaos at work.

🧩 How It Works

🌀 Runs two logistic maps with slightly different chaos parameters

🔀 XORs the two streams together

🧹 Von Neumann corrector removes bias

🔒 Optionally conditions output with SHA-256 for extra mixing

📂 Saves raw output (lite_rng_output.bin) + 🖼️ plots (lite_rng_plot.png)

🛠️ How To Run
# Generate 200k bits with SHA-256 conditioning
python LiteRNGEngine.py -n 200000

# Generate without hash conditioning
python LiteRNGEngine.py --nohash


Outputs:

lite_rng_output.bin → random bitstream file

lite_rng_plot.png → random walk + histogram

📊 Benchmarks

On a standard laptop:

⚡ 200k bits in ~1–2 seconds

✅ ~50% ones / ~50% zeros

🔀 Flip rate ~49–51%

🏃 Runs test passes reliably

🔒 With SHA-256 conditioning → smoother entropy, fewer edge-case biases

🔬 Verify It Yourself

Take the .bin file and run it through:

🧪 NIST STS

🎯 Dieharder

🚦 PractRand

You don’t have to trust us — you can test it directly.

📜 License
5FREΞ Lite RNG Engine — Evaluation License (Free of charge, non-commercial)
Copyright (c) 2025 Steven Ray Britt

Permission is granted, free of charge, to use and study this software for
non-commercial, research, and educational evaluation only.

Redistribution, modification, or commercial use requires prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
Author: Steven Ray Britt (Mr. Khaotic)
with the 5FREΞ Research Team

⚠️ Disclaimer

❌ Not cryptographically secure

✅ Stronger than Toy RNG, shows chaotic dual-map behavior

🕶️ Use at your own risk

🤝 Support Us

We’re pushing the boundaries of chaos, recursion, and randomness.
But to take this further, we need 💸 funding, 🧠 research partners, and 🔗 collaborators.

👉 If you’re an investor, researcher, or builder, join us.
Support the 5FREΞ Research Team and help turn this into the future of secure chaos.

✨ In short: The Lite RNG Engine is our next-level chaos generator. It’s free, open for study, and proof that raw math can bend toward randomness.
