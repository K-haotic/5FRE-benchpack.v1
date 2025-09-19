🎲⚡ 5FREΞ Toy RNG Engine

by Steven Ray Britt aka Mr. Khaotic & The 5FREΞ Research Team

🚀 What Is This?

The Toy RNG Engine is a chaos-based random number generator.
It’s tiny, fast, and made for education, research, and evaluation only.

This is not a cryptographic RNG — it’s a way to see chaos in action and check if randomness can emerge straight from math.

🧩 How It Works

🌀 Generates bits from a logistic map (r = 3.9)

🔀 Whitens the bits with an XOR step

🧹 Runs a Von Neumann corrector to remove bias

📂 Outputs both raw binary and plots for inspection

🛠️ How To Run
python ToyRNGEngine.py


You’ll get:

🗂️ toy_rng_output.bin → random bitstream file

🖼️ toy_rng_plot.png → random walk plot (first 10k bits)

📊 Test Results (100k corrected bits)

We ran the engine and here’s what came out:

⏱️ Runtime: 0.216 s

1️⃣ Ones: 50,185 (50.185%)

0️⃣ Zeros: 49,815 (49.815%)

🔬 Monobit p-value: 0.242 (balanced)

🔀 Flip rate: 50.98% (ideal ≈ 50%)

🏃 Runs: 50,985

📏 Longest run: 17 (reasonable for 100k bits)

📉 Autocorrelation: lag1 = −0.0197, lag2 = +0.0476 (near 0)

👉 Looks healthy for a toy chaos RNG!

🔬 How To Use the .bin File

The toy_rng_output.bin file is a packed bitstream.
Each 8 bits are stored as a byte.

You can:

Load it into Python and unpack bits for experiments

Run it through randomness test suites:

🧪 NIST STS

🎯 Dieharder

🚦 PractRand

📜 License
5FREΞ Toy RNG Engine — Demonstration / Evaluation License
Copyright (c) 2025 Steven Ray Britt

Permission is granted, free of charge, to use and study this software for
non-commercial, research, and educational evaluation only.

Redistribution, modification, or commercial use requires prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
Author: Steven Ray Britt (Mr. Khaotic)
with the 5FREΞ Research Team

⚠️ Disclaimer

❌ Not cryptographically secure

✅ Good for demos, research, and chaos exploration

🕶️ Use at your own risk

🤝 Support Us

This project is free of charge and driven by independent research.
But to grow we need:

💸 Funding

🧠 Research partners

🔗 Collaborators

👉 If you’re an investor, researcher, or builder, join us.
Support the 5FREΞ Research Team and help push chaos research further.

✨ In short: The Toy RNG Engine is our hello world of chaos.
Run it, test it, and if you see the vision — support the mission
