🎲⚡ 5FREΞ Toy RNG Engine

by Steven Ray Britt aka Mr. Khaotic & The 5FREΞ Research Team

🚀 What Is This?

The Toy RNG Engine is a chaos-based random number generator.
It’s simple, fast, and meant for education, research, and evaluation only.

This isn’t a cryptographic RNG — it’s a way to see chaos in action and test if raw math can produce randomness.

🧩 How It Works

🌀 Starts with a chaotic logistic map (r = 3.9)

🔀 Turns map outputs into 0s and 1s

✨ Whitens the bits by XOR-ing neighbors

🧹 Von Neumann corrector removes bias

📂 Saves raw output (toy_rng_output.bin) + 🖼️ plots (toy_rng_plot.png)

🛠️ How To Run
python ToyRNGEngine.py


You’ll get:

toy_rng_output.bin → random bitstream file

toy_rng_plot.png → random walk + histogram

📊 Benchmarks

Typical run on a laptop:

⚡ 200k bits generated in < 1 second

✅ ~50% ones / ~50% zeros (balanced)

🔀 Flip rate ~49–51%

🏃 Runs test passes, no obvious bias

🔬 Verify It Yourself

Want to check if it’s really random?
Take the .bin file and run it through:

🧪 NIST STS

🎯 Dieharder

🚦 PractRand

You don’t have to just trust us — you can test it.

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

✅ Great for demos, learning, chaos research

🕶️ Use at your own risk

🤝 Support Us

This is open research, built free of charge.
But projects like this need 💸 funding, 🧠 collaborators, and 🔗 partnerships.

👉 If you’re an investor, researcher, developer, or just curious, reach out and support the work.
Together, we can push 5FREΞ to the next level.

✨ In short: The Toy RNG Engine is our hello world of chaos. Run it, break it, study it — and if you see the vision, join us
