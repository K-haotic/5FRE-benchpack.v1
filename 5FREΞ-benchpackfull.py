# === BHUAT 5-Field Recursion — full harness + post-run analyzer (single cell) ===
# Paste this cell and run. Uses only NumPy + Matplotlib.

import numpy as np, csv, re as _re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# Text feature front-end  (for seeding from text if you want)
# =============================================================================
_PUNC = "!?.,;:"
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_IDX = {ch:i for i,ch in enumerate(_LETTERS)}
_NEG_RE   = _re.compile(r"\b(not|no|never|n't)\b", _re.I)
_SARC_RE  = _re.compile(r"\b(lol|lmao|/s|yeah\s*right)\b", _re.I)
_QUOTE_RE = _re.compile(r"['\"“”‘’]")

def amplitude_features(text: str) -> np.ndarray:
    v = np.zeros(34, float)
    t = (text or "")
    low = t.lower()
    for ch in low:
        if ch in _IDX: v[_IDX[ch]] += 1.0
    base = 26
    for j,ch in enumerate(_PUNC): v[base + j] = low.count(ch)
    toks = t.split()
    v[32] = float(len(toks))
    v[33] = float(np.mean([len(w) for w in toks])) if toks else 0.0
    n = float(np.linalg.norm(v))
    return v / (n + 1e-8)

def phase_vector(text: str) -> np.ndarray:
    t = (text or "")
    if not t: return np.zeros(8, float)
    toks = t.split(); N = max(1, len(toks))
    ups = sum(sum(ch.isupper() for ch in w) for w in toks)
    ups_tot = sum(len(w) for w in toks)
    upper_ratio = (ups / max(1, ups_tot)) if ups_tot else 0.0
    ex_ratio = min(1.0, t.count("!") / N)
    q_ratio  = min(1.0, t.count("?") / N)
    neg    = 1.0 if _NEG_RE.search(t) else 0.0
    quotes = 1.0 if _QUOTE_RE.search(t) else 0.0
    ellip  = 1.0 if "..." in t else 0.0
    sarc   = 1.0 if _SARC_RE.search(t) or "sure..." in t.lower() else 0.0
    vowels = sum(ch in "aeiouAEIOU" for ch in t)
    vow_ratio = vowels / max(1, len(t))
    phi0 =  upper_ratio * (np.pi/2)
    phi1 =  ex_ratio   * (np.pi/2)
    phi2 =  0.0 if neg == 0.0 else np.pi
    phi3 = (vow_ratio - 0.5) * (np.pi/2)
    phi4 =  q_ratio    * (np.pi/2)
    phi5 =  quotes     * (np.pi/3)
    phi6 =  ellip      * (np.pi/3)
    phi7 =  sarc       * (np.pi/4)
    return np.array([phi0,phi1,phi2,phi3,phi4,phi5,phi6,phi7], float)

def coherence_index(phi: np.ndarray) -> float:
    if phi.size == 0: return 0.0
    c = np.cos(phi).mean(); s = np.sin(phi).mean()
    return float(np.hypot(c, s))

# =============================================================================
# 5FRE map, Jacobian, Lyapunov, DKY
# =============================================================================
def default_params():
    return dict(a=1.4, k1=0.05, c1=0.25, c2=0.25, c3=0.25, c4=0.50, d=0.20)

def tuned_params():
    # Asymmetric, slightly lower damping — thicker projections
    return dict(a=1.56, k1=0.05, c1=0.29, c2=0.21, c3=0.33, c4=0.55, d=0.16)

def seed_from_text(text: str, scale_ic: float = 0.5, mod_params: bool = True):
    amp = amplitude_features(text); ph = phase_vector(text); coh = coherence_index(ph)
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    w = rng.normal(0, 1, size=5)
    base = np.array([
        amp[:6].sum() - amp[6:12].sum(),    # φ
        amp[12:18].sum() - amp[18:24].sum(),# τ
        amp[24:30].sum() - amp[30:].sum(),  # ψ
        np.sin(ph[:4]).sum() * 0.1,         # χ
        np.cos(ph[4:]).sum() * 0.1,         # Λ
    ])
    x0 = scale_ic * (base + 0.05 * w)
    p = default_params().copy()
    if mod_params:
        ex_q = (ph[1] + ph[4]) / (np.pi)
        p["d"]  = float(p["d"]  * (1.0 - 0.05 * coh))
        p["k1"] = float(p["k1"] * (1.0 + 0.05 * ex_q))
        p["c1"] = float(p["c1"] * (1.0 + 0.03 * np.tanh(ph[0] - ph[3])))
        p["c2"] = float(p["c2"] * (1.0 + 0.03 * np.tanh(ph[5] - ph[6])))
        p["c3"] = float(p["c3"] * (1.0 + 0.03 * np.tanh(ph[7] - ph[2])))
    return x0.astype(float), p

def step_5fre(x, p):
    phi,tau,psi,chi,Lam = x
    a,k1,c1,c2,c3,c4,d = p["a"],p["k1"],p["c1"],p["c2"],p["c3"],p["c4"],p["d"]
    return np.array([
        1.0 - a*phi*phi + k1*Lam,
        c1*phi + (1.0-c1)*tau,
        c2*tau + (1.0-c2)*psi,
        c3*psi + (1.0-c3)*chi,
        c4*chi - d*Lam
    ], float)

def jacobian_5fre(x, p):
    phi,tau,psi,chi,Lam = x
    a,k1,c1,c2,c3,c4,d = p["a"],p["k1"],p["c1"],p["c2"],p["c3"],p["c4"],p["d"]
    J = np.zeros((5,5), float)
    J[0,0] = -2.0*a*phi; J[0,4] = k1
    J[1,0] = c1;         J[1,1] = 1.0-c1
    J[2,1] = c2;         J[2,2] = 1.0-c2
    J[3,2] = c3;         J[3,3] = 1.0-c3
    J[4,3] = c4;         J[4,4] = -d
    return J

def run_map(params, x0, steps, burn):
    assert steps > burn >= 0
    x = np.array(x0, float)
    traj = np.empty((steps,5), float)
    for i in range(steps):
        traj[i] = x
        x = step_5fre(x, params)
    return traj[burn:]

def lyapunov_qr(params, traj):
    n, dim = traj.shape[0], 5
    Q = np.eye(dim); sums = np.zeros(dim, float)
    for i in range(n):
        Z = jacobian_5fre(traj[i], params) @ Q
        Q, R = np.linalg.qr(Z)
        sums += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
    lambdas = sums / n
    return np.sort(lambdas)[::-1]

def kaplan_yorke(lam):
    l = np.sort(np.asarray(lam))[::-1]
    S = 0.0; Spos = 0.0; j = 0
    for i in range(len(l)-1):
        S += l[i]
        if S >= 0:
            Spos = S; j = i + 1
        else:
            break
    if j == 0: return 0.0
    if j >= len(l): return float(len(l))
    return j + Spos / abs(l[j])

# Optional tiny Λ-gated micro-bursts for tuned mode
def step_with_kick(x, p, rng, sigma0=1e-3):
    phi,tau,psi,chi,Lam = x
    x1 = step_5fre(x, p)
    Lambda_c = 0.18 * (1.0 + 0.35 * (1.0 - np.tanh(3.0*chi)))
    if Lam >= Lambda_c:
        kick = sigma0 * (Lam - Lambda_c + 1e-6)
        x1[0] += rng.normal(0.0, kick)
        x1[2] += rng.normal(0.0, kick)
    return x1

# =============================================================================
# Improved 0–1 chaos test (RAM-light) for a contiguous series
# =============================================================================
def zero_one_K_improved(y, m=64, ngrid=128, seed=1):
    y = (y - y.mean()) / (y.std() + 1e-16)
    N = y.size
    t = np.arange(1, N+1, dtype=np.float64)
    rng = np.random.default_rng(seed)
    cs = rng.uniform(0.2, np.pi-0.2, size=m)
    Ks = []
    for c in cs:
        cosct = np.cos(c*t); sinct = np.sin(c*t)
        P = np.cumsum(y * cosct)
        Q = np.cumsum(y * sinct)
        nmax = N // 10
        ns = np.linspace(1, nmax, ngrid, dtype=int)
        M = np.empty(ns.size, dtype=np.float64)
        for k, n in enumerate(ns):
            dP = P[n:] - P[:-n]
            dQ = Q[n:] - Q[:-n]
            M[k] = np.mean(dP*dP + dQ*dQ)
        n0 = ns - ns.mean(); M0 = M - M.mean()
        Kc = (n0 @ M0) / (np.sqrt((n0 @ n0) * (M0 @ M0)) + 1e-16)
        Ks.append(np.clip(Kc, 0.0, 1.0))
    return float(np.median(Ks))

# =============================================================================
# Post-run ANALYZER for an existing trajectory.csv
# =============================================================================
def analyze_csv(path="trajectory.csv", spectrum_path=None,
                thin=1, do_plots=True, proj=(0,2,4),
                do_01=True, do_poincare=True, write_summary=True):
    # Load trajectory
    D = np.loadtxt(path, delimiter=",", skiprows=1)
    names = ["phi","tau","psi","chi","Lambda"]
    # Stats
    mins = D.min(axis=0); maxs = D.max(axis=0)
    means = D.mean(axis=0); stds = D.std(axis=0)
    # Correlations
    C = np.corrcoef(D.T)
    # Optional 0–1 test (phi and mixed z)
    K_phi = K_z = None
    if do_01:
        K_phi = zero_one_K_improved(D[:,0])
        z = 0.7*D[:,0] + 0.2*D[:,1] + 0.1*D[:,2]
        K_z = zero_one_K_improved(z)

    # Print summary
    print("\n=== Post-run analysis ===")
    for i,n in enumerate(names):
        print(f"{n:7s}: min={mins[i]:+.6f}  max={maxs[i]:+.6f}  mean={means[i]:+.6f}  std={stds[i]:+.6f}")
    print("\nCorrelation matrix (phi,tau,psi,chi,Lambda):\n", np.array_str(C, precision=3, suppress_small=True))
    if K_phi is not None:
        print(f"\n0–1 chaos K (phi): {K_phi:.3f}   |   0–1 chaos K (z=0.7φ+0.2τ+0.1ψ): {K_z:.3f}")

    # If there is a sidecar spectrum file, show it
    if spectrum_path is None and path.endswith(".csv"):
        spectrum_path = path.rsplit(".",1)[0] + "_spectrum.csv"
    try:
        S = np.loadtxt(spectrum_path, delimiter=",", skiprows=1)
        if S.ndim == 1 and S.size >= 6:
            lam = S[:5]; DKY = S[5]
            print("\nSidecar spectrum:", lam, "  DKY:", float(DKY))
    except Exception:
        pass

    # Plots (each on its own figure)
    if do_plots:
        plt.figure()
        plt.plot(D[:,0])
        plt.xlabel(r"$\phi$"); plt.ylabel("value"); plt.title(r"Time series: $\phi$")
        plt.tight_layout(); plt.show()

        i,j,k = proj
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(D[::thin,i], D[::thin,j], D[::thin,k], lw=0.3)
        labels = [r"$\phi$", r"$\tau$", r"$\psi$", r"$\chi$", r"$\Lambda$"]
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.set_zlabel(labels[k])
        ax.set_title(f"Projection (from CSV): {labels[i]}, {labels[j]}, {labels[k]}")
        fig.tight_layout(); plt.show()

        if do_poincare:
            # χ=0 upward crossings from file
            pts = []
            X = D
            for t in range(X.shape[0]-1):
                chi_prev, chi_now = X[t,3], X[t+1,3]
                if chi_prev < 0.0 <= chi_now and (chi_now - chi_prev) > 0.0:
                    pts.append([X[t,0], X[t,4]])  # φ vs Λ at crossing
            P = np.array(pts, float) if pts else np.zeros((0,2))
            plt.figure()
            if P.size:
                plt.plot(P[:,0], P[:,1], '.', ms=1)
            plt.xlabel(r"$\phi$"); plt.ylabel(r"$\Lambda$")
            plt.title("Poincaré section (from CSV): χ=0 upward — ϕ vs Λ")
            plt.tight_layout(); plt.show()

    # Optional summary.txt
    if write_summary:
        with open("summary.txt", "w") as f:
            f.write("BHUAT 5FRE — post-run summary\n")
            for i,n in enumerate(names):
                f.write(f"{n:7s}: min={mins[i]:+.6f}  max={maxs[i]:+.6f}  mean={means[i]:+.6f}  std={stds[i]:+.6f}\n")
            f.write("\nCorrelation matrix (phi,tau,psi,chi,Lambda):\n")
            for row in C:
                f.write(" ".join(f"{v:+.3f}" for v in row) + "\n")
            if K_phi is not None:
                f.write(f"\n0–1 chaos K (phi): {K_phi:.3f}\n0–1 chaos K (z): {K_z:.3f}\n")
        print("\nWrote summary.txt")

# =============================================================================
# Main runner (simulation + export + quick plots), then analyzer call
# =============================================================================
def run_inline(*, text="hello Λ-world!!",
               steps=80_000, burn=10_000, out="trajectory.csv",
               plot=True, proj=(0,2,4),
               use_tuned=True, use_text_mod=False, seed=123):
    x0, p = seed_from_text(text, mod_params=use_text_mod)
    if use_tuned: p = tuned_params()

    # simulate
    if use_tuned:
        rng = np.random.default_rng(seed)
        x = np.array(x0, float)
        traj = np.empty((steps,5), float)
        for i in range(steps):
            traj[i] = x
            x = step_with_kick(x, p, rng)
        traj = traj[burn:]
    else:
        traj = run_map(p, x0, steps, burn)

    # spectrum + DKY
    lam = lyapunov_qr(p, traj)
    DKY = kaplan_yorke(lam)

    # CSVs
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phi","tau","psi","chi","Lambda"]); w.writerows(traj.tolist())
    with open(out.rsplit(".",1)[0] + "_spectrum.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda_1","lambda_2","lambda_3","lambda_4","lambda_5","DKY"])
        w.writerow(list(lam) + [DKY])

    # Console summary
    print("Parameters:", p)
    print("Initial condition:", x0)
    print("Lyapunov spectrum (descending):", lam)
    print("Kaplan–Yorke dimension:", DKY)

    # Plots
    if plot:
        plt.figure()
        plt.plot(traj[:,0])
        plt.xlabel(r"$\phi$"); plt.ylabel("value"); plt.title(r"Time series: $\phi$")
        plt.tight_layout(); plt.show()

        i,j,k = proj
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:,i], traj[:,j], traj[:,k], lw=0.3)
        labels = [r"$\phi$", r"$\tau$", r"$\psi$", r"$\chi$", r"$\Lambda$"]
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.set_zlabel(labels[k])
        ax.set_title(f"Projection: {labels[i]}, {labels[j]}, {labels[k]}")
        fig.tight_layout(); plt.show()

    return out

# ===========================
# RUN (edit these knobs)
# ===========================
TEXT        = "hello Λ-world!!"
STEPS       = 80_000
BURN        = 10_000
OUT_CSV     = "trajectory.csv"
PROJECTION  = (0,2,4)     # (phi, psi, Lambda)
USE_TUNED   = True        # True -> tuned chaotic basin; False -> canonical
USE_TEXTMOD = False       # keep False for fixed params

# 1) simulate + export + plots
csv_path = run_inline(text=TEXT, steps=STEPS, burn=BURN, out=OUT_CSV,
                      plot=True, proj=PROJECTION, use_tuned=USE_TUNED,
                      use_text_mod=USE_TEXTMOD, seed=123)

# 2) analyze the CSV (stats, correlations, 0–1 test, Poincaré, summary.txt)
analyze_csv(path=csv_path, spectrum_path=None, thin=1,
            do_plots=True, proj=PROJECTION, do_01=True, do_poincare=True,
            write_summary=True)
# === Add-on: Correlation dimension + RQA from a trajectory CSV ===
import numpy as np, matplotlib.pyplot as plt, os

def _corrsum_theiler(X, idx_orig, radii, theiler=50, block=400):
    N = X.shape[0]
    counts = np.zeros_like(radii, dtype=np.float64)
    for i0 in range(0, N, block):
        i1 = min(N, i0+block)
        Xi = X[i0:i1]; idxi = idx_orig[i0:i1]
        # within block
        for ii in range(i1-i0-1):
            d = np.sqrt(np.sum((Xi[ii+1:]-Xi[ii])**2, axis=1))
            dt = np.abs(idxi[ii+1:] - idxi[ii])
            ok = dt > theiler
            if ok.any():
                dd = d[ok]
                counts += np.array([(dd < r).sum() for r in radii], float)
        # cross block
        Xrest = X[i1:]; idxrest = idx_orig[i1:]
        if Xrest.size:
            dists = np.sqrt(((Xi[:,None,:]-Xrest[None,:,:])**2).sum(axis=2))
            DT = np.abs(idxi[:,None] - idxrest[None,:])
            ok = DT > theiler
            if ok.any():
                for j,r in enumerate(radii):
                    counts[j] += (dists[ok] < r).sum()
    return counts * 2.0 / (N*(N-1))

def _kaplan_yorke(lam):
    l = np.sort(np.asarray(lam))[::-1]
    S = 0.0; Spos = 0.0; j = 0
    for i in range(len(l)-1):
        S += l[i]
        if S >= 0:
            Spos = S; j = i+1
        else:
            break
    if j == 0: return 0.0
    if j >= len(l): return float(len(l))
    return j + Spos/abs(l[j])

def analyze_from_csv(csv_path="trajectory.csv", spectrum_path=None,
                     theiler=200, maxN_corr=5000, maxN_rqa=2500):
    X = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    names = ["phi","tau","psi","chi","Lambda"]
    # ---- D2 on full 5D
    stride = max(1, X.shape[0] // maxN_corr)
    Xsub = X[::stride, :]
    idx_orig = np.arange(0, X.shape[0], stride)[:Xsub.shape[0]]
    # set radii from a quick sample
    rng = np.random.default_rng(0)
    m = min(1200, Xsub.shape[0])
    samp = Xsub[rng.choice(Xsub.shape[0], size=m, replace=False)]
    # condensed pairwise distances (sample) to pick radius range
    dsm = []
    for i in range(m-1):
        dsm.append(np.sqrt(np.sum((samp[i+1:]-samp[i])**2, axis=1)))
    dsm = np.concatenate(dsm)
    p = np.percentile(dsm, [5, 50, 90])
    radii = np.logspace(np.log10(max(1e-3, p[0]*0.6)),
                        np.log10(p[1]*1.4), 20)
    C = _corrsum_theiler(Xsub, idx_orig, radii, theiler=theiler, block=300)
    mask = (C > 1e-3) & (C < 0.5)
    logr = np.log(radii[mask]); logC = np.log(C[mask])
    A = np.vstack([logr, np.ones_like(logr)]).T
    slope, b = np.linalg.lstsq(A, logC, rcond=None)[0]
    print(f"Correlation dimension D2 (5D, Theiler={theiler}): ~ {slope:.3f}")
    # plot scaling
    plt.figure(); plt.loglog(radii, C, marker='o')
    plt.xlabel("r"); plt.ylabel("C(r)"); plt.title("Correlation sum scaling (5D)")
    plt.tight_layout(); plt.show()

    # ---- RQA on (phi,psi,Lambda)
    Y = X[:maxN_rqa, :][:,[0,2,4]]
    N = Y.shape[0]
    dists = np.sqrt(((Y[:,None,:]-Y[None,:,:])**2).sum(axis=2))
    for i in range(N):
        s = max(0, i-theiler); e = min(N, i+theiler+1)
        dists[i, s:e] = np.inf
    eps = np.percentile(dists[np.isfinite(dists)], 10.0)
    R = (dists < eps).astype(np.uint8)
    RR = R.sum() / (N*N)
    lmin = 2; DET_points = 0; rec_points = R.sum()
    for k in range(-N+1, N):
        diag = np.diag(R, k=k)
        if diag.size == 0: continue
        run = 0
        for v in diag:
            if v: run += 1
            else:
                if run >= lmin: DET_points += run
                run = 0
        if run >= lmin: DET_points += run
    DET = DET_points / max(1, rec_points)
    print(f"RQA (on (phi,psi,Lambda), N={N}, Theiler={theiler}): RR≈{RR:.3f}, DET≈{DET:.3f}")

    # ---- sidecar spectrum (if present)
    if spectrum_path is None and csv_path.endswith(".csv"):
        spectrum_path = csv_path.rsplit(".",1)[0] + "_spectrum.csv"
    try:
        S = np.loadtxt(spectrum_path, delimiter=",", skiprows=1)
        if S.ndim == 1 and S.size >= 6:
            lam = S[:5]; DKY = S[5]
            print("Sidecar λ:", lam, " DKY:", float(DKY), "| recomputed DKY:", _kaplan_yorke(lam))
    except Exception:
        pass

# Example:
# analyze_from_csv("trajectory.csv")
# === Toy-vs-Full comparison (single cell, paste below your harness) ===
import numpy as np, matplotlib.pyplot as plt, csv

# --- Toy model (3-field) ---
def step_toy3(x, p):
    phi, tau, psi = x
    a, c1, c2 = p["a"], p["c1"], p["c2"]
    return np.array([
        1.0 - a*phi*phi,                # logistic core, no Λ feedback
        c1*phi + (1.0-c1)*tau,          # follower 1
        c2*tau + (1.0-c2)*psi           # follower 2
    ], float)

def jacobian_toy3(x, p):
    phi, tau, psi = x
    a, c1, c2 = p["a"], p["c1"], p["c2"]
    J = np.zeros((3,3), float)
    J[0,0] = -2.0*a*phi
    J[1,0] = c1;     J[1,1] = 1.0-c1
    J[2,1] = c2;     J[2,2] = 1.0-c2
    return J

def run_toy3(p, x0, steps, burn):
    x = np.array(x0, float)
    traj = np.empty((steps,3), float)
    for i in range(steps):
        traj[i] = x
        x = step_toy3(x, p)
    return traj[burn:]

def lyapunov_qr_toy(p, traj):
    n = traj.shape[0]
    Q = np.eye(3); s = np.zeros(3)
    for i in range(n):
        Z = jacobian_toy3(traj[i], p) @ Q
        Q, R = np.linalg.qr(Z)
        s += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
    lam = np.sort(s / n)[::-1]
    return lam

# --- 0–1 chaos test (use your improved version if already defined) ---
def zero_one_K_improved(y, m=64, ngrid=128, seed=1):
    y = (y - y.mean()) / (y.std() + 1e-16)
    N = y.size
    t = np.arange(1, N+1, dtype=np.float64)
    rng = np.random.default_rng(seed)
    cs = rng.uniform(0.2, np.pi-0.2, size=m)
    Ks = []
    for c in cs:
        cosct = np.cos(c*t); sinct = np.sin(c*t)
        P = np.cumsum(y * cosct)
        Q = np.cumsum(y * sinct)
        nmax = N // 10
        ns = np.linspace(1, nmax, ngrid, dtype=int)
        M = np.empty(ns.size, dtype=np.float64)
        for k, n in enumerate(ns):
            dP = P[n:] - P[:-n]
            dQ = Q[n:] - Q[:-n]
            M[k] = np.mean(dP*dP + dQ*dQ)
        n0 = ns - ns.mean(); M0 = M - M.mean()
        Kc = (n0 @ M0) / (np.sqrt((n0 @ n0) * (M0 @ M0)) + 1e-16)
        Ks.append(np.clip(Kc, 0.0, 1.0))
    return float(np.median(Ks))

# --- Comparison driver ---
def compare_toy_vs_full(text="hello Λ-world!!", steps=80_000, burn=10_000,
                        use_tuned_full=True, use_text_mod_full=False,
                        plot=True):
    # seed from text (reuse your harness helpers)
    x0_full, p_full = seed_from_text(text, mod_params=use_text_mod_full)
    if use_tuned_full:
        p_full = tuned_params()
    # full trajectory via your runner pieces
    x = np.array(x0_full, float)
    traj_full = np.empty((steps,5), float)
    rng = np.random.default_rng(123)
    # we use your tuned/no-tuned policy: if tuned, use kick; else vanilla
    def step_with_kick(x, p, rng, sigma0=1e-3):
        phi,tau,psi,chi,Lam = x
        x1 = step_5fre(x, p)
        Lambda_c = 0.18 * (1.0 + 0.35 * (1.0 - np.tanh(3.0*chi)))
        if Lam >= Lambda_c:
            kick = sigma0 * (Lam - Lambda_c + 1e-6)
            x1[0] += rng.normal(0.0, kick)
            x1[2] += rng.normal(0.0, kick)
        return x1
    for i in range(steps):
        traj_full[i] = x
        if use_tuned_full:
            x = step_with_kick(x, p_full, rng)
        else:
            x = step_5fre(x, p_full)
    traj_full = traj_full[burn:]

    lam_full = lyapunov_qr(p_full, traj_full)
    DKY_full = kaplan_yorke(lam_full)
    K_full = zero_one_K_improved(traj_full[:,0])

    # toy parameters: borrow a,c1,c2 from full; ignore c3,c4,d,k1
    p_toy = dict(a=p_full["a"], c1=p_full["c1"], c2=p_full["c2"])
    # toy IC: (phi, tau, psi) from full IC projection
    x0_toy = x0_full[:3]
    traj_toy = run_toy3(p_toy, x0_toy, steps, burn)
    lam_toy = lyapunov_qr_toy(p_toy, traj_toy)
    # DKY for toy (3D)
    def kaplan_yorke_3(lam):
        l = np.sort(np.asarray(lam))[::-1]
        S = 0.0; Spos = 0.0; j = 0
        for i in range(len(l)-1):
            S += l[i]
            if S >= 0: Spos, j = S, i+1
            else: break
        if j == 0: return 0.0
        if j >= len(l): return float(len(l))
        return j + Spos/abs(l[j])
    DKY_toy = kaplan_yorke_3(lam_toy)
    K_toy = zero_one_K_improved(traj_toy[:,0])

    # Print comparison
    print("\n=== TOY vs FULL (same text) ===")
    print("Text:", text)
    print("\nFull params:", p_full)
    print("Toy params :", p_toy)
    print("\nLyapunov (toy):", lam_toy)
    print("DKY (toy):", DKY_toy, "   0–1 K (toy):", round(K_toy,3))
    print("\nLyapunov (full):", lam_full)
    print("DKY (full):", DKY_full, "   0–1 K (full):", round(K_full,3))

    if plot:
        # φ time series (toy vs full) -- one figure
        plt.figure()
        plt.plot(traj_toy[:,0], alpha=0.8)
        plt.plot(traj_full[:,0], alpha=0.6)
        plt.xlabel(r"$\phi$"); plt.ylabel("value"); plt.title(r"Time series: $\phi$ (toy vs full)")
        plt.tight_layout(); plt.show()

        # φ histograms (toy vs full) -- separate figures per requirement
        import numpy as np
        plt.figure()
        plt.hist(traj_toy[:,0], bins=100)
        plt.xlabel(r"$\phi$"); plt.ylabel("count"); plt.title(r"Toy: $\phi$ histogram")
        plt.tight_layout(); plt.show()

        plt.figure()
        plt.hist(traj_full[:,0], bins=100)
        plt.xlabel(r"$\phi$"); plt.ylabel("count"); plt.title(r"Full: $\phi$ histogram")
        plt.tight_layout(); plt.show()

    # Return a small dict for programmatic checks
    return {
        "toy":  {"lam": lam_toy,  "DKY": DKY_toy,  "K": K_toy},
        "full": {"lam": lam_full, "DKY": DKY_full, "K": K_full}
    }

# Example run:
_ = compare_toy_vs_full(text="BHUAT: compare toy vs full", steps=80_000, burn=10_000,
                        use_tuned_full=True, use_text_mod_full=False, plot=True)

# === Full 5FRE grid sweep over (a, d): λ1, DKY, χ–Λ corr, Σλ  ===
# Assumes your harness cell already defined:
#   seed_from_text, run_map, lyapunov_qr, kaplan_yorke, step_5fre, jacobian_5fre, default_params, tuned_params

import numpy as np, csv
import matplotlib.pyplot as plt

def sweep_full_ad(text="hello Λ-world!!",
                  a_grid=np.linspace(1.35, 1.65, 13),
                  d_grid=np.linspace(0.14, 0.22, 17),
                  base="tuned",          # "tuned" or "canonical"
                  use_text_mod=False,    # keep False for fair sweep
                  steps=80_000, burn=10_000,
                  out_csv="sweep_ad.csv",
                  plot=True):
    # --- pick baseline params
    if base.lower().startswith("t"):
        base_params = tuned_params()
    else:
        base_params = default_params()

    # --- seed IC (fixed) from text; do NOT mod params during sweep unless asked
    x0, _ = seed_from_text(text, mod_params=False)

    # storage
    A, D = np.meshgrid(a_grid, d_grid, indexing="xy")
    lam1_grid   = np.full(A.shape, np.nan, float)
    dky_grid    = np.full(A.shape, np.nan, float)
    sum_grid    = np.full(A.shape, np.nan, float)
    corr_chiLam = np.full(A.shape, np.nan, float)

    rows = []
    for ia, a in enumerate(a_grid):
        for idd, d in enumerate(d_grid):
            p = base_params.copy()
            if use_text_mod:
                # tiny, consistent modulation from text features
                _, p_mod = seed_from_text(text, mod_params=True)
                p.update({k: p_mod.get(k, p[k]) for k in p.keys()})
            p["a"] = float(a); p["d"] = float(d)

            # simulate
            traj = run_map(p, x0, steps, burn)

            # spectrum + DKY
            lam = lyapunov_qr(p, traj)
            DKY = kaplan_yorke(lam)
            lam1 = float(lam[0]); lam_sum = float(np.sum(lam))

            # χ–Λ correlation
            c = np.corrcoef(traj[:,3], traj[:,4])[0,1]

            lam1_grid[idd, ia]   = lam1
            dky_grid[idd, ia]    = DKY
            sum_grid[idd, ia]    = lam_sum
            corr_chiLam[idd, ia] = c

            rows.append([a, d, *lam.tolist(), DKY, lam_sum, c])

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a","d","lambda_1","lambda_2","lambda_3","lambda_4","lambda_5","DKY","sum_lambda","corr_chi_Lambda"])
        w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows.")

    # report best pockets under constraints λ1>0 and Σλ<0
    mask = (lam1_grid > 0) & (sum_grid < 0)
    if np.any(mask):
        # top-5 by lam1, then DKY
        idx = np.argwhere(mask)
        scores = lam1_grid[mask]
        order = np.argsort(-scores)  # descending
        print("\nTop pockets (λ1>0, Σλ<0):")
        for k in range(min(5, order.size)):
            i,j = idx[order[k]]
            print(f"  a={A[i,j]:.4f}, d={D[i,j]:.4f} | λ1={lam1_grid[i,j]:+.4f}, DKY={dky_grid[i,j]:.3f}, Σλ={sum_grid[i,j]:+.4f}, corr(χ,Λ)={corr_chiLam[i,j]:+.3f}")
    else:
        print("No (a,d) pairs satisfied λ1>0 and Σλ<0 in this grid.")

    if plot:
        # Each metric gets its own figure (default matplotlib style; no explicit colors)
        extent = [a_grid[0], a_grid[-1], d_grid[0], d_grid[-1]]

        plt.figure()
        plt.imshow(lam1_grid.T, origin="lower", extent=extent, aspect="auto")
        plt.colorbar()
        plt.xlabel("a"); plt.ylabel("d")
        plt.title("largest Lyapunov λ1")
        plt.tight_layout(); plt.show()

        plt.figure()
        plt.imshow(dky_grid.T, origin="lower", extent=extent, aspect="auto")
        plt.colorbar()
        plt.xlabel("a"); plt.ylabel("d")
        plt.title("Kaplan–Yorke DKY")
        plt.tight_layout(); plt.show()

        plt.figure()
        plt.imshow(sum_grid.T, origin="lower", extent=extent, aspect="auto")
        plt.colorbar()
        plt.xlabel("a"); plt.ylabel("d")
        plt.title("Σ λ (dissipativity)")
        plt.tight_layout(); plt.show()

        plt.figure()
        plt.imshow(corr_chiLam.T, origin="lower", extent=extent, aspect="auto")
        plt.colorbar()
        plt.xlabel("a"); plt.ylabel("d")
        plt.title("corr(χ, Λ)")
        plt.tight_layout(); plt.show()

    return {
        "a_grid": a_grid, "d_grid": d_grid,
        "lam1": lam1_grid, "DKY": dky_grid, "sumlam": sum_grid, "corr_chiLam": corr_chiLam
    }

# Example usage (matches your tuned-basin baseline, but sweep runs WITHOUT kicks for determinism):
_ = sweep_full_ad(
        text="BHUAT sweep",
        a_grid=np.linspace(1.40, 1.62, 12),
        d_grid=np.linspace(0.14, 0.20, 13),
        base="tuned",
        use_text_mod=False,
        steps=60_000, burn=8_000,
        out_csv="sweep_ad.csv",
        plot=True
    )
# === TOY vs FULL: same (a,d) grid, CSVs + side-by-side plots ===
import numpy as np, csv, os
import matplotlib.pyplot as plt

# --- Toy model (3-field) ------------------------------------------------------
def step_toy3(x, p):
    phi, tau, psi = x
    a, c1, c2 = p["a"], p["c1"], p["c2"]
    return np.array([
        1.0 - a*phi*phi,                # logistic core, no Λ feedback
        c1*phi + (1.0-c1)*tau,          # follower 1
        c2*tau + (1.0-c2)*psi           # follower 2
    ], float)

def jacobian_toy3(x, p):
    phi, tau, psi = x
    a, c1, c2 = p["a"], p["c1"], p["c2"]
    J = np.zeros((3,3), float)
    J[0,0] = -2.0*a*phi
    J[1,0] = c1;     J[1,1] = 1.0-c1
    J[2,1] = c2;     J[2,2] = 1.0-c2
    return J

def run_toy3(p, x0, steps, burn):
    x = np.array(x0, float)
    traj = np.empty((steps,3), float)
    for i in range(steps):
        traj[i] = x
        x = step_toy3(x, p)
    return traj[burn:]

def lyapunov_qr_toy(p, traj):
    n = traj.shape[0]
    Q = np.eye(3); s = np.zeros(3)
    for i in range(n):
        Z = jacobian_toy3(traj[i], p) @ Q
        Q, R = np.linalg.qr(Z)
        s += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
    lam = np.sort(s / n)[::-1]
    return lam

def kaplan_yorke_vec(lam):
    l = np.sort(np.asarray(lam))[::-1]
    S = 0.0; Spos = 0.0; j = 0
    for i in range(len(l)-1):
        S += l[i]
        if S >= 0: Spos, j = S, i+1
        else: break
    if j == 0: return 0.0
    if j >= len(l): return float(len(l))
    return j + Spos/abs(l[j])

# --- Toy sweep on (a,d) (d is a dummy axis for comparability) -----------------
def sweep_toy_ad(text="hello Λ-world!!",
                 a_grid=np.linspace(1.35, 1.65, 13),
                 d_grid=np.linspace(0.14, 0.22, 17),
                 base="tuned",
                 steps=60_000, burn=8_000,
                 out_csv="toy_sweep_ad.csv",
                 plot=False):
    if base.lower().startswith("t"):
        p_base = tuned_params()
    else:
        p_base = default_params()
    x0_full, _ = seed_from_text(text, mod_params=False)
    x0_toy = x0_full[:3]  # project IC to (phi,tau,psi)

    A, D = np.meshgrid(a_grid, d_grid, indexing="xy")
    lam1 = np.full(A.shape, np.nan); dky = np.full(A.shape, np.nan); ssum = np.full(A.shape, np.nan)

    rows = []
    for ia, a in enumerate(a_grid):
        for idd, d in enumerate(d_grid):  # d is ignored by toy; we still loop to match shapes
            p = dict(a=float(a), c1=p_base["c1"], c2=p_base["c2"])
            traj = run_toy3(p, x0_toy, steps, burn)
            lam = lyapunov_qr_toy(p, traj)
            DKY = kaplan_yorke_vec(lam)
            lam1[idd, ia] = lam[0]; dky[idd, ia] = DKY; ssum[idd, ia] = lam.sum()
            rows.append([a, d, *lam.tolist(), DKY, lam.sum()])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a","d","lambda_1","lambda_2","lambda_3","DKY","sum_lambda"])
        w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    if plot:
        extent = [a_grid[0], a_grid[-1], d_grid[0], d_grid[-1]]
        plt.figure(); plt.imshow(lam1.T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
        plt.xlabel("a"); plt.ylabel("d"); plt.title("TOY: λ1"); plt.tight_layout(); plt.show()
        plt.figure(); plt.imshow(dky.T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
        plt.xlabel("a"); plt.ylabel("d"); plt.title("TOY: DKY"); plt.tight_layout(); plt.show()
    return {"a_grid":a_grid, "d_grid":d_grid, "lam1":lam1, "DKY":dky, "sumlam":ssum}

# --- Combined comparison ------------------------------------------------------
def compare_sweeps(text="BHUAT sweep",
                   a_grid=np.linspace(1.40, 1.62, 12),
                   d_grid=np.linspace(0.14, 0.20, 13),
                   base="tuned",
                   steps=60_000, burn=8_000,
                   full_csv="sweep_ad.csv",
                   toy_csv="toy_sweep_ad.csv"):
    # 1) Full sweep: load if exists; else compute
    if os.path.exists(full_csv):
        print(f"Loading full sweep from {full_csv}")
        full_rows = np.loadtxt(full_csv, delimiter=",", skiprows=1)
        # columns: a,d, lambdas(5), DKY, sum, corr
        fa = full_rows[:,0]; fd = full_rows[:,1]
        A,D = np.meshgrid(a_grid, d_grid, indexing="xy")
        lam1_full = np.full(A.shape, np.nan); dky_full = np.full(A.shape, np.nan); sum_full = np.full(A.shape, np.nan)
        corr_full = np.full(A.shape, np.nan)
        for r in full_rows:
            a, d = r[0], r[1]
            ia = np.argmin(np.abs(a_grid - a)); idd = np.argmin(np.abs(d_grid - d))
            lam1_full[idd, ia] = r[2]; dky_full[idd, ia] = r[7]; sum_full[idd, ia] = r[8]; corr_full[idd, ia] = r[9]
    else:
        print("Running full model sweep (no existing CSV).")
        out = sweep_full_ad(text=text, a_grid=a_grid, d_grid=d_grid, base=base,
                            use_text_mod=False, steps=steps, burn=burn,
                            out_csv=full_csv, plot=False)
        lam1_full, dky_full, sum_full, corr_full = out["lam1"], out["DKY"], out["sumlam"], out["corr_chiLam"]

    # 2) Toy sweep
    toy = sweep_toy_ad(text=text, a_grid=a_grid, d_grid=d_grid, base=base,
                       steps=steps, burn=burn, out_csv=toy_csv, plot=False)
    lam1_toy, dky_toy, sum_toy = toy["lam1"], toy["DKY"], toy["sumlam"]

    # 3) Print pockets
    print("\nTop FULL pockets (λ1>0, Σλ<0):")
    maskF = (lam1_full > 0) & (sum_full < 0)
    if np.any(maskF):
        idx = np.argwhere(maskF); scores = lam1_full[maskF]; order = np.argsort(-scores)
        for k in range(min(5, order.size)):
            i,j = idx[order[k]]
            print(f"  a={a_grid[j]:.4f}, d={d_grid[i]:.4f} | λ1={lam1_full[i,j]:+.4f}, DKY={dky_full[i,j]:.3f}, Σλ={sum_full[i,j]:+.4f}, corr(χ,Λ)={corr_full[i,j]:+.3f}")
    else:
        print("  (none in grid)")

    print("\nTop TOY pockets (λ1>0, Σλ<0):")
    maskT = (lam1_toy > 0) & (sum_toy < 0)
    if np.any(maskT):
        idx = np.argwhere(maskT); scores = lam1_toy[maskT]; order = np.argsort(-scores)
        for k in range(min(5, order.size)):
            i,j = idx[order[k]]
            print(f"  a={a_grid[j]:.4f}, d={d_grid[i]:.4f} | λ1={lam1_toy[i,j]:+.4f}, DKY={dky_toy[i,j]:.3f}, Σλ={sum_toy[i,j]:+.4f}")
    else:
        print("  (none in grid)")

    # 4) Plots (each in its own figure)
    extent = [a_grid[0], a_grid[-1], d_grid[0], d_grid[-1]]

    plt.figure()
    plt.imshow(lam1_full.T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
    plt.xlabel("a"); plt.ylabel("d"); plt.title("FULL: λ1")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.imshow(lam1_toy.T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
    plt.xlabel("a"); plt.ylabel("d"); plt.title("TOY: λ1 (flat in d)")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.imshow((lam1_full - lam1_toy).T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
    plt.xlabel("a"); plt.ylabel("d"); plt.title("Δλ1 = FULL − TOY")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.imshow(dky_full.T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
    plt.xlabel("a"); plt.ylabel("d"); plt.title("FULL: DKY")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.imshow(dky_toy.T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
    plt.xlabel("a"); plt.ylabel("d"); plt.title("TOY: DKY (flat in d)")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.imshow((dky_full - dky_toy).T, origin="lower", extent=extent, aspect="auto"); plt.colorbar()
    plt.xlabel("a"); plt.ylabel("d"); plt.title("ΔDKY = FULL − TOY")
    plt.tight_layout(); plt.show()

    return {
        "full": {"lam1": lam1_full, "DKY": dky_full, "sum": sum_full, "corr_chiLam": corr_full},
        "toy":  {"lam1": lam1_toy,  "DKY": dky_toy,  "sum": sum_toy}
    }

# === Run the paired comparison (edit grids if you like) ===
_ = compare_sweeps(
        text="BHUAT paired toy vs full",
        a_grid=np.linspace(1.40, 1.62, 12),
        d_grid=np.linspace(0.14, 0.20, 13),
        base="tuned",
        steps=60_000, burn=8_000,
        full_csv="sweep_ad.csv",
        toy_csv="toy_sweep_ad.csv"
    )
# === Select best (a,d) from sweep_ad.csv and re-run full 5FRE ===
# Assumes your harness already defines:
#   default_params, tuned_params, run_map, lyapunov_qr, kaplan_yorke,
#   step_5fre, jacobian_5fre

import numpy as np, csv, os
import matplotlib.pyplot as plt

def pick_and_rerun_from_sweep(
    sweep_csv="sweep_ad.csv",
    base="tuned",               # "tuned" or "canonical" for the non-(a,d) params
    target_DKY=None,            # e.g., 1.5 or 2.3; None = ignore DKY target
    dky_tolerance=0.25,         # how close to target to prefer (if target_DKY is set)
    min_corr_chiLam=0.0,        # require corr(χ,Λ) ≥ this
    require_dissipative=True,   # Σλ < 0
    require_chaotic=True,       # λ1 > 0
    score_lambda_weight=1.0,    # weight for λ1 in score
    score_dky_weight=1.0,       # weight for DKY closeness (if target given)
    steps=80_000, burn=10_000,
    out_csv="best_run.csv",
    plot=True
):
    if not os.path.exists(sweep_csv):
        raise FileNotFoundError(f"Missing {sweep_csv}. Run the sweep first.")

    # Load sweep rows:
    # a,d, λ1..λ5, DKY, sum_lambda, corr_chi_Lambda
    M = np.loadtxt(sweep_csv, delimiter=",", skiprows=1)
    a = M[:,0]; d = M[:,1]
    lam1 = M[:,2]; DKY = M[:,7]; sumlam = M[:,8]; corr = M[:,9]

    # Constraints
    mask = np.ones(M.shape[0], dtype=bool)
    if require_dissipative: mask &= (sumlam < 0)
    if require_chaotic:     mask &= (lam1 > 0)
    mask &= (corr >= min_corr_chiLam)
    if target_DKY is not None:
        mask &= (np.abs(DKY - target_DKY) <= dky_tolerance)

    if not np.any(mask):
        print("No entries satisfy constraints. Relax filters and retry.")
        return None

    # Score: maximize λ1; optionally add DKY closeness bonus
    score = score_lambda_weight * lam1.copy()
    if target_DKY is not None:
        closeness = 1.0 - np.minimum(1.0, np.abs(DKY - target_DKY) / max(1e-9, dky_tolerance))
        score += score_dky_weight * closeness

    idx = np.argmax(np.where(mask, score, -np.inf))
    a_star, d_star = float(a[idx]), float(d[idx])
    lam_row = M[idx, 2:7]; DKY_row = float(DKY[idx]); sum_row = float(sumlam[idx]); corr_row = float(corr[idx])

    # Build full parameter set from base and chosen (a,d)
    p = tuned_params() if base.lower().startswith("t") else default_params()
    p["a"] = a_star; p["d"] = d_star

    # IC: small, neutral seed so results depend on params (not text modulation)
    x0 = np.array([0.1, 0.0, 0.0, 0.0, 0.2], float)

    # Re-run (no kicks, no text-mod)
    traj = run_map(p, x0, steps, burn)
    lam = lyapunov_qr(p, traj)
    DKY_re = kaplan_yorke(lam)
    corr_chiLam = float(np.corrcoef(traj[:,3], traj[:,4])[0,1])

    # Save CSVs
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phi","tau","psi","chi","Lambda"]); w.writerows(traj.tolist())
    with open(out_csv.rsplit(".",1)[0] + "_spectrum.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda_1","lambda_2","lambda_3","lambda_4","lambda_5","DKY"])
        w.writerow(list(lam) + [DKY_re])

    # Report
    print("\n=== Selected pocket from sweep ===")
    print(f"a*={a_star:.5f}, d*={d_star:.5f} | sweep λ1={lam1[idx]:+.5f}, DKY={DKY_row:.3f}, Σλ={sum_row:+.5f}, corr(χ,Λ)={corr_row:+.3f}")
    print("\n=== Re-run with selected params (no kicks) ===")
    print("Parameters:", p)
    print("Lyapunov spectrum (descending):", lam)
    print("Kaplan–Yorke dimension:", DKY_re)
    print("corr(χ,Λ):", corr_chiLam)
    print(f"Wrote {out_csv} and {out_csv.rsplit('.',1)[0]}_spectrum.csv")

    # Plots (each figure separate)
    if plot:
        plt.figure()
        plt.plot(traj[:,0])
        plt.xlabel(r"$\phi$"); plt.ylabel("value"); plt.title(r"Time series: $\phi$ (best pocket)")
        plt.tight_layout(); plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:,0], traj[:,2], traj[:,4], lw=0.3)
        labels = [r"$\phi$", r"$\tau$", r"$\psi$", r"$\chi$", r"$\Lambda$"]
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[2]); ax.set_zlabel(labels[4])
        ax.set_title("Projection: φ, ψ, Λ (best pocket)")
        fig.tight_layout(); plt.show()

    return dict(params=p, lam=lam, DKY=DKY_re, corr_chiLam=corr_chiLam, idx=idx)

# === Example: pick best pocket close to DKY≈2.3 with good λ1, dissipative, corr(χ,Λ)≥0.8 ===
_ = pick_and_rerun_from_sweep(
        sweep_csv="sweep_ad.csv",
        base="tuned",
        target_DKY=2.3, dky_tolerance=0.25,
        min_corr_chiLam=0.80,
        require_dissipative=True, require_chaotic=True,
        score_lambda_weight=1.0, score_dky_weight=0.8,
        steps=80_000, burn=10_000,
        out_csv="best_run.csv",
        plot=True
    )

# === Canonical vs Tuned (side-by-side) ===
import numpy as np, matplotlib.pyplot as plt, csv

# Fallback 0–1 test if not in notebook
def _zero_one_K_safe(y):
    try:
        return zero_one_K_improved(y)
    except NameError:
        y = (y - y.mean()) / (y.std() + 1e-16)
        N = y.size; t = np.arange(1, N+1, dtype=float)
        c = 1.7  # single frequency fallback
        P = np.cumsum(y * np.cos(c*t)); Q = np.cumsum(y * np.sin(c*t))
        n = np.arange(1, N//10+1)
        M = np.array([np.mean((P[k:]-P[:-k])**2 + (Q[k:]-Q[:-k])**2) for k in n])
        n0 = n - n.mean(); M0 = M - M.mean()
        K = (n0 @ M0) / np.sqrt((n0 @ n0) * (M0 @ M0) + 1e-16)
        return float(np.clip(K, 0, 1))

def canonical_params():
    return dict(a=1.4, k1=0.05, c1=0.25, c2=0.25, c3=0.25, c4=0.50, d=0.20)

def tuned_params_local():
    return dict(a=1.56, k1=0.05, c1=0.29, c2=0.21, c3=0.33, c4=0.55, d=0.16)

def compare_canonical_tuned(text="hello Λ-world!!", steps=80_000, burn=10_000,
                            proj=(0,2,4), save=False):
    # fixed IC independent of text modulation
    x0 = np.array([0.10, 0.00, 0.00, 0.00, 0.20], float)

    # --- canonical
    pC = canonical_params()
    trajC = run_map(pC, x0, steps, burn)
    lamC = lyapunov_qr(pC, trajC); DKYC = kaplan_yorke(lamC)
    KC   = _zero_one_K_safe(trajC[:,0])
    corrC = float(np.corrcoef(trajC[:,3], trajC[:,4])[0,1])

    # --- tuned
    pT = tuned_params_local()
    trajT = run_map(pT, x0, steps, burn)  # no kicks here for fair comparison
    lamT = lyapunov_qr(pT, trajT); DKYT = kaplan_yorke(lamT)
    KT   = _zero_one_K_safe(trajT[:,0])
    corrT = float(np.corrcoef(trajT[:,3], trajT[:,4])[0,1])

    # --- print compact table
    def row(label, lam, dky, K, corr):
        s = np.sum(lam)
        return f"{label:10s} | λ1={lam[0]:+7.4f}  Σλ={s:+7.4f}  DKY={dky:5.3f}  K={K:4.3f}  corr(χ,Λ)={corr:+5.3f}"
    print(row("canonical", lamC, DKYC, KC, corrC))
    print(row("tuned",     lamT, DKYT, KT, corrT))

    # optional CSVs
    if save:
        with open("canonical_traj.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["phi","tau","psi","chi","Lambda"]); w.writerows(trajC.tolist())
        with open("tuned_traj.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["phi","tau","psi","chi","Lambda"]); w.writerows(trajT.tolist())

    # --- plots (one figure per chart; default matplotlib style)
    plt.figure()
    plt.plot(trajC[:,0], alpha=0.9, label="canonical φ")
    plt.plot(trajT[:,0], alpha=0.7, label="tuned φ")
    plt.xlabel(r"$\phi$"); plt.ylabel("value"); plt.title(r"Time series: $\phi$ (canonical vs tuned)")
    plt.legend(); plt.tight_layout(); plt.show()

    i,j,k = proj
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajC[:,i], trajC[:,j], trajC[:,k], lw=0.3)
    labels = [r"$\phi$", r"$\tau$", r"$\psi$", r"$\chi$", r"$\Lambda$"]
    ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.set_zlabel(labels[k])
    ax.set_title("Projection (canonical): φ, ψ, Λ")
    fig.tight_layout(); plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajT[:,i], trajT[:,j], trajT[:,k], lw=0.3)
    ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.set_zlabel(labels[k])
    ax.set_title("Projection (tuned): φ, ψ, Λ")
    fig.tight_layout(); plt.show()

    return dict(canonical=dict(params=pC, lam=lamC, DKY=DKYC, K=KC, corr=corrC),
                tuned=dict(params=pT, lam=lamT, DKY=DKYT, K=KT, corr=corrT))

# Run it:
_ = compare_canonical_tuned()


