#!/usr/bin/env python3
# 5FRE-benchpack — single-file PoC (v1.5.0)
# Patent pending. Evaluation-only: internal non-commercial testing permitted;
# no production/redistribution/derivatives without written permission.

import os, sys, json, argparse, re, math, time, zipfile, io
from typing import List, Tuple, Dict, Any, Iterable

try:
    import numpy as np
except Exception:
    raise SystemExit("Missing dependency: numpy\nInstall with:  pip install numpy")

VERSION = "1.5.0"

# =============================================================================
# FEATURES
# =============================================================================

_PUNC = "!?.,;:"
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_IDX = {ch:i for i,ch in enumerate(_LETTERS)}

def amplitude_features(text: str) -> np.ndarray:
    v = np.zeros(34, dtype=float)
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

_NEG_RE   = re.compile(r"\b(not|no|never|n't)\b", re.I)
_SARC_RE  = re.compile(r"\b(lol|lmao|/s|yeah\s*right)\b", re.I)
_QUOTE_RE = re.compile(r"[\"'“”‘’]")

def phase_vector(text: str) -> np.ndarray:
    t = (text or "")
    if not t: return np.zeros(8, dtype=float)
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

    phi0 =  upper_ratio * (np.pi/2)         # CAPS
    phi1 =  ex_ratio   * (np.pi/2)          # !
    phi2 =  0.0 if neg == 0.0 else np.pi    # NOT flip
    phi3 = (vow_ratio - 0.5) * (np.pi/2)    # vowels
    phi4 =  q_ratio    * (np.pi/2)          # ?
    phi5 =  quotes     * (np.pi/3)          # quotes
    phi6 =  ellip      * (np.pi/3)          # …
    phi7 =  sarc       * (np.pi/4)          # sarcasm
    return np.array([phi0,phi1,phi2,phi3,phi4,phi5,phi6,phi7], float)

def coherence_index(phi: np.ndarray) -> float:
    if phi.size == 0: return 0.0
    c = np.cos(phi).mean(); s = np.sin(phi).mean()
    return float(np.hypot(c, s))  # 0..1

def phase_alignment(phi1: np.ndarray, phi2: np.ndarray,
                    w: Tuple[float,...] = None) -> float:
    if w is None:
        w = (1.0, 1.0, 4.0, 0.6, 0.8, 0.6, 0.6, 0.7)  # [caps,!,NOT,vowel,?,quotes,ellipsis,sarcasm]
    w = np.asarray(w, float)
    cs = np.cos(phi1 - phi2)
    align_raw = float((w * cs).sum() / (w.sum() + 1e-8))
    neg_same = (np.cos(phi1[2] - phi2[2]) + 1.0) / 2.0  # 1 if same side, 0 if π flip
    return float(neg_same * align_raw)

def lambda_gate(chi: float, align: float, lam: float,
                k_open: float = 0.6, k_close: float = 0.4, decay: float = 0.1):
    opened = lam > k_close
    if chi > k_open and align > 0.6:
        lam = min(1.0, lam + 0.2); opened = True
    else:
        lam = max(0.0, lam - decay)
    return lam, opened

def score_pair(a_text: str, b_text: str, signed: bool = False) -> Dict[str, float]:
    A1, A2 = amplitude_features(a_text), amplitude_features(b_text)
    cosA = float(np.dot(A1, A2))  # [0,1]
    phi1, phi2 = phase_vector(a_text), phase_vector(b_text)
    align = phase_alignment(phi1, phi2)       # [-1,1] with neg-guard
    chi = (coherence_index(phi1) + coherence_index(phi2)) / 2.0
    lam = 0.0; lam, opened = lambda_gate(chi, align, 0.0)
    phase_score = cosA * (align if signed else max(0.0, align))
    return {
        "a": a_text, "b": b_text,
        "cos_amp": round(cosA, 6),
        "phase_align": round(align, 6),
        "chi": round(chi, 6),
        "lambda": round(lam, 6),
        "gate_open": bool(opened),
        "phase_score": round(phase_score, 6)
    }

# =============================================================================
# METRICS / STATS
# =============================================================================

def _threshold_sweep(scores: List[float], labels: List[int]) -> Tuple[float, float, Dict[str,int]]:
    if not scores:
        return 0.0, 0.5, {"tp":0,"fp":0,"tn":0,"fn":0}
    zs = sorted(set(scores))
    cands = [(zs[i] + zs[min(i+1, len(zs)-1)])*0.5 for i in range(len(zs))]
    cands = sorted(set([0.0, 0.5, 1.0] + cands))
    best_acc, best_t, best_cm = -1.0, 0.5, None
    for t in cands:
        tp=fp=tn=fn=0
        for s,y in zip(scores, labels):
            yhat = 1 if s >= t else 0
            if y==1 and yhat==1: tp+=1
            elif y==0 and yhat==1: fp+=1
            elif y==0 and yhat==0: tn+=1
            else: fn+=1
        acc = (tp+tn)/max(1,(tp+tn+fp+fn))
        if acc > best_acc:
            best_acc, best_t, best_cm = acc, t, {"tp":tp,"fp":fp,"tn":tn,"fn":fn}
    return best_acc, best_t, best_cm

def _ece(scores: List[float], labels: List[int], bins: int = 10) -> float:
    if not scores: return 0.0
    scores = np.asarray(scores, float); labels = np.asarray(labels, float)
    edges = np.linspace(0.0, 1.0, bins+1)
    ece, N = 0.0, float(len(scores))
    for i in range(bins):
        lo, hi = edges[i], edges[i+1] + (1e-12 if i==bins-1 else 0.0)
        mask = (scores >= lo) & (scores < hi)
        m = int(mask.sum())
        if m == 0: continue
        conf = float(scores[mask].mean())
        acc = float(labels[mask].mean())
        ece += (m / N) * abs(conf - acc)
    return float(ece)

def roc_auc(scores: List[float], labels: List[int]) -> float:
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    ths = np.linspace(0.0, 1.0, 501)
    P = max(1, int(y.sum())); N = max(1, int((y==0).sum()))
    fprs, tprs = [], []
    for t in ths:
        yhat = (s >= t).astype(int)
        tp = int(((yhat==1)&(y==1)).sum())
        fp = int(((yhat==1)&(y==0)).sum())
        fn = int(((yhat==0)&(y==1)).sum())
        # tn = N - fp
        fprs.append(fp/N); tprs.append(tp/P)
    # trapezoid rule, ensuring monotonic sort by FPR
    fprs, tprs = np.array(fprs), np.array(tprs)
    order = np.argsort(fprs); fprs, tprs = fprs[order], tprs[order]
    return float(np.trapz(tprs, fprs))

def bootstrap_ci(vals: Iterable[float], B: int = 500, seed: int = 1234) -> Tuple[float,float]:
    vals = list(vals); n = len(vals)
    if n == 0: return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean([vals[i] for i in idx])))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)

# Calibration
def _sigmoid(z):  # numeric stability
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

def fit_platt(scores, labels, chis=None, iters=250, lr=0.5):
    s = np.asarray(scores, float); y = np.asarray(labels, float)
    x2 = (np.asarray(chis, float) - 0.5) if chis is not None else None
    a, b, c = 1.0, 0.0, 0.0
    for _ in range(iters):
        z = a*s + b + (c*x2 if x2 is not None else 0.0)
        p = _sigmoid(z)
        g_a = np.sum((p - y) * s)
        g_b = np.sum(p - y)
        if x2 is not None: g_c = np.sum((p - y) * x2)
        n = len(s) + 1e-8
        a -= lr * g_a / n; b -= lr * g_b / n
        if x2 is not None: c -= lr * g_c / n
    return (a,b,c) if x2 is not None else (a,b,0.0)

def apply_platt(scores, chis, params):
    a,b,c = params
    s = np.asarray(scores, float)
    x2 = (np.asarray(chis, float) - 0.5)
    return _sigmoid(a*s + b + c*x2)

def fit_isotonic(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, float)
    order = np.argsort(s); s = s[order]; y = y[order]
    p = y.astype(float).copy(); w = np.ones_like(p)
    i = 0
    while i < len(p)-1:
        if p[i] <= p[i+1]:
            i += 1; continue
        j = i
        while j >= 0 and p[j] > p[j+1]:
            tot_w = w[j] + w[j+1]
            avg = (p[j]*w[j] + p[j+1]*w[j+1]) / tot_w
            p[j] = p[j+1] = avg
            w[j] = w[j+1] = tot_w
            j -= 1
        i += 1
    return s, p

def apply_isotonic(scores, model):
    xs, ys = model
    s = np.asarray(scores, float)
    idx = np.searchsorted(xs, s, side="right") - 1
    idx = np.clip(idx, 0, len(xs)-1)
    return ys[idx]

def conformal_threshold(scores, labels, target_fpr=0.05):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    neg = s[y == 0]
    if neg.size == 0: return 0.5
    return float(np.quantile(neg, 1.0 - target_fpr))

# Plots (matplotlib, separate figures, no explicit colors)
def plot_reliability(scores: List[float], labels: List[int], out_png: str, bins: int = 10):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    scores = np.asarray(scores, float); labels = np.asarray(labels, float)
    edges = np.linspace(0,1,bins+1)
    xs, ys = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1] + (1e-12 if i==bins-1 else 0.0)
        m = (scores>=lo)&(scores<hi)
        if m.sum()==0: continue
        xs.append(float(scores[m].mean()))
        ys.append(float(labels[m].mean()))
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(xs, ys, marker="o")
    plt.xlabel("confidence"); plt.ylabel("empirical accuracy")
    plt.title("Reliability diagram")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
    return True

def plot_roc(scores: List[float], labels: List[int], out_png: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    ths = np.linspace(0.0, 1.0, 201)
    P = max(1, int(y.sum())); N = max(1, int((y==0).sum()))
    fprs, tprs = [], []
    for t in ths:
        yhat = (s >= t).astype(int)
        tp = int(((yhat==1)&(y==1)).sum())
        fp = int(((yhat==1)&(y==0)).sum())
        fprs.append(fp/N); tprs.append(tp/P)
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(fprs, tprs)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC (phase score)")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
    return True

# =============================================================================
# DATA (+ augmentation)
# =============================================================================

def default_pairs_labels() -> List[Tuple[str,str,int,str]]:
    return [
        ("yes", "YES", 1, "emphasis"),
        ("okay.", "ok", 1, "synonym"),
        ("thank you", "not thank you", 0, "negation"),
        ("hey!", "hey", 1, "emphasis"),
        ("I am happy", "I am not happy", 0, "negation"),
        ("very good", "good", 1, "modifier"),
        ("sure", "not sure", 0, "negation"),
        ("great!", "great", 1, "emphasis"),
    ]

EXTENDED_SAMPLE = [
  ["yes","YES",1,"emphasis"], ["okay.","ok",1,"synonym"],
  ["thank you","not thank you",0,"negation"], ["I am happy","I am not happy",0,"negation"],
  ["sure","not sure",0,"negation"], ["great!","great",1,"emphasis"],
  ["you sure?","sure",0,"question"], ["\"great\"", "great", 1, "quotes"],
  ["sure...", "sure", 1, "ellipsis"], ["yeah right", "great", 0, "sarcasm"],
  ["very good", "good", 1, "modifier"], ["really good", "not good", 0, "negation"],
  ["love this!", "love this", 1, "emphasis"], ["ok", "not ok", 0, "negation"],
  ["are you sure?", "you sure?", 1, "question"], ["fine...", "fine", 1, "ellipsis"],
  ["'fine'", "fine", 1, "quotes"], ["LOL ok", "ok", 1, "sarcasm"],
  ["happy", "unhappy", 0, "negation"], ["good", "GOOD", 1, "emphasis"],
  ["thank you!", "thanks", 1, "synonym"], ["no thanks", "thanks", 0, "negation"],
  ["really?", "really", 0, "question"], ["okay", "okay", 1, "synonym"]
]

def load_pairs_labels(path: str) -> List[Tuple[str,str,int,str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for row in data:
        if isinstance(row, list) and (len(row)==3 or len(row)==4):
            a,b,l = row[0], row[1], int(row[2])
            tag = row[3] if len(row)==4 else "__untagged__"
        elif isinstance(row, dict):
            a,b,l = row.get("a",""), row.get("b",""), int(row.get("label",0))
            tag = row.get("tag","__untagged__")
        else:
            raise SystemExit("pairs file must be [a,b,label,(tag)] or {a,b,label,tag}")
        out.append((str(a), str(b), l, str(tag)))
    return out

def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

# Simple augmenter to bulk the set with punctuation/case/negation jitter
def auto_augment(base: List[Tuple[str,str,int,str]], N: int, seed: int = 1337) -> List[Tuple[str,str,int,str]]:
    rng = np.random.default_rng(seed)
    def tweak(s: str) -> str:
        s2 = s
        # random caps
        if rng.random() < 0.3: s2 = s2.upper()
        # add/remove !
        if rng.random() < 0.3: s2 = s2 + "!"*(1+rng.integers(0,2))
        # add question
        if rng.random() < 0.2: s2 = s2 + "?"
        # add ellipsis/quotes
        if rng.random() < 0.2: s2 = s2 + ("..." if rng.random()<0.5 else "")
        if rng.random() < 0.1: s2 = '"' + s2 + '"'
        return s2
    def neg_flip(s: str) -> str:
        # naive inject/remove "not" at start for augmentation purposes
        if _NEG_RE.search(s): return re.sub(_NEG_RE, "", s).strip()
        return "not " + s
    out = list(base)
    k = len(base)
    while len(out) < N:
        a,b,l,tag = base[rng.integers(0,k)]
        if tag=="negation" and rng.random()<0.5:
            # swap polarity sometimes
            if l==0:
                out.append((a, tweak(neg_flip(b)), 0, tag))
            else:
                out.append((a, tweak(neg_flip(b)), 1, tag))
        else:
            out.append((tweak(a), tweak(b), l, tag))
    return out[:N]

# =============================================================================
# EVAL + REPORT
# =============================================================================

def evaluate(pairs_labels: List[Tuple[str,str,int,str]],
             out_jsonl: str, summary_json: str, csv_out: str,
             also_print: bool, signed: bool, seed: int,
             cal_mode: str, split_frac: float, ece_bins: int,
             target_fpr_list: List[float],
             readme_md: str,
             rel_raw_png: str, rel_cal_png: str, roc_png: str,
             zip_artifacts: str, write_eval_license: bool):

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)

    base_scores, phase_scores, y, chis, tags, logs = [], [], [], [], [], []
    gate_false_open = gate_missed_open = 0; pos = neg = 0

    for a,b,label,tag in pairs_labels:
        rec = score_pair(a,b, signed=signed)
        rec["label"] = int(label); rec["tag"] = tag
        logs.append(rec)
        base_scores.append(rec["cos_amp"])
        phase_scores.append(rec["phase_score"])
        y.append(int(label)); chis.append(rec["chi"]); tags.append(tag)
        if label==0:
            neg += 1;  gate_false_open += int(rec["gate_open"])
        else:
            pos += 1;  gate_missed_open += int(not rec["gate_open"])
        if also_print:
            print(json.dumps(rec, ensure_ascii=False))

    # save JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in logs: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # base vs phase
    base_acc, base_t, base_cm = _threshold_sweep(base_scores, y)
    ph_acc, ph_t, ph_cm       = _threshold_sweep(phase_scores, y)
    base_ece = _ece(base_scores, y, bins=ece_bins)
    ph_ece   = _ece(phase_scores, y, bins=ece_bins)
    fo_rate = gate_false_open / max(1, neg)
    mo_rate = gate_missed_open / max(1, pos)

    # AUC
    auc_phase = roc_auc(phase_scores, y)

    # per-tag (phase at τ*)
    per_tag: Dict[str, Dict[str, Any]] = {}
    for s,lab,tag in zip(phase_scores, y, tags):
        yhat = 1 if s >= ph_t else 0
        d = per_tag.setdefault(tag, {"n":0, "correct":0, "tp":0,"fp":0,"tn":0,"fn":0})
        d["n"] += 1; d["correct"] += int(yhat == lab)
        if lab==1 and yhat==1: d["tp"]+=1
        elif lab==0 and yhat==1: d["fp"]+=1
        elif lab==0 and yhat==0: d["tn"]+=1
        else: d["fn"]+=1
    for tag,d in per_tag.items():
        d["acc"] = round(d["correct"]/max(1,d["n"]), 4); d.pop("correct", None)

    # CSV export (per-record + preds at τ*)
    if csv_out:
        import csv
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a","b","tag","label","cos_amp","phase_align","chi","lambda","gate_open",
                        "phase_score","pred_phase@tau","pred_base@tau"])
            for rec, s_b, s_p, lab in zip(logs, base_scores, phase_scores, y):
                yhat_p = 1 if s_p >= ph_t else 0
                yhat_b = 1 if s_b >= base_t else 0
                w.writerow([rec["a"],rec["b"],rec.get("tag",""),lab,rec["cos_amp"],
                            rec["phase_align"],rec["chi"],rec["lambda"],rec["gate_open"],
                            rec["phase_score"], yhat_p, yhat_b])

    # calibration(s)
    n = len(y); split = max(2, int(max(0.1, min(0.9, split_frac)) * n))
    if n < 3: split = n
    cal_blocks: List[Dict[str,Any]] = []; best_cal: Dict[str,Any] = {}
    if cal_mode in ("platt","platt+chi","iso","auto") and n >= 3:
        tr, te = slice(0,split), slice(split,n)
        ph_train, y_train = phase_scores[tr], y[tr]
        ph_test,  y_test  = phase_scores[te],  y[te]
        chi_train,chi_test= chis[tr], chis[te]
        modes = ["platt","platt+chi","iso"] if cal_mode=="auto" else [cal_mode]
        for mode in modes:
            if mode=="platt":
                params = fit_platt(ph_train, y_train, chis=None)
                p_cal  = apply_platt(ph_test, [0]*len(y_test), params)
                meta = {"params": [float(x) for x in params]}
            elif mode=="platt+chi":
                params = fit_platt(ph_train, y_train, chis=chi_train)
                p_cal  = apply_platt(ph_test, chi_test, params)
                meta = {"params": [float(x) for x in params]}
            else:
                iso_model = fit_isotonic(ph_train, y_train)
                p_cal  = apply_isotonic(ph_test, iso_model)
                meta = {"knots": len(iso_model[0])}
            ece_cal = _ece(list(p_cal), list(y_test), bins=ece_bins)
            tp=fp=tn=fn=0
            for p_,ytrue in zip(p_cal, y_test):
                yhat = 1 if p_ >= 0.5 else 0
                if ytrue==1 and yhat==1: tp+=1
                elif ytrue==0 and yhat==1: fp+=1
                elif ytrue==0 and yhat==0: tn+=1
                else: fn+=1
            acc_cal = (tp+tn)/max(1,(tp+tn+fp+fn))
            block = {"mode": mode, "holdout_size": len(y_test),
                     "acc@0.5": round(acc_cal,4), "ece": round(ece_cal,4),
                     "cm": {"tp":tp,"fp":fp,"tn":tn,"fn":fn}, **meta}
            cal_blocks.append(block)
        best_cal = min(cal_blocks, key=lambda d: d["ece"]) if cal_blocks else {}

    # Conformal thresholds
    conf_list = []
    if target_fpr_list and n >= 3:
        tr, te = slice(0,split), slice(split,n)
        for tfpr in target_fpr_list:
            tau = conformal_threshold(phase_scores[tr], [int(v) for v in y[tr]], tfpr)
            tp=fp=tn=fn=0
            for s,ytrue in zip(phase_scores[te], y[te]):
                yhat = 1 if s >= tau else 0
                if ytrue==1 and yhat==1: tp+=1
                elif ytrue==0 and yhat==1: fp+=1
                elif ytrue==0 and yhat==0: tn+=1
                else: fn+=1
            acc = (tp+tn)/max(1,(tp+tn+fp+fn))
            conf_list.append({"target_fpr": float(tfpr), "tau": round(float(tau),6),
                              "acc": round(acc,4), "cm": {"tp":tp,"fp":fp,"tn":tn,"fn":fn}})

    # Bootstrap CIs (accuracy/ECE for phase)
    ph_preds_tau = [1 if s >= ph_t else 0 for s in phase_scores]
    acc_per_item = [int(p==yy) for p,yy in zip(ph_preds_tau, y)]
    # For ECE we approximate with bin contributions; here we bootstrap mean of |conf-acc| at item level using confidence=s (rough)
    ece_item = [abs(s - yy) for s,yy in zip(phase_scores, y)]
    acc_lo, acc_hi = bootstrap_ci(acc_per_item, B=400, seed=seed)
    ece_lo, ece_hi = bootstrap_ci(ece_item, B=400, seed=seed)

    summary = {
        "version": VERSION,
        "n": len(y),
        "seed": seed,
        "baseline": {"best_acc": round(base_acc,4), "threshold": round(base_t,4),
                     "ece": round(base_ece,4), "cm": base_cm},
        "phase_aware": {"best_acc": round(ph_acc,4), "threshold": round(ph_t,4),
                        "ece": round(ph_ece,4), "cm": ph_cm, "auc": round(auc_phase,4),
                        "acc_ci95": [round(acc_lo,4), round(acc_hi,4)],
                        "ece_ci95": [round(ece_lo,4), round(ece_hi,4)]},
        "gate_quality": {"false_open_rate": round(fo_rate,4),
                         "missed_open_rate": round(mo_rate,4),
                         "negatives": neg, "positives": pos},
        "per_tag": per_tag
    }
    if cal_blocks:
        summary["phase_calibrated_all"] = cal_blocks
        summary["phase_calibrated_best"] = best_cal
    if conf_list:
        summary["phase_conformal"] = conf_list

    write_json(summary, summary_json)

    # plots
    if rel_raw_png: plot_reliability(phase_scores, y, rel_raw_png, bins=max(6, ece_bins))
    if rel_cal_png and best_cal:
        tr, te = slice(0, split), slice(split, n)
        ph_train, y_train = phase_scores[tr], y[tr]
        ph_test,  y_test  = phase_scores[te],  y[te]
        chi_train, chi_test = chis[tr], chis[te]
        if best_cal["mode"]=="platt":
            params = fit_platt(ph_train, y_train, chis=None)
            p_cal  = apply_platt(ph_test, [0]*len(y_test), params)
        elif best_cal["mode"]=="platt+chi":
            params = fit_platt(ph_train, y_train, chis=chi_train)
            p_cal  = apply_platt(ph_test, chi_test, params)
        else:
            iso_model = fit_isotonic(ph_train, y_train)
            p_cal  = apply_isotonic(ph_test, iso_model)
        plot_reliability(list(p_cal), list(y_test), rel_cal_png, bins=max(6, ece_bins))
    if roc_png: plot_roc(phase_scores, y, roc_png)

    # README headline
    if readme_md:
        b, p = summary["baseline"], summary["phase_aware"]; g = summary["gate_quality"]
        lines = [
          f"# 5FRE-benchpack (v{VERSION}) — Evaluation-only",
          "",
          "Phase-preserving semantics score with complex phase, χ coherence and Λ gate.",
          "",
          "## Headline",
          f"- Baseline cosine: acc **{b['best_acc']}**, ECE **{b['ece']}**, τ≈**{b['threshold']}**",
          f"- Phase-aware:     acc **{p['best_acc']}**, ECE **{p['ece']}**, τ≈**{p['threshold']}**, AUC **{p['auc']}**",
          f"- Λ gate: false-open **{g.get('false_open_rate',0)}**, missed-open **{g.get('missed_open_rate',0)}**",
          "",
          "## Run",
          "```bash",
          "pip install numpy",
          "python benchpack.py --summary results/summary.json --print",
          "# optional calibration & conformal",
          "python benchpack.py --cal auto --target-fpr 0.05,0.10 --summary results/summary.json",
          "```",
          ""
        ]
        os.makedirs(os.path.dirname(readme_md) or ".", exist_ok=True)
        with open(readme_md, "w", encoding="utf-8") as f: f.write("\n".join(lines))

    if write_eval_license:
        with open("EVALUATION_LICENSE.txt","w",encoding="utf-8") as f:
            f.write(EVAL_LICENSE_TEXT.strip()+"\n")

    # Zip artifacts
    if zip_artifacts:
        os.makedirs(os.path.dirname(zip_artifacts) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_artifacts, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in [out_jsonl, summary_json, csv_out, readme_md, rel_raw_png, rel_cal_png, roc_png]:
                if p and os.path.exists(p): z.write(p)
            if os.path.exists("EVALUATION_LICENSE.txt"): z.write("EVALUATION_LICENSE.txt")

    return summary

EVAL_LICENSE_TEXT = """
Evaluation-Only License (Non-Commercial)
----------------------------------------
You are granted a non-transferable, non-exclusive right to evaluate this PoC
internally. No redistribution, sublicensing, modification for production, or
derivative works without prior written permission from the authors.
Copyright (c) 2025 Steven Ray Britt and contributors. All rights reserved.
"""

# =============================================================================
# CLI / UTIL
# =============================================================================

def parse_fpr_list(s: str) -> List[float]:
    if not s: return []
    out=[]
    for tok in s.split(","):
        tok=tok.strip()
        if not tok: continue
        try: out.append(float(tok))
        except Exception: pass
    return out

def main(argv=None):
    p = argparse.ArgumentParser(description="5FRE-benchpack PoC — phase-preserving demo (evaluation-only).")
    p.add_argument("--save", default="results/metrics.jsonl", help="JSONL output path")
    p.add_argument("--summary", default="results/summary.json", help="Summary JSON path")
    p.add_argument("--csv", default="results/metrics_expanded.csv", help="Per-record CSV export")
    p.add_argument("--pairs", default="", help="JSON file with list of [a,b,label,(tag)] or {a,b,label,tag}")
    p.add_argument("--pairs-append", default="", help="JSON file to append to the loaded pairs")
    p.add_argument("--print", action="store_true", help="Print each record to stdout")
    p.add_argument("--signed", action="store_true", help="Use signed phase score (can be negative)")
    p.add_argument("--seed", type=int, default=1337, help="Random seed (reproducible augmentation/CI)")
    p.add_argument("--cal", default="none", choices=["none","platt","platt+chi","iso","auto"],
                   help="Post-hoc calibration mode")
    p.add_argument("--split", type=float, default=0.6, help="Train fraction for calibration/conformal (0..1)")
    p.add_argument("--ece-bins", type=int, default=6, help="Bins for ECE")
    p.add_argument("--target-fpr", default="", help="Comma-separated list, e.g. 0.01,0.05,0.10")
    p.add_argument("--readme", default="README.md", help="Auto-generate README.md headline")
    p.add_argument("--plot-cal-raw", default="results/reliability_phase_raw.png", help="Reliability (raw phase)")
    p.add_argument("--plot-cal-best", default="results/reliability_phase_cal.png", help="Reliability (best calibrated)")
    p.add_argument("--plot-roc", default="results/roc_phase.png", help="ROC curve (phase score)")
    p.add_argument("--zip-artifacts", default="", help="Zip file path to bundle results ('' disables)")
    p.add_argument("--write-sample", default="", help="Write an extended sample pairs JSON to this path and exit")
    p.add_argument("--auto-augment", type=int, default=0, help="If >0, synthesize this many pairs into --pairs and exit")
    p.add_argument("--write-eval-license", action="store_true", help="Write EVALUATION_LICENSE.txt and continue")
    # notebook-safe parsing
    args, _ = p.parse_known_args([] if "ipykernel" in sys.modules else None)

    if args.write_sample:
        os.makedirs(os.path.dirname(args.write_sample) or ".", exist_ok=True)
        write_json(EXTENDED_SAMPLE, args.write_sample)
        print(f"Wrote sample pairs -> {args.write_sample}")
        return 0

    # load pairs
    if args.pairs:
        pairs = load_pairs_labels(args.pairs)
    else:
        pairs = default_pairs_labels()

    if args.pairs_append:
        pairs += load_pairs_labels(args.pairs_append)

    # augmentation mode
    if args.auto_augment and args.pairs:
        aug = auto_augment(pairs, args.auto_augment, seed=args.seed)
        write_json([[a,b,l,t] for (a,b,l,t) in aug], args.pairs)  # overwrite for clarity
        print(f"Auto-augmented to N={len(aug)} -> {args.pairs}")
        return 0

    summary = evaluate(
        pairs_labels=pairs,
        out_jsonl=args.save, summary_json=args.summary, csv_out=args.csv,
        also_print=args.print, signed=args.signed, seed=args.seed,
        cal_mode=args.cal, split_frac=args.split, ece_bins=args.ece_bins,
        target_fpr_list=parse_fpr_list(args.target_fpr),
        readme_md=args.readme,
        rel_raw_png=args.plot_cal_raw, rel_cal_png=args.plot_cal_best, roc_png=args.plot_roc,
        zip_artifacts=args.zip_artifacts, write_eval_license=args.write_eval_license
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote records -> {args.save}")
    print(f"Summary -> {args.summary}")
    print(f"CSV -> {args.csv}")
    if args.readme: print(f"README -> {args.readme}")
    if args.plot_cal_raw: print(f"Reliability (raw) -> {args.plot_cal_raw}")
    if args.plot_cal_best: print(f"Reliability (best cal) -> {args.plot_cal_best}")
    if args.plot_roc: print(f"ROC -> {args.plot_roc}")
    if args.zip_artifacts: print(f"Artifacts ZIP -> {args.zip_artifacts}")
    print("Done.")
    return 0

def _in_notebook():
    return any(m in sys.modules for m in ("ipykernel", "google.colab"))

if __name__ == "__main__":
    code = main()
    if not _in_notebook():
        sys.exit(code)
