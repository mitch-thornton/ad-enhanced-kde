#!/usr/bin/env python3
"""exp_consistency_v1.py: numerical probe for consistency of the residual stage
(AD-Wiener with the FIXED 1/n floor strip) ahead of any theorem.

Hypothesis to probe (analytic targets): for normal-mixture densities the
characteristic function decays exponentially, so the floor crossing T_n grows
like log n and the conjectured rate is MISE = O(log n / n). Diagnostic: the
compensated quantity n * ISE / log n should flatten to a constant as n grows.

Contrast target (polynomial-decay phi): the Laplace density, |phi(t)|^2 =
1/(1+t^2)^2 ~ t^{-4}, where the conjectured linear-filter rate is polynomial,
MISE ~ n^{-3/4} (T_n ~ n^{1/4}; T_n/n + tail energy beyond T_n both n^{-3/4}).
Diagnostic: log-log slope of ISE vs n near -0.75, clearly shallower than the
mixture targets.

Controls: the binning grid M is scaled generously and a doubling check at the
largest n guards against a discretization floor contaminating the large-n ISE.

Seed of record: 20260627 (rep r at size n uses default_rng(20260627 + 7919*r + n)).
Run: python3 exp_consistency_v1.py [--reps 25] [--quick]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ad_kde_v31 as K

SEED0 = 20260627

# ---------------- targets ----------------
MW = {
    "Gaussian":        [(1.0, 0.0, 1.0)],
    "Strongly skewed": [(1/8, 3*((2/3)**l - 1), (2/3)**l) for l in range(8)],
    "Claw":            [(0.5, 0.0, 1.0)] + [(0.1, l/2 - 1.0, 0.1) for l in range(5)],
}

def mw_pdf(comps, x):
    from math import sqrt, pi
    out = np.zeros_like(x)
    for w, m, s in comps:
        out += w * np.exp(-0.5*((x-m)/s)**2) / (s*np.sqrt(2*np.pi))
    return out

def mw_sample(comps, n, rng):
    ws = np.array([c[0] for c in comps]); ws /= ws.sum()
    idx = rng.choice(len(comps), size=n, p=ws)
    ms = np.array([c[1] for c in comps]); ss = np.array([c[2] for c in comps])
    return rng.normal(ms[idx], ss[idx])

def laplace_pdf(x):
    return 0.5*np.exp(-np.abs(x))

def laplace_sample(n, rng):
    return rng.laplace(0.0, 1.0, size=n)

TARGETS = {
    **{name: ("mw", comps) for name, comps in MW.items()},
    "Laplace": ("laplace", None),
}

# ---------------- estimator: the paper's residual stage, FIXED 1/n floor ----------------
def adwiener_fixed(d, xg_unused, M, span):
    """The bundle's own ad_wiener with strip="simple" (fixed 1/n floor + hard cutoff
    at the first floor crossing), evaluated on an M-point grid over span. Probing
    the shipped estimator, not a reimplementation."""
    xg = np.linspace(span[0], span[1], M)
    return xg, K.ad_wiener(d, xg, strip="simple")

def ise(fhat, ftrue, xg):
    return float(np.trapezoid((fhat - ftrue)**2, xg))

# ---------------- probe ----------------
def run(reps, ns, Ms, span=(-8.0, 8.0)):
    out = {"seed0": SEED0, "reps": reps, "ns": ns, "Ms": Ms, "ise": {}, "ise_Mx2": {}}
    for tname, (kind, comps) in TARGETS.items():
        for n, M in zip(ns, Ms):
            vals, vals2 = [], []
            for r in range(reps):
                rng = np.random.default_rng(SEED0 + 7919*r + n)
                d = mw_sample(comps, n, rng) if kind == "mw" else laplace_sample(n, rng)
                d = np.clip(d, span[0]+1e-9, span[1]-1e-9)
                xg, fh = adwiener_fixed(d, None, M, span)
                ftrue = mw_pdf(comps, xg) if kind == "mw" else laplace_pdf(xg)
                vals.append(ise(fh, ftrue, xg))
                if n == ns[-1]:
                    xg2, fh2 = adwiener_fixed(d, None, 2*M, span)
                    ft2 = mw_pdf(comps, xg2) if kind == "mw" else laplace_pdf(xg2)
                    vals2.append(ise(fh2, ft2, xg2))
            out["ise"]["%s|%d" % (tname, n)] = float(np.mean(vals))
            if vals2:
                out["ise_Mx2"][tname] = float(np.mean(vals2))
    return out

def report(out):
    ns = np.array(out["ns"], dtype=float)
    print("target                n-grid ISE (x1e3):")
    for tname in TARGETS:
        v = np.array([out["ise"]["%s|%d" % (tname, int(n))] for n in ns])
        print("%-18s %s" % (tname, " ".join("%.3f" % (1e3*x) for x in v)))
        # log-log slope over the last half of the grid
        h = len(ns)//2
        sl = np.polyfit(np.log(ns[h:]), np.log(v[h:]), 1)[0]
        print("   log-log slope (upper half): %.3f" % sl)
        for lbl, comp in (("n*ISE          ", v*ns),
                          ("n*ISE/sqrt(log)", v*ns/np.sqrt(np.log(ns))),
                          ("n*ISE/log n    ", v*ns/np.log(ns))):
            print("   %s: %s" % (lbl, " ".join("%.4f" % c for c in comp)))
        if tname in out["ise_Mx2"]:
            a, b = v[-1], out["ise_Mx2"][tname]
            print("   grid check at n=%d: ISE(M)=%.3e ISE(2M)=%.3e (ratio %.3f)"
                  % (int(ns[-1]), a, b, b/a))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        ns = [500, 2000, 8000]; Ms = [8192]*3; reps = 5
    else:
        ns = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
        Ms = [8192]*9   # fixed grid: rate measured at constant discretization
        reps = a.reps
    out = run(reps, ns, Ms)
    json.dump(out, open(os.path.join(HERE, "..", "results", "exp_consistency_v1.json"), "w"), indent=1)
    report(out)
    print("tool: exp_consistency_v1.py ; seed0=%d ; reps=%d" % (SEED0, reps))
