#!/usr/bin/env python3
"""Marron-Wand benchmark v28 (journal sweep): AD-Wiener / superposition vs strong baselines, exact ISE.

v27 changes versus v26 (the tables of paper v30 were produced by the external
mw_benchmark package; this script now reproduces them faithfully from the bundle):

1. ISJ/Botev bandwidth is a direct implementation of the Botev-Grotowski-Kroese
   (2010) fixed point, embedded below. The KDEpy improved_sheather_jones used by
   v26 (and by the mw_benchmark runs behind the v30 tables) was found to
   undersmooth systematically: h ~ 0.075 on a standard normal at n=5000 where the
   fixed point gives ~0.198 and the AMISE optimum is 0.193. The v30 ISJ column is
   therefore superseded by this script's output. KDEpy is no longer a dependency.
2. True least-squares cross-validation (closed-form Gaussian-kernel LSCV
   criterion), full sample at n=100 and n=500, 1000-point subsample at n=5000,
   matching the paper text. v26 carried a likelihood-CV shortcut (sklearn
   GridSearchCV on KernelDensity, 800-point subsample) that did not produce the
   published column.
3. Per-rep draws are unchanged: rng = default_rng(1000*r + n), the same stream as
   v26 and as the published tables (verified: Silverman / GMM-BIC / AD-Wiener
   reproduce the printed values exactly). Columns regenerated here are therefore
   computed on the identical draws as the columns carried from v30.
4. Results are written to ../results/exp_benchmark_v27.json and a SUMMARY block
   is printed for paste-back.
Usage:
  python3 exp_benchmark_v27.py --quick                      # smoke test
  python3 exp_benchmark_v27.py --ests "ISJ/Botev,LSCV"      # regenerate columns
  python3 exp_benchmark_v27.py                              # all 7 bundled columns
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v31 as K
import exp_datagen_v30 as E
import adkde_plugins as P
from scipy.fftpack import dct as _dct
from scipy import optimize as _opt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ---------------- Marron-Wand 15 densities as (weight, mean, sd) lists ----------------
def _mw():
    d = {}
    d["1 Gaussian"] = [(1, 0, 1)]
    d["2 Skewed unimodal"] = [(1/5,0,1),(1/5,1/2,2/3),(3/5,13/12,5/9)]
    d["3 Strongly skewed"] = [(1/8, 3*((2/3)**l - 1), (2/3)**l) for l in range(8)]
    d["4 Kurtotic unimodal"] = [(2/3,0,1),(1/3,0,1/10)]
    d["5 Outlier"] = [(1/10,0,1),(9/10,0,1/10)]
    d["6 Bimodal"] = [(1/2,-1,2/3),(1/2,1,2/3)]
    d["7 Separated bimodal"] = [(1/2,-3/2,1/2),(1/2,3/2,1/2)]
    d["8 Skewed bimodal"] = [(3/4,0,1),(1/4,3/2,1/3)]
    d["9 Trimodal"] = [(9/20,-6/5,3/5),(9/20,6/5,3/5),(1/10,0,1/4)]
    d["10 Claw"] = [(1/2,0,1)] + [(1/10, l/2 - 1, 1/10) for l in range(5)]
    d["11 Double claw"] = [(49/100,-1,2/3),(49/100,1,2/3)] + [(1/350,(l-3)/2,1/100) for l in range(7)]
    d["12 Asymmetric claw"] = [(1/2,0,1)] + [(2**(1-l)/31, l+1/2, 2**(-l)/10) for l in range(-2,3)]
    d["13 Asym double claw"] = [(46/100,2*l-1,2/3) for l in range(2)] + \
        [(1/300,-l/2,1/100) for l in range(1,4)] + [(7/300,l/2,7/100) for l in range(1,4)]
    d["14 Smooth comb"] = [(2**(5-l)/63, (65-96*(1/2)**l)/21, (32/63)/2**l) for l in range(6)]
    d["15 Discrete comb"] = [(2/7,(12*l-15)/7,2/7) for l in range(3)] + [(1/21,2*l/7,1/21) for l in range(8,11)]
    return d

MW = _mw()

def true_pdf(comps, x):
    return sum(w * np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi)) for w,m,s in comps)

def sample(comps, n, rng):
    w = np.array([c[0] for c in comps]); w = w/w.sum()
    idx = rng.choice(len(comps), size=n, p=w)
    return np.array([rng.normal(comps[i][1], comps[i][2]) for i in idx])

def grid_for(comps):
    lo = min(m-6*s for _,m,s in comps); hi = max(m+6*s for _,m,s in comps)
    return np.linspace(lo, hi, 2048)

# ---------------- estimators: data -> density on xg ----------------
def _nrm(f, xg):
    f = np.clip(f, 0, None); z = np.trapezoid(f, xg); return f/z if z>0 else f

def kde_fixed_bw(d, xg, h):
    return _nrm(np.mean(np.exp(-0.5*((xg[:,None]-d[None,:])/h)**2)/(h*np.sqrt(2*np.pi)), axis=1), xg)

def h_silverman(d):
    """Exact replication of the KDEpy silvermans_rule used for the published
    Silverman and Abramson columns: min(sd, IQR/1.3489795) * (3n/4)^(-1/5)."""
    n = len(d)
    if n == 1: return 1.0
    sigma = np.std(d, ddof=1)
    iqr = (np.percentile(d, 75) - np.percentile(d, 25)) / 1.3489795003921634
    sigma = min(sigma, iqr) if iqr > 0 else sigma
    return sigma * (n * 3 / 4.0) ** (-1 / 5)

def est_silverman(d, xg):
    return kde_fixed_bw(d, xg, max(h_silverman(d), 1e-3))

# ---- ISJ/Botev: direct Botev-Grotowski-Kroese (2010) fixed point ----
def botev_isj_bandwidth(data, n_grid=2**14):
    d = np.asarray(data, float)
    N = len(np.unique(d))
    lo, hi = d.min(), d.max(); R = hi - lo
    if not np.isfinite(R) or R <= 0 or N < 4: return h_silverman(d)
    lo -= R/10.0; hi += R/10.0; R = hi - lo
    hist, _ = np.histogram(d, bins=n_grid, range=(lo, hi))
    a = _dct(hist/len(d), norm=None)
    I = np.arange(1, n_grid, dtype=float)**2
    a2 = (a[1:]/2.0)**2
    def _fp(t):
        l = 7
        f = 2*np.pi**(2*l)*np.sum(I**l*a2*np.exp(-I*np.pi**2*t))
        for s in range(l-1, 1, -1):
            K0 = np.prod(np.arange(1, 2*s, 2))/np.sqrt(2*np.pi)
            c = (1 + 0.5**(s+0.5))/3.0
            tj = (2*c*K0/(N*f))**(2.0/(3+2*s))
            f = 2*np.pi**(2*s)*np.sum(I**s*a2*np.exp(-I*np.pi**2*tj))
        return t - (2*N*np.sqrt(np.pi)*f)**(-2.0/5)
    ts = np.linspace(1e-8, 0.1, 200)
    vals = np.array([_fp(t) for t in ts])
    ok = np.isfinite(vals)
    sc = np.where(ok[:-1] & ok[1:] & (np.sign(vals[:-1]) != np.sign(vals[1:])))[0]
    if len(sc) == 0: return h_silverman(d)
    t_star = _opt.brentq(_fp, ts[sc[0]], ts[sc[0]+1])
    return float(np.sqrt(t_star)*R)

def est_isj(d, xg):
    try: h = botev_isj_bandwidth(d)
    except Exception: h = h_silverman(d)
    return kde_fixed_bw(d, xg, max(h, 1e-3))

# ---- true least-squares cross-validation (Gaussian kernel, closed form) ----
def lscv_bandwidth(d, hs):
    """LSCV(h) = int fhat_h^2 - (2/n) sum_i fhat_{h,-i}(x_i), Gaussian kernel,
    computed in closed form from pairwise differences. Returns argmin over hs."""
    n = len(d)
    diff = d[:,None] - d[None,:]
    best, bh = np.inf, hs[0]
    sq = diff**2
    for h in hs:
        s2 = np.sqrt(2.0)*h
        term1 = np.sum(np.exp(-0.5*sq/s2**2))/(s2*np.sqrt(2*np.pi))/n**2
        Ksum = np.sum(np.exp(-0.5*sq/h**2))/(h*np.sqrt(2*np.pi))
        Kdiag = n/(h*np.sqrt(2*np.pi))
        term2 = 2.0*(Ksum - Kdiag)/(n*(n-1))
        cv = term1 - term2
        if cv < best: best, bh = cv, h
    return bh

def est_lscv(d, xg):
    sub = d if len(d) <= 500 else d[np.random.default_rng(0).choice(len(d), 1000, replace=False)] \
        if len(d) > 1000 else d
    hs = np.geomspace(0.02, 1.2, 40)*sub.std()
    h = lscv_bandwidth(sub, hs)
    return kde_fixed_bw(d, xg, max(h, 1e-3))

def est_abramson(d, xg):
    h0 = max(h_silverman(d), 1e-3)
    pilot = np.clip(np.mean(np.exp(-0.5*((d[:,None]-d[None,:])/h0)**2)/(h0*np.sqrt(2*np.pi)), axis=1), 1e-12, None)
    g = np.exp(np.mean(np.log(pilot))); hi = h0*np.sqrt(g/pilot)
    f = np.mean(np.exp(-0.5*((xg[:,None]-d[None,:])/hi[None,:])**2)/(hi[None,:]*np.sqrt(2*np.pi)), axis=1)
    return _nrm(f, xg)

def est_gmm(d, xg):
    from sklearn.mixture import GaussianMixture
    best = None
    for k in range(1, 11):
        gm = GaussianMixture(k, covariance_type="full", reg_covar=1e-5, max_iter=100,
                             random_state=0).fit(d.reshape(-1,1))
        b = gm.bic(d.reshape(-1,1))
        if best is None or b < best[1]: best = (gm, b)
    f = np.exp(best[0].score_samples(xg.reshape(-1,1)))
    return _nrm(f, xg)

def est_adwiener(d, xg):
    return _nrm(K.ad_wiener(d, xg, strip="residue"), xg)

def est_superpose(d, xg):
    return _nrm(E.superpose(d, xg)[0], xg)

ESTS = {"Silverman":est_silverman, "ISJ/Botev":est_isj, "LSCV":est_lscv, "Abramson":est_abramson,
        "GMM-BIC":est_gmm, "AD-Wiener":est_adwiener, "superpose":est_superpose}

def ise(fhat, ftrue, xg):
    return np.trapezoid((fhat-ftrue)**2, xg)

def run(densities, ns, reps, ests):
    out = {}
    t0 = time.time(); total = len(densities)*len(ns)*reps; done = 0
    for dname in densities:
        comps = MW[dname]; xg = grid_for(comps); ft = true_pdf(comps, xg)
        for n in ns:
            key = (dname, n); out[key] = {e: [] for e in ests}
            for r in range(reps):
                rng = np.random.default_rng(1000*r + n)   # SAME stream as v26 / published tables
                d = sample(comps, n, rng); d = np.clip(d, xg[0], xg[-1])
                for ename, efn in ests.items():
                    try: out[key][ename].append(float(ise(efn(d, xg), ft, xg)))
                    except Exception: out[key][ename].append(float("nan"))
                done += 1
                if done % 150 == 0:
                    el = time.time()-t0
                    print("  heartbeat: %d/%d reps, %.0fs elapsed, ~%.0fs remaining"
                          % (done, total, el, el/done*(total-done)), flush=True)
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--ns", default="100,500,5000")
    ap.add_argument("--ests", default="")   # comma list; empty = the 7 bundled estimators
    ap.add_argument("--lo", type=int, default=0)   # density slice [lo:hi] for chunked runs
    ap.add_argument("--hi", type=int, default=15)  # (per-rep seeds are density-independent,
    ap.add_argument("--merge", action="store_true")# so chunking does not change the draws)
    a = ap.parse_args()

    if a.merge:
        import glob
        merged = {"script": "exp_benchmark_v28.py", "seed_rule": "default_rng(1000*r+n)",
                  "mean_ise_x1e3": {}, "per_rep_ise": {}}
        for fp in sorted(glob.glob(os.path.join(RESULTS, "exp_benchmark_v28_*chunk*.json"))):
            with open(fp) as f: c = json.load(f)
            merged["mean_ise_x1e3"].update(c["mean_ise_x1e3"])
            merged["per_rep_ise"].update(c["per_rep_ise"])
            merged["reps"] = c["reps"]; merged["ns"] = c["ns"]; merged["ests"] = c["ests"]
        outp = os.path.join(RESULTS, "exp_benchmark_v28_sweep.json")
        with open(outp, "w") as f: json.dump(merged, f, indent=1)
        print("merged %d chunk files -> %s (%d density x n cells)"
              % (len(glob.glob(os.path.join(RESULTS, "exp_benchmark_v28_*chunk*.json"))),
                 outp, len(merged["mean_ise_x1e3"])))
        sys.exit(0)

    if a.ests:
        ests = {k: ESTS[k] for k in a.ests.split(",")}
    else:
        ests = ESTS
    ns = [int(x) for x in a.ns.split(",")]

    if a.quick:
        t0 = time.time()
        res = run(["1 Gaussian","10 Claw","15 Discrete comb"], [500], 5, ests)
        print("quick test %.1fs" % (time.time()-t0))
        for (dn,n),r in res.items():
            print("%-20s n=%-5d " % (dn,n) + " ".join("%s=%.3f"%(e,1e3*np.nanmean(v)) for e,v in r.items()))
        sys.exit(0)

    names = list(MW.keys())[a.lo:a.hi]; t0 = time.time()
    res = run(names, ns, a.reps, ests)
    os.makedirs(RESULTS, exist_ok=True)
    payload = {"script": "exp_benchmark_v28.py", "seed_rule": "default_rng(1000*r+n)",
               "reps": a.reps, "ns": ns, "ests": list(ests),
               "mean_ise_x1e3": {"%s|n=%d"%(dn,n): {e: round(1e3*float(np.nanmean(v)),4)
                                 for e,v in r.items()} for (dn,n),r in res.items()},
               "per_rep_ise": {"%s|n=%d"%(dn,n): {e: v for e,v in r.items()}
                               for (dn,n),r in res.items()}}
    ntag = "n" + "-".join(str(x) for x in ns)
    tag = "_%s" % ntag if (a.lo, a.hi) == (0, 15) else "_%s_chunk%02d-%02d" % (ntag, a.lo, a.hi)
    outp = os.path.join(RESULTS, "exp_benchmark_v28%s.json" % tag)
    with open(outp, "w") as f: json.dump(payload, f, indent=1)
    print("ran %d densities x %s x %d reps in %.0fs -> %s"
          % (len(names), ns, a.reps, time.time()-t0, outp))

    print("\n===== SUMMARY (paste this back) =====")
    print("tool: exp_benchmark_v28.py  seed rule: default_rng(1000*r+n)  reps=%d" % a.reps)
    print("estimators: %s" % ", ".join(ests))
    for n in ns:
        print("--- n=%d (mean ISE x1e3) ---" % n)
        hdr = "%-22s" % "density" + "".join("%12s" % e for e in ests)
        print(hdr)
        for dn in names:
            row = res[(dn,n)]
            print("%-22s" % dn + "".join("%12.2f" % (1e3*np.nanmean(row[e])) for e in ests))
    print("===== END SUMMARY =====")
