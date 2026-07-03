#!/usr/bin/env python3
"""exp_nf_probe_v2.py -- FOR LOCAL RUN (needs torch + zuko: pip install torch zuko).
Neural-flow probe: a 1D neural spline flow (NSF) trained by maximum likelihood on
Marron-Wand samples, ISE against the true density, compared with the corrected-ISJ
Gaussian KDE and (if the ad_kde module is on the path) AD-Wiener and AD-KDE.
Probe scope: 3 targets, n in {1000, 2000}, reps=5, seed lineage 20260627.

Run:  python3 exp_nf_probe_v2.py            (from the journal bundle's scripts/)
"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch, zuko
import ad_kde_v31 as K

SEED0 = 20260627
XG = np.linspace(-8, 8, 512)

MW = {
    "Gaussian":        [(1.0, 0.0, 1.0)],
    "Strongly skewed": [(1/8, 3*((2/3)**l - 1), (2/3)**l) for l in range(8)],
    "Claw":            [(0.5, 0.0, 1.0)] + [(0.1, l/2 - 1.0, 0.1) for l in range(5)],
}
def mw_pdf(comps, x):
    out = np.zeros_like(x)
    for w, m, s in comps:
        out += w*np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))
    return out
def mw_sample(comps, n, rng):
    ws = np.array([c[0] for c in comps]); ws /= ws.sum()
    idx = rng.choice(len(comps), size=n, p=ws)
    ms = np.array([c[1] for c in comps]); ss = np.array([c[2] for c in comps])
    return rng.normal(ms[idx], ss[idx])
def ise(f, ftrue): return float(np.trapezoid((f-ftrue)**2, XG))
def norm(f):
    f = np.clip(f, 0, None); return f/np.trapezoid(f, XG)

def nsf_density(d, seed):
    torch.manual_seed(seed)
    x = torch.tensor(d, dtype=torch.float32).unsqueeze(-1)
    mu, sd = x.mean(), x.std()
    xs = (x - mu)/sd
    # bound=8 so the spline support covers the standardized evaluation grid;
    # outside the spline bound zuko's transform is identity and log_prob values
    # there are not meaningful density estimates, which corrupted probe v1.
    try:
        flow = zuko.flows.NSF(features=1, transforms=3,
                              hidden_features=(64, 64), bound=8.0)
    except TypeError:
        flow = zuko.flows.NSF(features=1, transforms=3, hidden_features=(64, 64))
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    ntr = int(0.9*len(xs))
    perm = torch.randperm(len(xs)); tr, va = xs[perm[:ntr]], xs[perm[ntr:]]
    best, best_state, bad = np.inf, None, 0
    for epoch in range(400):
        opt.zero_grad()
        loss = -flow().log_prob(tr).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            vl = float(-flow().log_prob(va).mean())
        if vl < best - 1e-4: best, bad = vl, 0; best_state = [p.detach().clone() for p in flow.parameters()]
        else:
            bad += 1
            if bad > 40: break
    if best_state is not None:
        with torch.no_grad():
            for p, b in zip(flow.parameters(), best_state): p.copy_(b)
    with torch.no_grad():
        xg = torch.tensor((XG - float(mu))/float(sd), dtype=torch.float32).unsqueeze(-1)
        lp = (flow().log_prob(xg) - np.log(float(sd))).numpy()
        lp = np.nan_to_num(lp, nan=-1e9, posinf=-1e9, neginf=-1e9)
        lp = np.clip(lp, -1e9, 10.0)          # densities above e^10 are junk
        f = np.exp(lp)
        f[~np.isfinite(f)] = 0.0
        return norm(f), float(best)           # also return best val NLL

def gkde(d, h):
    return norm(np.mean(np.exp(-0.5*((XG[:,None]-d[None,:])/h)**2)/(h*np.sqrt(2*np.pi)), axis=1))

print("%-16s %6s %10s %10s %10s %10s" % ("target","n","ISJ","NSF-flow","AD-Wiener","AD-KDE"))
try:
    import exp_datagen_v30 as E; HAVE_SUPER = True
except Exception:
    HAVE_SUPER = False
for tname, comps in MW.items():
    ftrue = mw_pdf(comps, XG)
    for n in (1000, 2000):
        acc = {k: [] for k in ("isj","nsf","adw","adkde")}
        for r in range(5):
            rng = np.random.default_rng(SEED0 + 7919*r + n)
            d = np.clip(mw_sample(comps, n, rng), -8+1e-9, 8-1e-9)
            acc["isj"].append(ise(gkde(d, K.h_isj(d)), ftrue))
            fnsf, vnll = nsf_density(d, SEED0+r)
            acc["nsf"].append(ise(fnsf, ftrue))
            acc.setdefault("vnll", []).append(vnll)
            acc["adw"].append(ise(norm(K.ad_wiener(d, XG, strip="residue")), ftrue))
            acc["adkde"].append(ise(norm(E.superpose(d, XG)[0]), ftrue) if HAVE_SUPER else np.nan)
        print("%-16s %6d %10.4f %10.4f %10.4f %10.4f   val-NLL %.3f" % (tname, n,
            1e3*np.mean(acc["isj"]), 1e3*np.mean(acc["nsf"]),
            1e3*np.mean(acc["adw"]), 1e3*np.mean(acc["adkde"]),
            float(np.mean(acc["vnll"]))))
        # sanity: a trained flow on the standardized Gaussian should show
        # val-NLL near 1.419 (standard-normal entropy); values far above mean
        # the flow output is pathological and the ISE column untrustworthy.
print("tool: exp_nf_probe_v2.py ; ISE x1e3 ; reps=5 seed0=%d" % SEED0)
