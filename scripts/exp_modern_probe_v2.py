#!/usr/bin/env python3
"""exp_modern_probe_v2.py -- probe-level comparison of AD-KDE against two modern
paradigms outside the bandwidth-selection lineage:
  (1) local likelihood density estimation (Loader 1996; local log-quadratic)
  (2) the full adaptive diffusion estimator (Botev-Grotowski-Kroese 2010, Sec. 5:
      pilot-weighted diffusion, not just the ISJ bandwidth)
Marron-Wand targets, ISE vs truth. v2: reps=15 (program standard), mean +/- sd, untuned local-likelihood arm
dropped (the author-run locfit probe is the fair local-likelihood evidence).
"""
import sys, numpy as np
sys.path.insert(0, '/home/claude/work/adkde/core_v2/ad_core_v3_0_bundle/scripts')
import ad_kde_v31 as K
import exp_consistency_v1 as C

SEED0 = 20260627
XG = np.linspace(-8, 8, 512)
DX = XG[1]-XG[0]

def ise(f, ftrue): return float(np.trapezoid((f-ftrue)**2, XG))

def norm(f):
    f = np.clip(f, 0, None); return f/np.trapezoid(f, XG)

def local_likelihood(d, xg, h):
    """Loader-style local log-quadratic likelihood density estimate.
    At each grid point x0, fit log f ~ a0 + a1 u + a2 u^2 by maximizing the
    local likelihood sum_i w_i logf(x_i) - n int w(u) f(u) du via Newton."""
    n = len(d); out = np.empty(len(xg))
    for j, x0 in enumerate(xg):
        u = (d - x0)/h
        w = np.exp(-0.5*u*u)          # Gaussian local weights at data
        # quadrature for the integral term on a local stencil
        vg = np.linspace(-4, 4, 41); wg = np.exp(-0.5*vg*vg)
        a = np.array([np.log(max(np.sum(w)/(n*h*np.sqrt(2*np.pi)), 1e-12)), 0.0, -0.5])
        for _ in range(25):
            lf = a[0] + a[1]*vg + a[2]*vg*vg
            lf = np.clip(lf, -60, 60)
            fv = np.exp(lf)
            # gradient: sum_i w_i psi(u_i) - n h int w psi f
            gi = np.vstack([np.ones_like(u), u, u*u])
            gq = np.vstack([np.ones_like(vg), vg, vg*vg])
            grad = gi @ w - n*h*np.trapezoid(wg*gq*fv, vg, axis=1)
            H = -(n*h)*np.trapezoid(wg*gq[:,None,:]*gq[None,:,:]*fv, vg, axis=2)
            try:
                step = np.linalg.solve(H - 1e-8*np.eye(3), grad)
            except np.linalg.LinAlgError:
                break
            a = a - np.clip(step, -2, 2)
            if np.max(np.abs(step)) < 1e-8: break
        out[j] = np.exp(np.clip(a[0], -60, 60))
    return norm(out)

def adaptive_diffusion(d, xg):
    """BGK 2010 full adaptive diffusion: dp/dt = 1/2 (p/pi)'' with pilot pi,
    run to the ISJ diffusion time; implicit Euler on the grid."""
    n = len(d)
    # initial histogram delta mixture on the grid
    idx = np.clip(np.searchsorted(xg, d), 1, len(xg)-1)
    p = np.zeros(len(xg)); np.add.at(p, idx, 1.0/(n*DX))
    # pilot: Silverman Gaussian KDE
    hs = K.h_silverman(d)
    pi = norm(np.mean(np.exp(-0.5*((xg[:,None]-d[None,:])/hs)**2)/(hs*np.sqrt(2*np.pi)), axis=1))
    pi = np.maximum(pi, 1e-6)
    tstar = K.h_isj(d)**2          # diffusion time = h^2 (up to convention)
    M = len(xg); steps = 40; dt = tstar/steps
    # implicit Euler: (I - dt/2 D2 diag(1/pi)) p_{k+1} = p_k, D2 = second-difference
    main = np.full(M, -2.0)/DX**2; off = np.full(M-1, 1.0)/DX**2
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    D2 = sp.diags([off, main, off], [-1,0,1], format='csc')
    A = sp.identity(M, format='csc') - (dt/2.0)*D2 @ sp.diags(1.0/pi)
    lu = spla.splu(A)
    for _ in range(steps):
        p = lu.solve(p)
    return norm(p)

TARGETS = ["Gaussian", "Strongly skewed", "Claw"]
print("%-16s %14s %14s %14s %14s" % ("target","ISJ","AdaptDiff","AD-Wiener","AD-KDE"))
for tname in TARGETS:
    comps = C.MW[tname]; ftrue = C.mw_pdf(comps, XG)
    acc = {k: [] for k in ("isj","ad","adw","adkde")}
    for r in range(15):
        rng = np.random.default_rng(SEED0 + 7919*r + 1000)
        d = np.clip(C.mw_sample(comps, 1000, rng), -8+1e-9, 8-1e-9)
        hI = K.h_isj(d)
        fisj = norm(np.mean(np.exp(-0.5*((XG[:,None]-d[None,:])/hI)**2)/(hI*np.sqrt(2*np.pi)), axis=1))
        acc["isj"].append(ise(fisj, ftrue))
        acc["ad"].append(ise(adaptive_diffusion(d, XG), ftrue))
        acc["adw"].append(ise(norm(K.ad_wiener(d, XG, strip="residue")), ftrue))
        sys.path.insert(0, '/home/claude/work/heaping/heaping_spl_v2_4/scripts'); import exp_datagen_v30 as E
        acc["adkde"].append(ise(norm(E.superpose(d, XG)[0]), ftrue))
    def ms(k): return "%.3f+-%.3f" % (1e3*np.mean(acc[k]), 1e3*np.std(acc[k], ddof=1))
    print("%-16s %14s %14s %14s %14s" % (tname, ms("isj"), ms("ad"), ms("adw"), ms("adkde")))
print("tool: exp_modern_probe_v2.py ; ISE x1e3 ; n=1000 reps=15 seed0=%d" % SEED0)
