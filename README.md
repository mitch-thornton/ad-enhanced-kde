# Enhanced Kernel Density Estimation using Algebraic Diversity

Accompanying repository for a manuscript currently under review. Sole author: Mitchell A. Thornton.

The repository contains the estimator implementation (AD-KDE: a Gaussian-mixture base peeled
from the data, a floor-stripped Wiener stage on the residual, and their superposition), a
corrected improved-Sheather-Jones bandwidth (the widely used KDEpy implementation undersmooths
about 2.6x versus the Botev fixed point; the corrected fixed point is embedded directly in
`ad_kde_v31.py`), and every script, seed, and results file needed to regenerate the figures and
tables of the manuscript. The methods build on the estimator construction described in
arXiv:2606.15450.

Reading bandwidth selection as a spectral-support problem: the cyclic group-averaged covariance of the
binned data has the squared empirical characteristic function as its spectrum, the bandwidth is the
cutoff where it meets the 1/n sampling-noise floor. Two enhancements follow: an automatic bandwidth
selector, and an adaptive per-frequency Wiener estimator that generalizes the fixed kernel. A
single AD-KDE estimator, a scale decomposition of a smooth mixture base and a band-limited residual
with no mixing weight, collects the strengths of both stages and is the paper's headline object.
Evaluated on the Marron-Wand benchmark across seven sample sizes, n = 50 to 5000, with an honest,
continuously traced sample-size crossover
and on four real datasets (CRSP returns, CMS dimuon and SDSS galaxy-redshift spectra, UNSW-NB15 network
traffic).

## Build
    ./build.sh           # regenerate self-contained figures, then compile
    ./build.sh --check    # also verify page count and reference resolution

## Contents
- `figures/`, `scripts/` - figures and the code that produces them
- `DATA.md`, `the release notes` - data provenance/reproduction and the provenance of this pull

## Provenance
Pulled from the author's "Enhanced Kernel Density Estimation using Algebraic Diversity" master
manuscript, scoped to the core spectral method (selector, AD-Wiener, superposition), the benchmark,
and the real-data validations. The heaped/rounded-data treatment is a separate companion paper; the
multivariate covariance, bounded-support, deconvolution, and synthetic-generation material is left to
the master and the author's other manuscripts. 

## Reproduction map

| Result | Script | Command |
|---|---|---|
| Marron-Wand benchmark anchor tables and win counts | `scripts/exp_benchmark_v28.py` | `python3 exp_benchmark_v28.py` |
| Rank-sweep table and figure (seven sample sizes) | `scripts/make_ranksweep_v2.py` | `python3 make_ranksweep_v2.py` |
| Consistency and rate-verification table | `scripts/exp_consistency_v1.py` | `python3 exp_consistency_v1.py` |
| Bandwidth-selector comparison | `scripts/ad_bw_core_v1.py` | `python3 ad_bw_core_v1.py` |
| CRSP daily-returns study (needs a CRSP subscription) | `scripts/wrds_extract_crsp.py`, then `scripts/analyze_crsp_adkde.py`, then `scripts/make_crsp_figs.py` | run in that order |
| CMS dimuon study (public CERN Open Data) | `scripts/exp_cern_dimuon_v17.py` | `python3 exp_cern_dimuon_v17.py` |
| SDSS redshift study (public) | `scripts/exp_sdss_redshift_v16.py` | `python3 exp_sdss_redshift_v16.py` |
| UNSW-NB15 network-security table (public) | `scripts/exp_cyber_unsw_v28.py` | `python3 exp_cyber_unsw_v28.py` |
| Health-claims frequency study | `scripts/exp_financial_v9.py` | `python3 exp_financial_v9.py` |
| Adjacent-paradigm probes (local likelihood, adaptive diffusion, neural spline flow) | `scripts/exp_locfit_probe_v2.R`, `scripts/exp_modern_probe_v2.py`, `scripts/exp_nf_probe_v2.py` | see script headers |

All synthetic results use fixed recorded seeds and run with no external data; the four real-data
studies download or read public data except CRSP, which reruns on any subscription.
