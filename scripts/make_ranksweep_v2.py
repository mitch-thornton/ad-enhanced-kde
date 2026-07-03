#!/usr/bin/env python3
"""make_ranksweep_v2.py: assemble fig_ranksweep.pdf and the tab:ranksweep rows.

Inputs
  results/exp_benchmark_v28_n{N}.json (or the merged exp_benchmark_v28_sweep.json)
  for the non-anchor sizes N in {50, 200, 1000, 2000}, produced by exp_benchmark_v28.py,
  plus the anchor per-density matrices (n = 100, 500, 5000) embedded below, which are
  exactly the printed bodies of the paper's Tables (carried same-draw columns with the
  corrected fixed-point ISJ column, at reported precision).

Convention
  Ranks at every n use average-tie ranking of the REPORTED-PRECISION per-density means
  (two decimals below 100, one at or above), so the figure, the anchor tables, and the
  tab:ranksweep rows are mutually recomputable by a reader.

Output
  figures/fig_ranksweep.pdf and, on stdout, the LaTeX rows for tab:ranksweep.
"""
import json, os, glob
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(HERE, "..", "figures")

COLS = ["Silverman", "ISJ/Botev", "LSCV", "Abramson", "GMM-BIC", "AD-Wiener", "AD-KDE"]
KEYS = ["Silverman", "ISJ/Botev", "LSCV", "Abramson", "GMM-BIC", "AD-Wiener", "superpose"]
DJ = ["1 Gaussian", "2 Skewed unimodal", "3 Strongly skewed", "4 Kurtotic unimodal",
      "5 Outlier", "6 Bimodal", "7 Separated bimodal", "8 Skewed bimodal", "9 Trimodal",
      "10 Claw", "11 Double claw", "12 Asymmetric claw", "13 Asym double claw",
      "14 Smooth comb", "15 Discrete comb"]

# Anchor matrices: printed bodies of Tables tab:bench100 / tab:bench500 / tab:benchmark.
ANCHOR = {
 100: [[5.83,6.06,10.53,7.24,2.64,9.98,2.64],[9.05,9.76,13.38,9.57,12.18,11.15,13.3],
       [142.2,53.21,48.64,112.5,87.07,53.24,53.45],[101.7,55.95,52.4,58.83,79.23,55.8,56.22],
       [61.34,66.06,85.01,71,45.32,56.71,31.9],[8.73,9.14,11.02,8.68,16.96,12.46,17.54],
       [46.74,12.87,14.94,40.26,9.12,16.15,16.34],[12.57,12.54,12.99,12.75,22.89,13.61,21.21],
       [11.16,10.67,12.69,11.33,16.47,13.59,16.11],[53.17,49,42.47,54.59,57.88,45.19,55.26],
       [10.21,10.51,12.1,10.35,20.1,13.56,18.7],[27.81,27.26,28.18,28.39,33.5,26.3,31.43],
       [14.42,14.22,17.54,14.07,21.44,17.68,21.92],[85.06,39.76,39.1,79.85,45.18,41.65,41.91],
       [113.5,38.36,39.15,113.5,39.55,40.69,40.55]],
 500: [[1.86,1.88,2.54,2.24,0.48,2.37,0.48],[2.85,2.96,4.14,2.6,2.33,2.74,1.86],
       [106.3,16.38,15.69,73.7,42.54,16.71,16.59],[56.23,13.59,13.96,16.31,8.28,12.47,12.61],
       [19.62,20,23.53,21.54,6.17,15.45,6.15],[3.01,2.52,3.16,2.19,1.61,3.16,1.84],
       [20.86,3.55,3.97,13.02,1.66,3.69,3.75],[5.2,3.86,4.09,4.03,7.76,3.73,3.86],
       [4.81,3.36,3.62,3.71,5.42,3.96,5.67],[45.95,12.72,12.49,46.03,49.2,14.21,42.47],
       [4.43,3.95,4.32,3.6,3.07,4.52,3.32],[20.82,11.51,11.03,20.12,20.49,11.7,19.71],
       [7.87,6.94,7.28,6.76,6.08,7.42,6.23],[63.54,17.54,16.4,58.54,20.36,17.43,17.44],
       [88.28,18.96,16.06,83.11,22.44,16.43,16.2]],
 5000: [[0.31,0.32,0.47,0.37,0.06,0.28,0.06],[0.48,0.48,0.72,0.51,0.46,0.37,0.24],
       [58.9,2.59,3.96,30.74,9.42,2.32,2.32],[18.41,2.25,3.04,1.88,2.12,1.63,1.63],
       [2.89,2.91,4.17,3.28,0.62,1.55,0.62],[0.6,0.43,0.61,0.38,0.15,0.38,0.16],
       [5.07,0.59,0.83,1.49,0.15,0.46,0.15],[1.18,0.56,0.82,0.49,1.29,0.44,0.18],
       [1.35,0.57,0.89,0.75,1.28,0.53,1.29],[32.93,1.98,2.69,26.42,2.35,1.5,1.5],
       [2.09,1.85,2.06,1.85,1.66,1.85,1.66],[12.51,2.51,3.49,10.73,7.76,2.57,6.17],
       [4.74,2.01,2.77,4.23,4.58,2.32,4.59],[40.79,3.89,7.57,37.55,11.24,3.8,3.78],
       [47.65,2.74,7.46,39.19,14.92,2.43,2.42]]
}



def rounded(v):
    return round(v, 2) if v < 100 else round(v, 1)


def load_nonanchor(n):
    for src in (os.path.join(RESULTS, "exp_benchmark_v28_n%d.json" % n),
                os.path.join(RESULTS, "exp_benchmark_v28_sweep.json")):
        if not os.path.exists(src):
            continue
        d = json.load(open(src))["mean_ise_x1e3"]
        try:
            return [[rounded(d["%s|n=%d" % (dj, n)][k]) for k in KEYS] for dj in DJ]
        except KeyError:
            continue
    raise FileNotFoundError("no results for n=%d" % n)


def main():
    pts, missing = {}, []
    for n in (50, 200, 1000, 2000):
        try:
            pts[n] = np.array(load_nonanchor(n), dtype=float)
        except (FileNotFoundError, KeyError):
            missing.append(n)
    for n, M in ANCHOR.items():
        pts[n] = np.array(M, dtype=float)
    if missing:
        print("WARNING: missing sweep results for n in %s; run exp_benchmark_v28.py" % missing)

    ns = sorted(pts)
    ranks = {n: rankdata(pts[n], axis=1, method="average").mean(axis=0) for n in ns}

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    styles = [("0.65", "-", "o"), ("0.35", "--", "s"), ("0.5", ":", "^"),
              ("0.55", "-.", "v"), ("0.2", "--", "D"), ("0.15", "-", "P"), ("0.0", "-", "*")]
    for j, c in enumerate(COLS):
        col, ls, mk = styles[j]
        ax.plot(ns, [ranks[n][j] for n in ns], ls, color=col, marker=mk, ms=5,
                lw=1.8 if c == "AD-KDE" else 1.1, label=c)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("$n$")
    ax.set_ylabel("average rank (lower is better)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_ylim(1, 7.5)
    plt.tight_layout()
    out = os.path.join(FIGS, "fig_ranksweep.pdf")
    plt.savefig(out)
    print("figure written: fig_ranksweep.pdf")

    print("%% tab:ranksweep rows (intermediate sizes)")
    for n in (50, 200, 1000, 2000):
        if n not in pts or n in missing:
            continue
        r = ranks[n]
        b = int(np.argmin(r))
        cells = ["\\textbf{%.2f}" % v if j == b else "%.2f" % v for j, v in enumerate(r)]
        print("%d & %s \\\\" % (n, " & ".join(cells)))


if __name__ == "__main__":
    main()
