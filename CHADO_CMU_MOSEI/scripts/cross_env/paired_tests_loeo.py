import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def holm_bonferroni(pvals):
    """Return adjusted p-values (Holm-Bonferroni)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (m - i) * pvals[idx])
    # enforce monotonicity
    for i in range(1, m):
        adj[order[i]] = max(adj[order[i]], adj[order[i-1]])
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="outputs/cross_env/loeo_summary_modalities.csv")
    ap.add_argument("--metric", default="test_wf1", choices=["test_acc", "test_wf1"])
    ap.add_argument("--out_csv", default="outputs/cross_env/paired_significance.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.metric])
    df[args.metric] = df[args.metric].astype(float)

    abls = sorted(df["ablation"].unique().tolist())
    folds = sorted(df["fold"].unique().tolist())

    # pivot: rows=fold, cols=ablation
    pv = df.pivot_table(index="fold", columns="ablation", values=args.metric, aggfunc="mean")
    pv = pv.loc[folds, abls]

    rows = []
    pairs = list(itertools.combinations(abls, 2))
    pvals_t, pvals_w = [], []

    for a, b in pairs:
        x = pv[a].values
        y = pv[b].values
        # paired t-test
        t_stat, p_t = ttest_rel(x, y, nan_policy="omit")
        # wilcoxon (paired, non-parametric)
        try:
            w_stat, p_w = wilcoxon(x, y, zero_method="wilcox", correction=False)
        except ValueError:
            w_stat, p_w = np.nan, np.nan

        pvals_t.append(p_t)
        pvals_w.append(p_w)
        rows.append([a, b, float(np.mean(x-y)), float(t_stat), float(p_t), float(w_stat) if not np.isnan(w_stat) else np.nan, float(p_w) if not np.isnan(p_w) else np.nan])

    # multiple-comparison correction
    adj_t = holm_bonferroni(np.array(pvals_t, dtype=float))
    # Wilcoxon pvals may contain nan
    pvals_w_arr = np.array(pvals_w, dtype=float)
    mask = ~np.isnan(pvals_w_arr)
    adj_w = np.full_like(pvals_w_arr, np.nan)
    if mask.sum() > 0:
        adj_w[mask] = holm_bonferroni(pvals_w_arr[mask])

    out = pd.DataFrame(rows, columns=[
        "A", "B", "mean(A-B)", "t_stat", "p_t", "w_stat", "p_w"
    ])
    out["p_t_holm"] = adj_t
    out["p_w_holm"] = adj_w

    # also print a compact summary
    print(f"[OK] metric={args.metric}  folds={len(folds)}  ablations={abls}")
    print(out.sort_values("p_t_holm").to_string(index=False))

    out.to_csv(args.out_csv, index=False)
    print(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()
