import os, json, argparse
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.chado.mad import compute_mad_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="outputs/cache/test_probs.pt")
    ap.add_argument("--labels", default="outputs/cache/test_labels.pt")
    ap.add_argument("--out", default="outputs/analysis/mad_correlations.csv")
    args = ap.parse_args()

    probs = torch.load(args.probs)          # [N,C]
    y = torch.load(args.labels)             # [N,C] in {0,1}

    # Human disagreement proxy: entropy of labels
    eps = 1e-8
    y_mean = y.float().mean(dim=1).clamp(eps,1-eps)
    human_dis = -(y_mean*torch.log(y_mean)+(1-y_mean)*torch.log(1-y_mean))

    mad = compute_mad_scores(probs)

    pear = pearsonr(mad.numpy(), human_dis.numpy())
    spear = spearmanr(mad.numpy(), human_dis.numpy())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f:
        f.write("metric,value\n")
        f.write(f"pearson_r,{pear.statistic}\n")
        f.write(f"pearson_p,{pear.pvalue}\n")
        f.write(f"spearman_r,{spear.statistic}\n")
        f.write(f"spearman_p,{spear.pvalue}\n")

    print("Pearson:", pear)
    print("Spearman:", spear)
    print("[SAVED]", args.out)

if __name__=="__main__":
    main()
