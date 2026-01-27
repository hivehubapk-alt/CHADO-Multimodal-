import torch, matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def main():
    mad = torch.load("outputs/cache/test_mad.pt").numpy()
    err = torch.load("outputs/cache/test_errors.pt").numpy()

    r_p, p_p = pearsonr(mad, err)
    r_s, p_s = spearmanr(mad, err)

    plt.figure(figsize=(5,4))
    plt.scatter(mad, err, alpha=0.4)
    plt.xlabel("MAD ambiguity")
    plt.ylabel("Prediction error")
    plt.title(f"Pearson r={r_p:.3f}, Spearman ρ={r_s:.3f}")
    plt.grid(alpha=0.3)
    plt.savefig("outputs/analysis/plots/mad_vs_error.png", dpi=300)
    print("[SAVED] mad_vs_error.png")

if __name__ == "__main__":
    main()
