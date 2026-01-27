import torch, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

EMO = ["happy","sad","angry","fearful","disgust","surprise"]

def main():
    probs = torch.load("outputs/cache/test_probs.pt")
    y = torch.load("outputs/cache/test_labels.pt")

    plt.figure(figsize=(6,5))
    for i,e in enumerate(EMO):
        p,r,_ = precision_recall_curve(y[:,i], probs[:,i])
        plt.plot(r,p,label=e)

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("outputs/analysis/plots/pr_curves.png", dpi=300)
    print("[SAVED] pr_curves.png")

if __name__ == "__main__":
    main()
