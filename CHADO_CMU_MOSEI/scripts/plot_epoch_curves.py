import torch
import matplotlib.pyplot as plt

def plot_curves(curves, labels, title="CMU-MOSEI", ylabel="Accuracy"):
    plt.figure(figsize=(8,5))

    for path, label, color in curves:
        data = torch.load(path)
        y = data["acc"]
        x = list(range(1, len(y)+1))
        plt.plot(
            x, y,
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
            color=color
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    curves = [
        ("outputs/logs/TAV_curve.pt", "CHADO TAV", "green"),
        ("outputs/logs/TA_curve.pt",  "CHADO TA",  "red"),
    ]
    plot_curves(curves)
