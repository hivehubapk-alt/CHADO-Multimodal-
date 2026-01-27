# import os
# import csv
# import math
# from collections import defaultdict

# import matplotlib.pyplot as plt

# CSV_PATH = "outputs/hparam_sensitivity/lambda_sensitivity_summary.csv"
# OUT_DIR = "outputs/hparam_sensitivity"
# os.makedirs(OUT_DIR, exist_ok=True)

# def safe_float(x):
#     try:
#         if x is None:
#             return None
#         x = str(x).strip()
#         if x == "" or x.lower() == "none":
#             return None
#         return float(x)
#     except Exception:
#         return None

# def load_rows(path):
#     rows = []
#     with open(path, "r", encoding="utf-8") as f:
#         r = csv.DictReader(f)
#         for row in r:
#             rows.append({
#                 "lambda_type": row["lambda_type"],
#                 "lambda_value": safe_float(row["lambda_value"]),
#                 "test_acc": safe_float(row["test_acc"]),
#                 "test_wf1": safe_float(row["test_wf1"]),
#             })
#     return rows

# def plot_metric(rows, metric, out_png):
#     by_type = defaultdict(list)
#     for row in rows:
#         if row["lambda_value"] is None or row[metric] is None:
#             continue
#         by_type[row["lambda_type"]].append((row["lambda_value"], row[metric]))

#     plt.figure()
#     for t, pts in sorted(by_type.items()):
#         pts = sorted(pts, key=lambda x: x[0])
#         xs = [p[0] for p in pts]
#         ys = [p[1] for p in pts]
#         plt.plot(xs, ys, marker="o", label=t)

#     plt.xlabel("lambda value")
#     plt.ylabel(metric)
#     plt.title(f"Hyperparameter sensitivity: {metric}")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=200)
#     print(f"[SAVED] {out_png}")

# def main():
#     if not os.path.exists(CSV_PATH):
#         raise FileNotFoundError(f"Missing: {CSV_PATH}")

#     rows = load_rows(CSV_PATH)
#     plot_metric(rows, "test_acc", os.path.join(OUT_DIR, "lambda_sensitivity_acc.png"))
#     plot_metric(rows, "test_wf1", os.path.join(OUT_DIR, "lambda_sensitivity_wf1.png"))

# if __name__ == "__main__":
#     main()
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def _pretty_name(t: str) -> str:
    t = str(t).lower().strip()
    if t == "mad":
        return r"$\lambda_1$ (MAD)"
    if t == "ot":
        return r"$\lambda_2$ (OT)"
    if t in ("hyp", "hyperbolic"):
        return r"$\lambda_3$ (Hyperbolic)"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to lambda_sensitivity_summary.csv")
    ap.add_argument("--out_dir", default="outputs/hparam_sensitivity", help="Where to save plots")
    ap.add_argument("--title_prefix", default="Hyperparameter sensitivity", help="Plot title prefix")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = {"lambda_type", "lambda_value", "test_acc", "test_wf1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}. Found: {df.columns.tolist()}")

    # Coerce numeric (robust to strings)
    df["lambda_value"] = pd.to_numeric(df["lambda_value"], errors="coerce")
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
    df["test_wf1"] = pd.to_numeric(df["test_wf1"], errors="coerce")
    df = df.dropna(subset=["lambda_type", "lambda_value", "test_acc", "test_wf1"])

    # Sort for clean curves
    df = df.sort_values(["lambda_type", "lambda_value"]).reset_index(drop=True)

    # --- ACC curve ---
    plt.figure(figsize=(7.2, 4.2), dpi=200)
    for lt, g in df.groupby("lambda_type"):
        g = g.sort_values("lambda_value")
        plt.plot(
            g["lambda_value"].values,
            g["test_acc"].values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=_pretty_name(lt),
        )

    plt.xlabel(r"Weight ($\lambda$)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"{args.title_prefix}: Accuracy vs. λ")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    acc_out = os.path.join(args.out_dir, "lambda_sensitivity_acc.png")
    plt.savefig(acc_out)
    plt.close()

    # --- WF1 curve ---
    plt.figure(figsize=(7.2, 4.2), dpi=200)
    for lt, g in df.groupby("lambda_type"):
        g = g.sort_values("lambda_value")
        plt.plot(
            g["lambda_value"].values,
            g["test_wf1"].values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=_pretty_name(lt),
        )

    plt.xlabel(r"Weight ($\lambda$)")
    plt.ylabel("Weighted F1 (%)")
    plt.title(f"{args.title_prefix}: WF1 vs. λ")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    wf1_out = os.path.join(args.out_dir, "lambda_sensitivity_wf1.png")
    plt.savefig(wf1_out)
    plt.close()

    print("[SAVED]", acc_out)
    print("[SAVED]", wf1_out)


if __name__ == "__main__":
    main()
