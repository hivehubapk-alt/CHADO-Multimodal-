#!/usr/bin/env python3
import argparse
import os
import re
import matplotlib.pyplot as plt


def parse_method(s: str):
    """
    Parse: "Name,acc,f1"
    Example: "MDERC (Li et al.),62.73,0.607"
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--method must be 'Name,acc,f1' but got: {s}")
    name, acc, f1 = parts[0], float(parts[1]), float(parts[2])
    return name, acc, f1


def safe_float(x):
    if x is None:
        return None
    return float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_name", default="Baseline", type=str)
    ap.add_argument("--ours_name", default="Ours (CHADO)", type=str)

    ap.add_argument("--baseline_acc", required=True, type=float)
    ap.add_argument("--baseline_f1", required=True, type=float)

    ap.add_argument("--ours_acc", required=True, type=float)
    ap.add_argument("--ours_f1", required=True, type=float)

    # Optional: add other methods like your sample table
    ap.add_argument(
        "--method",
        action="append",
        default=[],
        help="Extra method row: 'Name,acc,f1' (can repeat). Example: --method \"MDERC (Li et al.),62.73,0.607\"",
    )

    ap.add_argument("--title", default="MELD Comparison (Accuracy / WF1)", type=str)
    ap.add_argument("--out", default="figures/meld_comparison_table.png", type=str)
    ap.add_argument("--dpi", default=300, type=int)

    args = ap.parse_args()

    rows = []
    # Extra methods first (top of table)
    for m in args.method:
        rows.append(parse_method(m))

    # Then baseline and ours (ours last, bolded)
    rows.append((args.baseline_name, args.baseline_acc, args.baseline_f1))
    rows.append((args.ours_name, args.ours_acc, args.ours_f1))

    # ---- Build figure (table-like) ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Sizing: scale height with number of rows
    n = len(rows)
    fig_h = max(2.4, 0.55 + 0.38 * n)
    fig_w = 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Column headers
    col_labels = ["Method", "Acc.", "WF1"]

    # Format cells
    cell_text = []
    for (name, acc, f1) in rows:
        cell_text.append([name, f"{acc:.3f}", f"{f1:.3f}"])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        edges="horizontal",
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Column widths: make "Method" wider
    # (table cells are indexed [row, col], where row 0 is header)
    for r in range(0, n + 1):
        table[(r, 0)].set_width(0.62)
        table[(r, 1)].set_width(0.19)
        table[(r, 2)].set_width(0.19)

    # Header style
    for c in range(3):
        table[(0, c)].get_text().set_weight("bold")

    # Left align method names
    for r in range(1, n + 1):
        table[(r, 0)].get_text().set_ha("left")

    # Bold the last row (Ours)
    last_r = n  # header is 0, last data row is n
    for c in range(3):
        table[(last_r, c)].get_text().set_weight("bold")

    # Slightly increase row height
    table.scale(1.0, 1.35)

    # Title
    ax.set_title(args.title, fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {args.out}")
    print("Rows:")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
