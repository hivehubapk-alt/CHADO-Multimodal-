import os
import re
import csv
import glob
import argparse
import numpy as np

TEST_RE = re.compile(r"\[TEST\]\s+Acc=([0-9.]+)%\s+WF1=([0-9.]+)")

def parse_last_test_from_log(log_path: str):
    acc = wf1 = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TEST_RE.search(line)
            if m:
                acc = float(m.group(1))
                wf1 = float(m.group(2))
    return acc, wf1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_env", help="outputs/cross_env")
    ap.add_argument("--out_csv", default="outputs/cross_env/loeo_summary_modalities.csv")
    args = ap.parse_args()

    rows = []
    for fold_dir in sorted(glob.glob(os.path.join(args.root, "fold_*"))):
        fold = int(os.path.basename(fold_dir).split("_")[-1])
        for abl in ["T", "TA", "TV", "TAV"]:
            out_dir = os.path.join(fold_dir, abl)
            # Expect a training stdout log if user captured; otherwise look into curves.csv? (not enough)
            # Best practical: user should redirect console to a file per run.
            # If not available, we still record NA.
            log_candidates = sorted(glob.glob(os.path.join(out_dir, "*.log")))
            acc = wf1 = None
            if log_candidates:
                acc, wf1 = parse_last_test_from_log(log_candidates[-1])

            rows.append({"fold": fold, "ablation": abl, "test_acc": acc, "test_wf1": wf1})

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "ablation", "test_acc", "test_wf1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print mean±std for convenience (ignore missing)
    print("[LOEO SUMMARY] mean±std across folds")
    for abl in ["T", "TA", "TV", "TAV"]:
        a = [r["test_acc"] for r in rows if r["ablation"] == abl and r["test_acc"] is not None]
        f1 = [r["test_wf1"] for r in rows if r["ablation"] == abl and r["test_wf1"] is not None]
        if len(a) == 0:
            print(f"  {abl}: (no parsed logs; redirect stdout to .log per run)")
            continue
        print(f"  {abl}: Acc={np.mean(a):.2f}±{np.std(a):.2f}  WF1={np.mean(f1):.2f}±{np.std(f1):.2f}")

    print(f"[SAVED] {args.out_csv}")
    print("Note: to auto-parse metrics, redirect each run's stdout to a .log file.")

if __name__ == "__main__":
    main()
