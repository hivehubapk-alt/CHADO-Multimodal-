import os
import re
import csv
import subprocess

CFG = "src/configs/chado_mosei_emo6.yaml"
ABLATION = "TAV"          # usually best to test sensitivity on full model
SEED = 42

# λ grids
LAMBDA_GRID = {
    "mad": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],     # λ₁
    "ot":  [0.0, 0.01, 0.03, 0.05, 0.1],       # λ₂
    "hyp": [0.0, 0.01, 0.05, 0.1, 0.2],        # λ₃
}

OUT_ROOT = "outputs/hparam_sensitivity"
os.makedirs(OUT_ROOT, exist_ok=True)

TEST_RE = re.compile(r"\[TEST\]\s+Acc=([0-9.]+)%\s+WF1=([0-9.]+)")

def run_and_parse(tag: str, overrides: dict):
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "src/train/train.py",
        "--cfg", CFG,
        "--ablation", ABLATION,
        "--override", f"experiment.seed={SEED}",
        # optional: if your code supports it, it will place logs/ckpts under this folder
        "--override", f"experiment.out_dir={out_dir}",
    ]
    for k, v in overrides.items():
        cmd.extend(["--override", f"{k}={v}"])

    log_path = os.path.join(out_dir, "run.log")
    print("\n[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # save full log (critical for debugging + reproducibility)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    if proc.returncode != 0:
        print(f"[FAIL] {tag} (returncode={proc.returncode}). See: {log_path}")
        return None, None

    m = TEST_RE.search(proc.stdout)
    if not m:
        print(f"[WARN] Could not find [TEST] line for {tag}. See: {log_path}")
        return None, None

    test_acc = float(m.group(1))   # already percent
    test_wf1 = float(m.group(2))   # already percent
    print(f"[PARSED] {tag}  Acc={test_acc:.2f}  WF1={test_wf1:.2f}")
    return test_acc, test_wf1

def main():
    results_csv = os.path.join(OUT_ROOT, "lambda_sensitivity_summary.csv")

    # write header once
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda_type", "lambda_value", "test_acc", "test_wf1"])

        # λ₁: MAD
        for v in LAMBDA_GRID["mad"]:
            acc, wf1 = run_and_parse(
                tag=f"mad_{v}",
                overrides={
                    "chado.mad.enable": True if v > 0 else False,
                    "chado.mad.lambda": v,
                },
            )
            w.writerow(["mad", v, acc, wf1])
            f.flush()

        # λ₂: OT
        for v in LAMBDA_GRID["ot"]:
            acc, wf1 = run_and_parse(
                tag=f"ot_{v}",
                overrides={
                    "chado.ot.enable": True if v > 0 else False,
                    "chado.ot.lambda": v,
                },
            )
            w.writerow(["ot", v, acc, wf1])
            f.flush()

        # λ₃: Hyperbolic
        for v in LAMBDA_GRID["hyp"]:
            acc, wf1 = run_and_parse(
                tag=f"hyp_{v}",
                overrides={
                    "chado.hyperbolic.enable": True if v > 0 else False,
                    "chado.hyperbolic.lambda": v,
                },
            )
            w.writerow(["hyp", v, acc, wf1])
            f.flush()

    print(f"\n[SAVED] {results_csv}")
    print(f"[LOGS ] Per-run logs under: {OUT_ROOT}/<tag>/run.log")

if __name__ == "__main__":
    main()
