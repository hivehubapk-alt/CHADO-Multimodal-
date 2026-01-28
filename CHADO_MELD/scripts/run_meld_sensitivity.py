#!/usr/bin/env python3
import argparse, csv, os, re, subprocess
from pathlib import Path
from datetime import datetime

_RE_ACC = re.compile(r"Weighted Test Accuracy\s*:\s*([0-9]*\.?[0-9]+)")
_RE_F1  = re.compile(r"Weighted Test F1\s*:\s*([0-9]*\.?[0-9]+)")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize(x: float) -> float:
    return x / 100.0 if x > 1.5 else x

def _tail(path: Path, n=80):
    try:
        lines = path.read_text(errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""

def _parse_weighted_test_metrics(text: str):
    m1 = _RE_ACC.search(text)
    m2 = _RE_F1.search(text)
    if not (m1 and m2):
        return None, None
    acc = _normalize(float(m1.group(1)))
    f1  = _normalize(float(m2.group(1)))
    return acc, f1

def _find_ckpt(run_dir: Path):
    # Prefer best.pt then *_best.pt then newest .pt
    candidates = []
    p = run_dir / "best.pt"
    if p.exists():
        return p
    candidates += list(run_dir.rglob("*_best.pt"))
    if candidates:
        return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    candidates = list(run_dir.rglob("*.pt"))
    if candidates:
        return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_config", required=True)
    ap.add_argument("--train_entry", required=True)     # python file launched by torchrun
    ap.add_argument("--eval_entry", required=True)      # python file for evaluation (prints Weighted Test Accuracy/F1)
    ap.add_argument("--config_flag", default="--config")
    ap.add_argument("--nproc_per_node", type=int, default=1)
    ap.add_argument("--cuda_visible_devices", default="")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--tag", default="meld_sensitivity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out_csv", required=True)

    # override keys
    ap.add_argument("--mad_key", default="chado.w_transport")
    ap.add_argument("--ot_key",  default="chado.w_transport")
    ap.add_argument("--hyp_key", default="chado.w_hyperbolic")
    ap.add_argument("--curv_key", default="chado.curvature")

    # sweep values
    ap.add_argument("--mad_values", default="0,0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--ot_values",  default="0,0.01,0.03,0.05,0.1")
    ap.add_argument("--hyp_values", default="0,0.01,0.05,0.1,0.2")
    ap.add_argument("--curv_values", default="0.1,0.3,0.5,1.0,2.0")

    # IMPORTANT: how your code overrides yaml
    # Many repos accept: --override key=value
    # If yours does not, replace the `override_args` section below.
    ap.add_argument("--override_flag", default="--override")

    # extra args
    ap.add_argument("--extra_train_args", default="")
    ap.add_argument("--extra_eval_args", default="")

    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    _ensure_dir(runs_root)
    out_csv = Path(args.out_csv)
    _ensure_dir(out_csv.parent)

    def parse_list(s):
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    sweep = [
        ("mad",  args.mad_key,  parse_list(args.mad_values)),
        ("ot",   args.ot_key,   parse_list(args.ot_values)),
        ("hyp",  args.hyp_key,  parse_list(args.hyp_values)),
        ("curv", args.curv_key, parse_list(args.curv_values)),
    ]

    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["lambda_type", "lambda_value", "test_acc", "test_wf1"])

        env = os.environ.copy()
        if args.cuda_visible_devices.strip():
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()

        for lam_type, key, values in sweep:
            for v in values:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{args.tag}_{lam_type}_{v}_seed{args.seed}_{stamp}"
                run_dir = runs_root / run_name
                _ensure_dir(run_dir)

                train_log = run_dir / "train_stdout.log"
                eval_log  = run_dir / "eval_stdout.log"

                override_args = [args.override_flag, f"{key}={v}"]

                train_cmd = [
                    "torchrun",
                    f"--nproc_per_node={args.nproc_per_node}",
                    args.train_entry,
                    args.config_flag, args.base_config,
                    "--seed", str(args.seed),
                    "--epochs", str(args.epochs),
                    *override_args,
                ]
                if args.extra_train_args.strip():
                    train_cmd += args.extra_train_args.strip().split()

                print(f"\n[TRAIN] {lam_type} {key}={v}")
                print("[CMD] " + " ".join(train_cmd))

                with train_log.open("w") as lf:
                    p = subprocess.Popen(train_cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                    rc = p.wait()

                if rc != 0:
                    print(f"[FAIL] train rc={rc}  log={train_log}")
                    w.writerow([lam_type, v, "", ""])
                    f.flush()
                    continue

                ckpt = _find_ckpt(run_dir)
                if ckpt is None:
                    print(f"[FAIL] no ckpt found under {run_dir}")
                    print(_tail(train_log, 60))
                    w.writerow([lam_type, v, "", ""])
                    f.flush()
                    continue

                # EVAL: must print Weighted Test Accuracy/F1
                eval_cmd = [
                    "python", args.eval_entry,
                    args.config_flag, args.base_config,
                    "--ckpt", str(ckpt),
                    "--split", "test",
                ]
                if args.extra_eval_args.strip():
                    eval_cmd += args.extra_eval_args.strip().split()

                print(f"[EVAL] ckpt={ckpt.name}")
                print("[CMD] " + " ".join(eval_cmd))

                with eval_log.open("w") as lf:
                    p = subprocess.Popen(eval_cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                    rc = p.wait()

                if rc != 0:
                    print(f"[FAIL] eval rc={rc}  log={eval_log}")
                    w.writerow([lam_type, v, "", ""])
                    f.flush()
                    continue

                text = eval_log.read_text(errors="ignore")
                acc, wf1 = _parse_weighted_test_metrics(text)
                if acc is None or wf1 is None:
                    print(f"[FAIL] could not parse metrics from eval log: {eval_log}")
                    print(_tail(eval_log, 80))
                    w.writerow([lam_type, v, "", ""])
                    f.flush()
                    continue

                w.writerow([lam_type, v, round(acc*100, 2), round(wf1*100, 2)])
                f.flush()
                print(f"[OK] {lam_type}={v}  Acc={acc*100:.2f}  WF1={wf1*100:.2f}")

    print(f"\n[DONE] {out_csv}")

if __name__ == "__main__":
    main()
