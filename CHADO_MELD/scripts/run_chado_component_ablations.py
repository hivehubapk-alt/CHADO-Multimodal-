#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


ACC_RE = re.compile(r"Weighted Test Accuracy\s*:\s*([0-9]*\.?[0-9]+)")
F1_RE  = re.compile(r"Weighted Test F1\s*:\s*([0-9]*\.?[0-9]+)")


# ----------------- config utilities -----------------

def _set_first_existing_key(d: dict, keys, value):
    """
    Set the first key in `keys` that exists in dict d. Return True if set, else False.
    """
    for k in keys:
        if k in d:
            d[k] = value
            return True
    return False


def _require_any_key_exists(d: dict, keys, what: str):
    if any(k in d for k in keys):
        return
    raise KeyError(
        f"[FATAL] Could not find any config key for '{what}'. "
        f"Tried aliases: {keys}. "
        f"Please open your CHADO config and add the correct key name to the alias list in this script."
    )


def apply_component_flags(cfg: dict, causal: bool, hyperbolic: bool, transport: bool):
    """
    Toggle CHADO components in cfg['model'].

    IMPORTANT: You MUST ensure your CHADO training script reads these keys.
    This function uses alias lists to match your repo naming.
    """
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        raise KeyError("[FATAL] Config missing top-level 'model:' section.")

    m = cfg["model"]

    # Add/adjust aliases here ONCE if your repo uses different names.
    CAUSAL_KEYS = [
        "use_causal", "causal", "causal_enabled", "enable_causal", "do_causal"
    ]
    HYP_KEYS = [
        "use_hyperbolic", "hyperbolic", "hyperbolic_enabled", "enable_hyperbolic", "do_hyperbolic"
    ]
    # "transport" often implemented as OT; include OT aliases
    TR_KEYS = [
        "use_transport", "transport", "transport_enabled", "enable_transport",
        "use_ot", "ot", "ot_enabled", "enable_ot", "use_geomloss"
    ]

    # Require that at least one alias exists for each component.
    _require_any_key_exists(m, CAUSAL_KEYS, "causal")
    _require_any_key_exists(m, HYP_KEYS, "hyperbolic")
    _require_any_key_exists(m, TR_KEYS, "transport/ot")

    ok1 = _set_first_existing_key(m, CAUSAL_KEYS, bool(causal))
    ok2 = _set_first_existing_key(m, HYP_KEYS, bool(hyperbolic))
    ok3 = _set_first_existing_key(m, TR_KEYS, bool(transport))

    if not (ok1 and ok2 and ok3):
        # This should not happen due to _require_any_key_exists, but keep it safe.
        raise RuntimeError("[FATAL] Failed to set component flags (unexpected).")

    return cfg


def write_yaml(cfg: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ----------------- log parsing -----------------

def parse_metrics_from_log(log_text: str):
    acc_m = ACC_RE.findall(log_text)
    f1_m  = F1_RE.findall(log_text)

    acc = float(acc_m[-1]) if acc_m else None
    f1  = float(f1_m[-1]) if f1_m else None
    return acc, f1


# ----------------- run -----------------

def run_one(torchrun_cmd, env, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        p = subprocess.Popen(torchrun_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        all_lines = []
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
            all_lines.append(line)
        rc = p.wait()
    return rc, "".join(all_lines)


def mean_std(values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None, None
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True, help="Path to your CHADO base YAML config (tri-modal).")
    ap.add_argument("--train_script", required=True, help="Your CHADO training entrypoint, e.g. src/train/train_chado.py")
    ap.add_argument("--out_dir", default="runs/chado_component_ablations", help="Where to write configs/logs/results.")
    ap.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds for mean±std.")
    ap.add_argument("--nproc_per_node", type=int, default=5)
    ap.add_argument("--cuda_visible_devices", default=None, help="e.g. 5,6,7,8,9")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.base_config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Ablation definitions exactly matching your paper table
    ablations = [
        ("Causal only",             dict(causal=True,  hyperbolic=False, transport=False)),
        ("Hyperbolic only",         dict(causal=False, hyperbolic=True,  transport=False)),
        ("Transport only",          dict(causal=False, hyperbolic=False, transport=True)),
        ("Causal + Hyperbolic",     dict(causal=True,  hyperbolic=True,  transport=False)),
        ("Causal + Transport",      dict(causal=True,  hyperbolic=False, transport=True)),
        ("Hyperbolic + Transport",  dict(causal=False, hyperbolic=True,  transport=True)),
        ("All (CHADO)",             dict(causal=True,  hyperbolic=True,  transport=True)),
    ]

    # Env
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    results = []  # per (ablation, seed)
    summary = []  # per ablation mean±std

    for ab_name, flags in ablations:
        accs, f1s = [], []
        safe_tag = ab_name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("__", "_")

        for seed in seeds:
            cfg = deepcopy(base_cfg)

            # Update seed (if exists)
            if "train" in cfg and isinstance(cfg["train"], dict):
                cfg["train"]["seed"] = seed

            # Apply component flags
            cfg = apply_component_flags(cfg, **flags)

            # Set run_name and out_dir if present
            if "logging" in cfg and isinstance(cfg["logging"], dict):
                cfg["logging"]["run_name"] = f"chado_{safe_tag}_seed{seed}"
                cfg["logging"]["out_dir"] = str(out_dir / "runs")
            # Also set top-level out_dir if your script uses it
            if "out_dir" in cfg:
                cfg["out_dir"] = str(out_dir / "runs")

            cfg_path = out_dir / "configs" / f"{safe_tag}_seed{seed}.yaml"
            log_path = out_dir / "logs" / f"{safe_tag}_seed{seed}.log"
            write_yaml(cfg, cfg_path)

            torchrun_cmd = [
                "torchrun",
                f"--nproc_per_node={args.nproc_per_node}",
                args.train_script,
                "--config",
                str(cfg_path),
            ]

            print(f"\n==============================")
            print(f"RUN: {ab_name} | seed={seed}")
            print(f"CFG: {cfg_path}")
            print(f"LOG: {log_path}")
            print(f"==============================\n")

            rc, log_text = run_one(torchrun_cmd, env, log_path)
            if rc != 0:
                raise RuntimeError(f"[FATAL] Run failed for {ab_name} seed={seed}. See log: {log_path}")

            acc, f1 = parse_metrics_from_log(log_text)
            if acc is None or f1 is None:
                raise RuntimeError(
                    f"[FATAL] Could not parse test metrics from log for {ab_name} seed={seed}. "
                    f"Ensure your training script prints lines:\n"
                    f"  Weighted Test Accuracy : <num>\n"
                    f"  Weighted Test F1       : <num>\n"
                    f"Log: {log_path}"
                )

            results.append((ab_name, seed, acc, f1))
            accs.append(acc)
            f1s.append(f1)

        acc_mean, acc_std = mean_std(accs)
        f1_mean, f1_std   = mean_std(f1s)
        summary.append((ab_name, acc_mean, acc_std, f1_mean, f1_std))

    # Write CSVs
    per_seed_csv = out_dir / "per_seed_results.csv"
    summary_csv  = out_dir / "summary_mean_std.csv"

    with open(per_seed_csv, "w") as f:
        f.write("ablation,seed,test_acc,test_f1w\n")
        for ab, seed, acc, f1 in results:
            f.write(f"{ab},{seed},{acc:.6f},{f1:.6f}\n")

    with open(summary_csv, "w") as f:
        f.write("ablation,acc_mean,acc_std,f1w_mean,f1w_std\n")
        for ab, am, ast, fm, fst in summary:
            f.write(f"{ab},{am:.6f},{ast:.6f},{fm:.6f},{fst:.6f}\n")

    print("\n==============================")
    print("DONE. Results written to:")
    print(f"  {per_seed_csv}")
    print(f"  {summary_csv}")
    print("==============================\n")

    # Pretty print summary (paper-friendly)
    print("Ablation Summary (mean ± std):")
    for ab, am, ast, fm, fst in summary:
        print(f"- {ab:22s}  Acc: {am:.4f} ± {ast:.4f}   F1w: {fm:.4f} ± {fst:.4f}")


if __name__ == "__main__":
    main()
