#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path
import yaml
import pandas as pd

def load_yaml(p): 
    with open(p,"r") as f: return yaml.safe_load(f)

def expand_env(x):
    if isinstance(x,str): return os.path.expandvars(x)
    if isinstance(x,dict): return {k: expand_env(v) for k,v in x.items()}
    if isinstance(x,list): return [expand_env(v) for v in x]
    return x

def run(cmd, env=None):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["run_component_ablations","run_sensitivity"])
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--master_port_base", type=int, required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--tri_config", required=True)
    ap.add_argument("--ablations_config", default="")
    ap.add_argument("--sensitivity_config", default="")
    args = ap.parse_args()

    paths = expand_env(load_yaml(args.paths))
    tri   = expand_env(load_yaml(args.tri_config))
    out_root = expand_env(paths["outputs_root"])

    train_csv = expand_env(paths["splits"]["train_csv"])
    val_csv   = expand_env(paths["splits"]["val_csv"])
    test_csv  = expand_env(paths["splits"]["test_csv"])

    if args.mode == "run_component_ablations":
        abl = load_yaml(args.ablations_config)
        seed = int(abl["seed"])
        out_base = f"{out_root}/iemocap/ablations_seed{seed}"
        Path(out_base).mkdir(parents=True, exist_ok=True)

        rows = []
        port = args.master_port_base

        for name, flags in abl["variants"].items():
            out_dir = f"{out_base}/{name}"
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            # write a temp config based on tri and overridden flags
            cfg = tri.copy()
            cfg["seed"] = seed
            cfg["chado"] = dict(cfg["chado"])
            cfg["chado"]["use_causal"] = bool(flags.get("use_causal", cfg["chado"]["use_causal"]))
            cfg["chado"]["use_hyperbolic"] = bool(flags.get("use_hyperbolic", cfg["chado"]["use_hyperbolic"]))
            cfg["chado"]["use_ot"] = bool(flags.get("use_ot", cfg["chado"]["use_ot"]))

            tmp_cfg = f"{out_dir}/config.yaml"
            with open(tmp_cfg, "w") as f:
                yaml.safe_dump(cfg, f)

            print("------------------------------------------------------------")
            print(f"[RUN] {name}")
            print(f" out_dir={out_dir}")
            print(f" port={port} nproc={args.nproc}")
            print("------------------------------------------------------------")

            cmd = [
                "torchrun",
                f"--nproc_per_node={args.nproc}",
                f"--master_port={port}",
                "scripts/train_chado_ddp.py",
                "--train_csv", train_csv,
                "--val_csv", val_csv,
                "--test_csv", test_csv,
                "--out_dir", out_dir,
                "--config", tmp_cfg,
                "--save_reports",
            ]
            run(cmd)

            # parse test line from training output not trivial; instead read from reports summary if you add it.
            # Here we read the checkpoint and compute using eval_confusion_and_metrics for stable CSV.
            eval_out = f"{out_dir}/icml_eval"
            Path(eval_out).mkdir(parents=True, exist_ok=True)
            run([
                "python", "scripts/eval_confusion_and_metrics.py",
                "--split_csv", test_csv,
                "--ckpt", f"{out_dir}/best.pt",
                "--ablation", "tri",
                "--batch_size", str(tri["batch_size"]),
                "--device", "cuda:0",
                "--out_dir", eval_out
            ])

            summ_path = f"{eval_out}/summary.txt"
            acc = f1 = mp = mr = None
            with open(summ_path, "r") as f:
                for line in f:
                    if line.startswith("Accuracy:"): acc = float(line.split()[-1])
                    if line.startswith("Macro Precision:"): mp = float(line.split()[-1])
                    if line.startswith("Macro Recall:"): mr = float(line.split()[-1])
                    if line.startswith("Macro F1:"): f1 = float(line.split()[-1])

            rows.append(dict(variant=name, test_acc=acc, test_macro_f1=f1, macro_precision=mp, macro_recall=mr, out_dir=out_dir))
            port += 1

        df = pd.DataFrame(rows)
        out_csv = f"{out_base}/ablations_results.csv"
        df.to_csv(out_csv, index=False)
        print("[OK] saved:", out_csv)

    else:
        sens = load_yaml(args.sensitivity_config)
        seed = int(sens["seed"])
        out_base = f"{out_root}/iemocap/sensitivity_seed{seed}"
        Path(out_base).mkdir(parents=True, exist_ok=True)

        rows = []
        port = args.master_port_base

        for hp_name, values in sens["sweeps"].items():
            for v in values:
                out_dir = f"{out_base}/{hp_name}_{v}"
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                cfg = tri.copy()
                cfg["seed"] = seed
                cfg["weights"] = dict(cfg["weights"])
                cfg["hyper"] = dict(cfg["hyper"])

                if hp_name in cfg["weights"]:
                    cfg["weights"][hp_name] = float(v)
                elif hp_name in cfg["hyper"]:
                    cfg["hyper"][hp_name] = float(v)
                else:
                    raise ValueError(f"Unknown hyperparameter in sensitivity: {hp_name}")

                tmp_cfg = f"{out_dir}/config.yaml"
                with open(tmp_cfg, "w") as f:
                    yaml.safe_dump(cfg, f)

                print("------------------------------------------------------------")
                print(f"[RUN] {hp_name}={v}")
                print(f" out_dir={out_dir}")
                print(f" port={port} nproc={args.nproc}")
                print("------------------------------------------------------------")

                run([
                    "torchrun",
                    f"--nproc_per_node={args.nproc}",
                    f"--master_port={port}",
                    "scripts/train_chado_ddp.py",
                    "--train_csv", train_csv,
                    "--val_csv", val_csv,
                    "--test_csv", test_csv,
                    "--out_dir", out_dir,
                    "--config", tmp_cfg,
                    "--save_reports",
                ])

                # evaluate and store summary
                eval_out = f"{out_dir}/icml_eval"
                Path(eval_out).mkdir(parents=True, exist_ok=True)
                run([
                    "python", "scripts/eval_confusion_and_metrics.py",
                    "--split_csv", test_csv,
                    "--ckpt", f"{out_dir}/best.pt",
                    "--ablation", "tri",
                    "--batch_size", str(tri["batch_size"]),
                    "--device", "cuda:0",
                    "--out_dir", eval_out
                ])

                summ_path = f"{eval_out}/summary.txt"
                acc = f1 = mp = mr = None
                with open(summ_path, "r") as f:
                    for line in f:
                        if line.startswith("Accuracy:"): acc = float(line.split()[-1])
                        if line.startswith("Macro Precision:"): mp = float(line.split()[-1])
                        if line.startswith("Macro Recall:"): mr = float(line.split()[-1])
                        if line.startswith("Macro F1:"): f1 = float(line.split()[-1])

                rows.append(dict(sweep=hp_name, value=float(v), test_acc=acc, test_macro_f1=f1,
                                 macro_precision=mp, macro_recall=mr, out_dir=out_dir))
                port += 1

        df = pd.DataFrame(rows)
        out_csv = f"{out_base}/sensitivity_results.csv"
        df.to_csv(out_csv, index=False)
        print("[OK] saved:", out_csv)

if __name__ == "__main__":
    main()
