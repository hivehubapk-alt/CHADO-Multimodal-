import os
import csv
import argparse
import subprocess
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIGS = [
    ("Causal only",             dict(causal=True,  hyp=False, ot=False, mad=False, selfcorr=False)),
    ("Hyperbolic only",         dict(causal=False, hyp=True,  ot=False, mad=False, selfcorr=False)),
    ("Transport only",          dict(causal=False, hyp=False, ot=True,  mad=False, selfcorr=False)),
    ("Causal + Hyperbolic",     dict(causal=True,  hyp=True,  ot=False, mad=False, selfcorr=False)),
    ("Causal + Transport",      dict(causal=True,  hyp=False, ot=True,  mad=False, selfcorr=False)),
    ("Hyperbolic + Transport",  dict(causal=False, hyp=True,  ot=True,  mad=False, selfcorr=False)),
    ("All (CHADO)",             dict(causal=True,  hyp=True,  ot=True,  mad=False, selfcorr=False)),
    ("All (CHADO) + MAD",       dict(causal=True,  hyp=True,  ot=True,  mad=True,  selfcorr=False)),
    ("All + MAD + SelfCorrect", dict(causal=True,  hyp=True,  ot=True,  mad=True,  selfcorr=True)),
]


def parse_extended(stdout: str):
    """
    Parse outputs from eval_extended_metrics.py.
    """
    overall_acc = overall_wf1 = amb_acc = amb_wf1 = inf_ms = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Overall:"):
            # Overall: Acc=xx.xx% WF1=yy.yy
            overall_acc = float(s.split("Acc=")[1].split("%")[0]) / 100.0
            overall_wf1 = float(s.split("WF1=")[1].strip()) / 100.0
        elif s.startswith("Ambiguous"):
            amb_acc = float(s.split("Acc=")[1].split("%")[0]) / 100.0
            amb_wf1 = float(s.split("WF1=")[1].strip()) / 100.0
        elif s.startswith("Inference:"):
            inf_ms = float(s.split(":")[1].split("ms")[0].strip())
    if None in (overall_acc, overall_wf1, amb_acc, amb_wf1, inf_ms):
        return None
    return overall_acc, overall_wf1, amb_acc, amb_wf1, inf_ms


def run_one(cfg, ablation, port, flags, seeds, ambiguous_topk):
    rows = []
    for seed in seeds:
        cmd_train = f"""
cd /home/tahirahmad/CHADO_CMU_MOSEI
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port={port} src/train/train.py \
  --cfg {cfg} --ablation {ablation} \
  --override experiment.seed={seed} \
  --override chado.causal.enable={str(flags['causal']).lower()} \
  --override chado.hyperbolic.enable={str(flags['hyp']).lower()} \
  --override chado.ot.enable={str(flags['ot']).lower()} \
  --override chado.mad.enable={str(flags['mad']).lower()} \
  --override chado.self_correction.enable={str(flags['selfcorr']).lower()}
"""
        p = subprocess.run(["bash", "-lc", cmd_train], capture_output=True, text=True)
        print(p.stdout)
        if p.returncode != 0:
            print(p.stderr)
            rows.append(None)
            continue

        ckpt = f"outputs/checkpoints/{ablation}_best.pt"
        if not os.path.exists(f"/home/tahirahmad/CHADO_CMU_MOSEI/{ckpt}"):
            print(f"[ERR] checkpoint missing: {ckpt}")
            rows.append(None)
            continue

        cmd_eval = f"""
cd /home/tahirahmad/CHADO_CMU_MOSEI
python scripts/eval_extended_metrics.py --cfg {cfg} --ablation {ablation} --ckpt {ckpt} --ambiguous_topk {ambiguous_topk}
"""
        q = subprocess.run(["bash", "-lc", cmd_eval], capture_output=True, text=True)
        print(q.stdout)
        if q.returncode != 0:
            print(q.stderr)
            rows.append(None)
            continue

        parsed = parse_extended(q.stdout)
        rows.append(parsed)

    vals = [r for r in rows if r is not None]
    if not vals:
        return None

    arr = np.array(vals, dtype=float)
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="src/configs/chado_mosei_emo6.yaml")
    ap.add_argument("--ablation", type=str, default="TAV", choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--port", type=int, default=29624)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--ambiguous_topk", type=float, default=0.20)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    out_csv = "outputs/table_extended_metrics.csv"
    table = []

    for name, flags in CONFIGS:
        print("\n==============================")
        print("CONFIG:", name)
        print("==============================\n")
        res = run_one(args.cfg, args.ablation, args.port, flags, seeds, args.ambiguous_topk)
        if res is None:
            table.append([name, "ERR", "ERR", "ERR", "ERR", "ERR"])
            continue

        mu, sd = res
        overall_acc, overall_wf1, amb_acc, amb_wf1, inf_ms = mu
        overall_acc_sd, overall_wf1_sd, amb_acc_sd, amb_wf1_sd, inf_ms_sd = sd

        table.append([
            name,
            f"{overall_acc*100:.2f} ± {overall_acc_sd*100:.2f}",
            f"{overall_wf1*100:.2f} ± {overall_wf1_sd*100:.2f}",
            f"{amb_acc*100:.2f} ± {amb_acc_sd*100:.2f}",
            f"{amb_wf1*100:.2f} ± {amb_wf1_sd*100:.2f}",
            f"{inf_ms:.2f} ± {inf_ms_sd:.2f}",
        ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Components Used", "Overall Accuracy", "Overall WF1", "Ambiguous Accuracy", "Ambiguous WF1", "Inference Time (ms/sample)"])
        w.writerows(table)

    print("\nSaved:", out_csv)

    # plots
    names = [r[0] for r in table if r[1] != "ERR"]
    if not names:
        print("[WARN] All rows ERR; plots will be empty.")
        return

    wf1 = []
    ms = []
    for r in table:
        if r[1] == "ERR":
            continue
        wf1.append(float(r[2].split("±")[0].strip()))
        ms.append(float(r[5].split("±")[0].strip()))

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(names)), wf1)
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.ylabel("Overall WF1 (%)")
    plt.tight_layout()
    plt.savefig("outputs/plots/overall_wf1.png", dpi=200)

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(names)), ms)
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.ylabel("Inference Time (ms/sample)")
    plt.tight_layout()
    plt.savefig("outputs/plots/inference_time.png", dpi=200)

    print("Saved plots:")
    print("  outputs/plots/overall_wf1.png")
    print("  outputs/plots/inference_time.png")


if __name__ == "__main__":
    main()
