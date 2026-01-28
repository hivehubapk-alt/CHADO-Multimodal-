import os
import copy
import csv
import subprocess
import argparse
from pathlib import Path

def set_cfg(cfg_path: str, overrides: dict) -> str:
    """
    Create a temporary YAML by copying cfg and applying key overrides with yq-like sed-free approach:
    we keep it simple: we write a minimal override YAML and use the train.py to merge it (supported below).
    """
    # we rely on train.py supporting --override key=value pairs; if not, we just use multiple cfgs.
    raise NotImplementedError("This runner uses train.py --override. Ensure train.py supports it.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="src/configs/chado_mosei_emo6.yaml")
    ap.add_argument("--ablation", type=str, default="TAV", choices=["T","TA","TV","TAV"])
    ap.add_argument("--port", type=int, default=29624)
    args = ap.parse_args()

    # Configurations matching your professor table
    configs = [
        ("Baseline (Euclidean, no disentanglement)", dict(causal=False, hyp=False, ot=False, mad=False, selfcorr=False)),
        ("+ Causal Disentanglement only",           dict(causal=True,  hyp=False, ot=False, mad=False, selfcorr=False)),
        ("+ Hyperbolic Embedding only",             dict(causal=False, hyp=True,  ot=False, mad=False, selfcorr=False)),
        ("+ Optimal Transport only",                dict(causal=False, hyp=False, ot=True,  mad=False, selfcorr=False)),
        ("+ Causal + Hyperbolic",                   dict(causal=True,  hyp=True,  ot=False, mad=False, selfcorr=False)),
        ("+ Causal + OT",                           dict(causal=True,  hyp=False, ot=True,  mad=False, selfcorr=False)),
        ("+ Hyperbolic + OT",                       dict(causal=False, hyp=True,  ot=True,  mad=False, selfcorr=False)),
        ("Full CHADO (all three)",                  dict(causal=True,  hyp=True,  ot=True,  mad=False, selfcorr=False)),
        ("+ MAD",                                   dict(causal=True,  hyp=True,  ot=True,  mad=True,  selfcorr=False)),
        ("+ Self-correction",                       dict(causal=True,  hyp=True,  ot=True,  mad=True,  selfcorr=True)),
    ]

    out_csv = Path("outputs/ablation_table2_mosei.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, flags in configs:
        cmd = [
            "bash", "-lc",
            f"""
cd /home/tahirahmad/CHADO_CMU_MOSEI
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port={args.port} src/train/train.py \
  --cfg {args.cfg} --ablation {args.ablation} \
  --override chado.causal.enable={str(flags['causal']).lower()} \
  --override chado.hyperbolic.enable={str(flags['hyp']).lower()} \
  --override chado.ot.enable={str(flags['ot']).lower()} \
  --override chado.mad.enable={str(flags['mad']).lower()} \
  --override chado.self_correction.enable={str(flags['selfcorr']).lower()}
"""
        ]

        print("\n=== RUN:", name, "===\n")
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout)
        if p.returncode != 0:
            print(p.stderr)
            rows.append([name, "ERR", "ERR"])
            continue

        # parse final [TEST] line: [TEST] Acc=.. WF1=..
        acc, wf1 = None, None
        for line in p.stdout.splitlines():
            if line.strip().startswith("[TEST]"):
                # expected: [TEST] Acc=xx.xx%  WF1=yy.yy
                s = line.strip()
                # fallback parse patterns
                if "Acc=" in s and "WF1=" in s:
                    try:
                        acc_part = s.split("Acc=")[1].split("%")[0]
                        wf1_part = s.split("WF1=")[1].strip()
                        acc = float(acc_part)
                        wf1 = float(wf1_part)
                    except Exception:
                        pass
        if acc is None or wf1 is None:
            rows.append([name, "NA", "NA"])
        else:
            rows.append([name, f"{acc:.2f}", f"{wf1:.2f}"])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Configuration", "Acc", "WF1"])
        w.writerows(rows)

    print("\nSaved:", str(out_csv))

if __name__ == "__main__":
    main()
