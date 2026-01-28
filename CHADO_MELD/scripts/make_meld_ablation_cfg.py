# scripts/make_meld_ablation_cfg.py
#!/usr/bin/env python3
import argparse
import os
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--out_config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--use_causal", type=int, required=True)
    ap.add_argument("--use_hyperbolic", type=int, required=True)
    ap.add_argument("--use_transport", type=int, required=True)
    ap.add_argument("--use_refinement", type=int, default=1)

    ap.add_argument("--baseline_ckpt", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.base_config, "r"))

    # Fix MELD label column (your CSV has 'emotion')
    cfg.setdefault("data", {})
    if cfg["data"].get("label_col", "emotion") == "label":
        cfg["data"]["label_col"] = "emotion"

    # Ensure modalities ON (T+A+V)
    cfg.setdefault("model", {})
    cfg["model"]["use_text"] = True
    cfg["model"]["use_audio"] = True
    cfg["model"]["use_video"] = True

    # CHADO toggles
    cfg.setdefault("chado", {})
    cfg["chado"]["use_causal"] = bool(args.use_causal)
    cfg["chado"]["use_hyperbolic"] = bool(args.use_hyperbolic)
    cfg["chado"]["use_transport"] = bool(args.use_transport)
    cfg["chado"]["use_refinement"] = bool(args.use_refinement)

    # initialize from trusted baseline
    cfg["chado"]["eval_from_baseline_ckpt"] = args.baseline_ckpt
    cfg["chado"]["eval_only"] = False

    # logging
    cfg.setdefault("logging", {})
    cfg["logging"]["out_dir"] = args.out_dir
    cfg["logging"]["run_name"] = args.run_name

    os.makedirs(os.path.dirname(args.out_config), exist_ok=True)
    yaml.safe_dump(cfg, open(args.out_config, "w"), sort_keys=False)

    print(f"[OK] wrote {args.out_config}")
    print(f"[CFG] run_name={args.run_name}")
    print(f"[CFG] out_dir={args.out_dir}")
    print(f"[CFG] label_col={cfg['data'].get('label_col')}")
    print(f"[CFG] baseline_ckpt={args.baseline_ckpt}")
    print(
        f"[CFG] use_causal={cfg['chado']['use_causal']} "
        f"use_hyperbolic={cfg['chado']['use_hyperbolic']} "
        f"use_transport={cfg['chado']['use_transport']} "
        f"use_refinement={cfg['chado']['use_refinement']}"
    )
    print(f"[CFG] eval_only={cfg['chado']['eval_only']}")


if __name__ == "__main__":
    main()




# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import argparse
# import copy
# import os
# import yaml


# def _set_many(d, keys, value):
#     for k in keys:
#         d[k] = value


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--base_config", required=True)
#     ap.add_argument("--out_config", required=True)

#     ap.add_argument("--run_name", required=True)
#     ap.add_argument("--out_dir", required=True)

#     ap.add_argument("--use_causal", type=int, required=True)
#     ap.add_argument("--use_hyperbolic", type=int, required=True)
#     ap.add_argument("--use_transport", type=int, required=True)
#     ap.add_argument("--use_refinement", type=int, default=1)

#     ap.add_argument(
#         "--baseline_ckpt",
#         default="/CHADO_MELD/runs/baseline_trimodal_meld_best.pt"
#     )
#     args = ap.parse_args()

#     with open(args.base_config, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     cfg = copy.deepcopy(cfg)

#     # ---- logging ----
#     cfg.setdefault("logging", {})
#     cfg["logging"]["run_name"] = args.run_name
#     cfg["logging"]["out_dir"] = args.out_dir

#     # ---- MELD label column ----
#     cfg.setdefault("data", {})
#     if cfg["data"].get("label_col", "emotion") == "label":
#         cfg["data"]["label_col"] = "emotion"
#     else:
#         cfg["data"]["label_col"] = cfg["data"].get("label_col", "emotion")

#     # ---- CHADO section ----
#     cfg.setdefault("chado", {})

#     # IMPORTANT: ensure training mode is enabled in *any* repo variant
#     # Some repos use eval_only=True to run a wrapper evaluation; for ablations we train.
#     cfg["chado"]["eval_only"] = False
#     cfg["chado"]["train"] = True
#     cfg["chado"]["train_enabled"] = True
#     cfg["chado"]["enable_train"] = True

#     # Write baseline ckpt into all common key names that different codebases expect
#     ckpt = args.baseline_ckpt
#     _set_many(cfg["chado"], [
#         # common names
#         "baseline_ckpt",
#         "base_ckpt",
#         "init_ckpt",
#         "init_from_ckpt",
#         "init_from_baseline_ckpt",
#         "from_baseline_ckpt",
#         "pretrained_ckpt",
#         "load_from",

#         # eval-only wrappers often use these (your file printed this message)
#         "eval_from_baseline_ckpt",
#         "eval_only_baseline_ckpt",
#         "eval_baseline_ckpt",
#         "eval_ckpt",
#         "eval_only_ckpt",
#         "eval_init_ckpt",
#     ], ckpt)

#     # Component toggles
#     cfg["chado"]["use_causal"] = bool(args.use_causal)
#     cfg["chado"]["use_hyperbolic"] = bool(args.use_hyperbolic)
#     cfg["chado"]["use_transport"] = bool(args.use_transport)
#     cfg["chado"]["use_refinement"] = bool(args.use_refinement)

#     os.makedirs(os.path.dirname(args.out_config), exist_ok=True)
#     with open(args.out_config, "w", encoding="utf-8") as f:
#         yaml.safe_dump(cfg, f, sort_keys=False)

#     print(f"[OK] wrote {args.out_config}")
#     print(f"[CFG] run_name={cfg['logging']['run_name']}")
#     print(f"[CFG] out_dir={cfg['logging']['out_dir']}")
#     print(f"[CFG] label_col={cfg['data']['label_col']}")
#     print(f"[CFG] baseline_ckpt={ckpt}")
#     print(
#         f"[CFG] use_causal={cfg['chado']['use_causal']} "
#         f"use_hyperbolic={cfg['chado']['use_hyperbolic']} "
#         f"use_transport={cfg['chado']['use_transport']} "
#         f"use_refinement={cfg['chado']['use_refinement']}"
#     )
#     print(f"[CFG] eval_only={cfg['chado']['eval_only']}")
#     print(f"[CFG] train_enabled={cfg['chado'].get('train_enabled', None)}")


# if __name__ == "__main__":
#     main()
