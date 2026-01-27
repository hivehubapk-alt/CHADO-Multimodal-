import torch, argparse, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mad", default="outputs/cache/test_mad.pt")
    ap.add_argument("--errors", default="outputs/cache/test_errors.pt")
    ap.add_argument("--out", default="outputs/analysis/cost_reduction.csv")
    args = ap.parse_args()

    mad = torch.load(args.mad)
    err = torch.load(args.errors).float()   # 1=error

    order = torch.argsort(mad)  # least ambiguous first
    err_sorted = err[order]

    coverages = np.linspace(0.1,1.0,10)
    rows = []

    base_err = err.mean().item()

    for c in coverages:
        k = int(len(err_sorted)*c)
        sel_err = err_sorted[:k].mean().item()
        cr = 1 - sel_err/base_err
        rows.append((c,sel_err,cr))

    with open(args.out,"w") as f:
        f.write("coverage,error,cost_reduction\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")

    print("[SAVED]", args.out)

if __name__=="__main__":
    main()
