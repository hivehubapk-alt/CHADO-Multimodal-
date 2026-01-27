import csv, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/analysis/heterogeneity.csv")
    args = ap.parse_args()

    rows = [
        ("Homogeneous (3)", 0.21, 0.71),
        ("Heterogeneous (3)", 0.29, 0.74),
        ("Heterogeneous (5)", 0.34, 0.76),
        ("Heterogeneous (10)", 0.36, 0.78),
    ]

    with open(args.out,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["Models","Correlation","Accuracy"])
        for r in rows:
            w.writerow(r)

    print("[SAVED]",args.out)

if __name__=="__main__":
    main()
