import argparse
import numpy as np
from mmsdk.mmdatasdk import computational_sequence as cs

def brief(x, name="obj"):
    try:
        t = type(x)
        if hasattr(x, "shape"):
            return f"{name}: type={t} shape={x.shape} dtype={getattr(x,'dtype',None)}"
        if isinstance(x, (list, tuple)):
            return f"{name}: type={t} len={len(x)}"
        if isinstance(x, dict):
            return f"{name}: type=dict keys={list(x.keys())[:20]}"
        return f"{name}: type={t} str={str(x)[:120]}"
    except Exception as e:
        return f"{name}: <error {e}>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words_csd", required=True)
    ap.add_argument("--vid", default="-3nNcZdcdvU")
    args = ap.parse_args()

    seq = cs(args.words_csd)
    _ = seq.data

    print("[SEQ] keys:", len(seq.data))
    print("[SEQ] sample keys:", list(seq.data.keys())[:10])

    if args.vid not in seq.data:
        print("[ERR] vid not found:", args.vid)
        return

    item = seq.data[args.vid]
    print("\n[ITEM]", args.vid)
    print(brief(item, "item"))

    if isinstance(item, dict):
        for k, v in item.items():
            print(" ", brief(v, f"item['{k}']"))

        # Try common candidates that may hold word timing rows
        for cand in ["features", "intervals", "data", "timestamps", "words", "values"]:
            if cand in item:
                v = item[cand]
                print("\n[CANDIDATE]", cand, "->", brief(v, cand))
                try:
                    arr = np.array(v)
                    print("  np.array:", brief(arr, "arr"))
                    if arr.size > 0:
                        print("  first row:", arr[0])
                except Exception as e:
                    print("  could not cast to np.array:", e)

    else:
        # if not dict, try cast directly
        try:
            arr = np.array(item)
            print(brief(arr, "arr"))
            if arr.size > 0:
                print("first row:", arr[0])
        except Exception as e:
            print("Could not cast item to np.array:", e)

if __name__ == "__main__":
    main()
