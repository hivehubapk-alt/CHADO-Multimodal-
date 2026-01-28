import os
from pathlib import Path

ROOT = Path("/home/tahirahmad/CHADO_CMU_MOSEI/data/raw/cmu-mosei/CMU-MOSEI")

print("ROOT exists:", ROOT.exists())
print("ROOT path  :", ROOT)

if not ROOT.exists():
    raise SystemExit("ERROR: CMU-MOSEI folder not found at expected path.")

# Show top-level structure
print("\n[Top-level listing]")
for p in sorted(ROOT.iterdir()):
    kind = "DIR " if p.is_dir() else "FILE"
    print(f"  {kind}  {p.name}")

# Walk and summarize
print("\n[Walk summary]")
file_count = 0
dir_count = 0
ext_hist = {}

for dirpath, dirnames, filenames in os.walk(ROOT):
    dir_count += len(dirnames)
    file_count += len(filenames)
    for fn in filenames:
        ext = Path(fn).suffix.lower()
        ext_hist[ext] = ext_hist.get(ext, 0) + 1

print("Total dirs :", dir_count)
print("Total files:", file_count)

print("\n[Top extensions]")
for ext, n in sorted(ext_hist.items(), key=lambda x: x[1], reverse=True)[:25]:
    print(f"  {ext or '<no-ext>'}: {n}")

# Print a few example files (deep)
print("\n[Sample files (first 80)]")
shown = 0
for p in ROOT.rglob("*"):
    if p.is_file():
        print(" ", str(p))
        shown += 1
        if shown >= 80:
            break

print("\nDone.")
