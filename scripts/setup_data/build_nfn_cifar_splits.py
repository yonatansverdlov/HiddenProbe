"""Build an INRDataset-format splits.json for the NFN `siren_cifar_wts` CIFAR-10 INR dataset (your Drive
tar = the arXiv:2305.13546 data), replicating github.com/AllanYangZhou/nfn exactly:

  experiments/classify_siren_configs/dset/cifar.yaml :  prefix=randinit_smaller, split_points=[45000,50000]
  experiments/data_utils.py  SirenDataset            :  label = _(\d)s in the subdir name;
                                                        train=net idx<45000, val=[45000,50000), test=[50000,60000)
  experiments/classify_siren.py extra_aug=20         :  train = base + 20 randinit augmentations (all copies)

So: labels come from the DIRECTORY name (randinit_smaller_aug10_9s -> 9) — no CIFAR lookup needed; the split is
purely by net<N> index; train keeps every augmentation copy of each train image, val/test keep one copy.

Usage:  python scripts/build_nfn_cifar_splits.py --siren_dir <.../siren_cifar_wts> --out <.../nfn_cifar_split.json>
Then:   train_np_e2e.py --dataset nfn_cifar_inr --dataset_dir <.../siren_cifar_wts> --splits <.../nfn_cifar_split.json> \
                        --L 2 --H 32 --out_dim 3 --n_classes 10 --models_c_in 2 --gen_type linear_2_no_acts ...
(INRDataset now remaps the net.{i}.linear.* keys to seq.{i}.* automatically — see data.py.)
"""
import argparse, glob, json, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--siren_dir", required=True, help="the siren_cifar_wts dir (contains randinit_smaller_*/ subdirs)")
ap.add_argument("--out", required=True, help="output splits.json path")
ap.add_argument("--prefix", default="randinit_smaller")
ap.add_argument("--val_point", type=int, default=45000)      # split_points[0]
ap.add_argument("--test_point", type=int, default=50000)     # split_points[1]
ap.add_argument("--no_aug", action="store_true", help="NON-AUGMENTED (normal): train keeps ONE SIREN per image "
                "(like val/test) -> ~45k train, not ~945k. = NFN extra_aug=0.")
args = ap.parse_args()

idx_re = re.compile(r"net(\d+)\.pth$")
lbl_re = re.compile(r"_(\d+)s")    # label = digit(s) before 's' in the subdir name (NFN convention).
                                   # \d+ handles BOTH CIFAR-10 (0-9) AND CIFAR-100 (0-99, e.g. _19s);
                                   # backward-compatible with CIFAR-10 (single-digit is a subset).

paths_by_idx = defaultdict(list); label_by_idx = {}
files = glob.glob(os.path.join(args.siren_dir, f"{args.prefix}_*", "net*.pth"))
print(f"[nfn-split] globbed {len(files)} .pth under {args.siren_dir}/{args.prefix}_*")
for p in files:
    dirname = os.path.basename(os.path.dirname(p))
    if args.no_aug and "aug" in dirname:     # non-aug: use ONLY base randinit_smaller_{label}s dirs
        continue
    m = idx_re.search(os.path.basename(p));  lm = lbl_re.search(dirname)
    if not (m and lm):
        continue
    n = int(m.group(1))
    paths_by_idx[n].append(os.path.relpath(p, args.siren_dir))
    label_by_idx[n] = int(lm.group(1))     # consistent per image index

def split_for(n):
    return "train" if n < args.val_point else ("val" if n < args.test_point else "test")

out = {s: {"path": [], "label": []} for s in ("train", "val", "test")}
for n in sorted(paths_by_idx):
    s = split_for(n); lbl = label_by_idx[n]; copies = sorted(paths_by_idx[n])
    chosen = copies if (s == "train" and not args.no_aug) else copies[:1]   # aug: all train copies; no_aug/val/test: one
    for c in chosen:
        out[s]["path"].append(c); out[s]["label"].append(lbl)

os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(out, f)
print(f"[nfn-split] wrote {args.out}")
for s in ("train", "val", "test"):
    print(f"  {s}: {len(out[s]['label'])} INRs  labels={sorted(set(out[s]['label']))}")
print(f"  distinct train images (idx<{args.val_point}): {sum(1 for n in paths_by_idx if n < args.val_point)}"
      f"  | expect NFN ~45000 imgs x ~21 copies")
