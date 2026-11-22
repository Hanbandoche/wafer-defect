"""諛붾몣??移??ш린蹂?(ver6.1 12px / ver6.2 6px / ver6.5 4px, ?섎㉧吏 ?숈씪) die ??援ш컙蹂?Scratch ?ы쁽?? CPU, ?ы븰???놁쓬."""
import os, sys
import numpy as np, torch
proj = sys.argv[1]; os.chdir(proj); sys.path.insert(0, proj)
from wafer.data import WM811K, CLASSES
from ensemble_eval import probs

cache = os.path.join(os.path.expanduser("~"), "wafer_cache")
dies = np.load(os.path.join(cache, "wm811k_shapes_dies.npy"))[:, 2]
ds = WM811K(cache)
S, N, L = (CLASSES.index(c) for c in ("Scratch", "none", "Loc"))
ts = ds.test[ds.labels[ds.test] == S]
nt = ds.test[ds.labels[ds.test] == N]
edges = [0, 600, 1200, 2500, 10 ** 6]
names = ["<600", "600~1199", "1200~2499", "2500+"]
runs = [("ver7", "ver7"), ("ver7.7", "ver7.7"), ("ver7.X (crop)", "ver7.X"), ("ver7.X1 (A)", "ver7.X1"), ("ver3 (SimCLR)", "ver3")]
bs, bn = np.digitize(dies[ts], edges[1:-1]), np.digitize(dies[nt], edges[1:-1])
print("Scratch n per bin:", [int((bs == b).sum()) for b in range(4)], "| none n per bin:", [int((bn == b).sum()) for b in range(4)])
print(f"\n{'model':20s}" + "".join(f"{n:>16s}" for n in names) + "   | none?뭆cratch ?ㅽ깘 (援ш컙蹂?")
for label, run in runs:
    ps = probs(f"{run}_ssl50_ft30", ds, ts, torch.device("cpu")).argmax(1)
    pn = probs(f"{run}_ssl50_ft30", ds, nt, torch.device("cpu")).argmax(1)
    cells = "".join(f"{(ps[bs == b] == S).sum():>8d}/{(bs == b).sum():<3d}{(ps[bs == b] == S).mean():>4.0%}" for b in range(4))
    fp = " ".join(f"{int((pn[bn == b] == S).sum()):>3d}" for b in range(4))
    print(f"{label:20s}{cells}   | {fp}   (total hit {(ps == S).sum()}, ?뭤one {(ps == N).sum()}, ?묹oc {(ps == L).sum()})")




