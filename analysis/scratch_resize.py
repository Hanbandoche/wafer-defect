"""가설 확인: 96x96 nearest 리사이즈가 Scratch 선을 끊는가? (분석 전용)"""
import os, pickle, sys
import numpy as np
import torch
from scipy import ndimage

proj, out_dir = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj)
from wafer.data import WM811K, CLASSES, _resize_nearest
from wafer.models import build_model, to_input

cache = os.path.join(os.path.expanduser("~"), "wafer_cache")
ds = WM811K(cache)
SCR = CLASSES.index("Scratch")
scr_idx = np.nonzero(ds.labels == SCR)[0]

orig_path = os.path.join(out_dir, "scratch_orig.pkl")
if not os.path.exists(orig_path):
    import pandas.core.indexes
    sys.modules.setdefault("pandas.indexes", pandas.core.indexes)
    with open("LSWMD.pkl", "rb") as f:
        df = pickle.load(f, encoding="latin1")
    maps = df["waferMap"].values
    orig = {int(i): np.asarray(maps[i]).astype(np.uint8) for i in scr_idx}
    del df, maps
    pickle.dump(orig, open(orig_path, "wb"))
orig = pickle.load(open(orig_path, "rb"))

EIGHT = np.ones((3, 3))
def stats(m):
    fail = m == 2
    lab, n = ndimage.label(fail, structure=EIGHT)
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return n, sizes.max() / sizes.sum()

rows = []
for i in scr_idx:
    o = orig[int(i)]
    r = _resize_nearest(o[None], 96)[0]
    no, lo = stats(o); nr, lr = stats(r)
    rows.append((int(i), o.shape[0], o.shape[1], int((o == 2).sum()), no, lo, nr, lr))
R = np.array(rows, dtype=float)
big = np.maximum(R[:, 1], R[:, 2]) > 96
print(f"labeled Scratch wafers: {len(R)}; original size median {np.median(R[:,1]):.0f}x{np.median(R[:,2]):.0f}, "
      f"max side>96 (downsampled): {big.mean():.1%}")
for name, sel in [("downsampled (side>96)", big), ("upsampled (side<=96)", ~big)]:
    if sel.any():
        print(f"  {name:22s} n={sel.sum():4d}  components orig->96: {np.median(R[sel,4]):.0f}->{np.median(R[sel,6]):.0f} (median)  "
              f"largest-component share orig->96: {R[sel,5].mean():.2f}->{R[sel,7].mean():.2f}  "
              f"wafers whose largest-share dropped >0.1: {(R[sel,5] - R[sel,7] > 0.1).mean():.1%}")

# ver3 가 test Scratch 중 놓친 것 vs 맞힌 것
dev = torch.device("cuda")
m = build_model("resnet18", 9).to(dev)
m.load_state_dict(torch.load(os.path.join("runs", "resnet18", "p0.05", "seed0", "ver3_ssl50_ft30", "best.pt"), map_location=dev, weights_only=True))
m.eval()
test_scr = np.intersect1d(ds.test, scr_idx)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    pred = m(to_input(ds.fetch(test_scr, dev))).argmax(1).cpu().numpy()
pos = {int(r[0]): k for k, r in enumerate(R)}
T = R[[pos[int(i)] for i in test_scr]]
hit = pred == SCR
print(f"\ntest Scratch {len(test_scr)}: ver3 caught {hit.sum()}, missed {(~hit).sum()}")
for name, sel in [("caught", hit), ("missed", ~hit)]:
    print(f"  {name:7s} max side median {np.median(np.maximum(T[sel,1],T[sel,2])):.0f}, downsampled share {(np.maximum(T[sel,1],T[sel,2])>96)[...].mean():.1%}, "
          f"fail dies median {np.median(T[sel,3]):.0f}, largest-share orig {T[sel,5].mean():.2f} -> 96 {T[sel,7].mean():.2f}")

# 놓친 것 중 가장 크게 축소된 6장 그림
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
miss = test_scr[~hit]
order = sorted(miss, key=lambda i: -max(orig[int(i)].shape))[:6]
fig, ax = plt.subplots(2, len(order), figsize=(3 * len(order), 6.4))
cmap = matplotlib.colors.ListedColormap(["white", "#d9dfe7", "#c0392b"])
for k, i in enumerate(order):
    o = orig[int(i)]
    ax[0, k].imshow(o, cmap=cmap, vmin=0, vmax=2, interpolation="nearest"); ax[0, k].set_title(f"orig {o.shape[0]}x{o.shape[1]}\npred={CLASSES[pred[list(test_scr).index(i)]]}", fontsize=9)
    ax[1, k].imshow(_resize_nearest(o[None], 96)[0], cmap=cmap, vmin=0, vmax=2, interpolation="nearest"); ax[1, k].set_title("96x96 (model input)", fontsize=9)
    for a in ax[:, k]: a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "scratch_missed.png"), dpi=90)
print("saved figure")
