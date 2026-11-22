"""웨이퍼 원래 크기 / die 수와 Scratch 정답률의 관계 (CPU, 재학습 없음)."""
import os, sys, pickle
import numpy as np, torch
proj = sys.argv[1]; os.chdir(proj); sys.path.insert(0, proj)
from wafer.data import WM811K, CLASSES
from ensemble_eval import probs

cache = os.path.join(os.path.expanduser("~"), "wafer_cache")
info_path = os.path.join(cache, "wm811k_shapes_dies.npy")
if not os.path.exists(info_path):
    import pandas.core.indexes
    sys.modules.setdefault("pandas.indexes", pandas.core.indexes)
    with open("LSWMD.pkl", "rb") as f:
        df = pickle.load(f, encoding="latin1")
    maps = df["waferMap"].values
    del df
    info = np.array([(m.shape[0], m.shape[1], int((np.asarray(m) > 0).sum())) for m in maps], np.int64)
    del maps
    np.save(info_path, info)
info = np.load(info_path)            # (n, 3) = 높이, 너비, die 수
ds = WM811K(cache)
assert len(info) == len(ds.labels)
h, w, dies = info.T
key = h * 1000 + w

lab = ds.labeled_subset(0.05, seed=0)
print(f"전체 크기 종류: {len(np.unique(key))}, 라벨 웨이퍼의 크기 종류: {len(np.unique(key[ds.labels >= 0]))}, "
      f"5% 라벨의 크기 종류: {len(np.unique(key[lab]))}")
print(f"die 수: 최소 {dies.min()}, 중앙값 {int(np.median(dies))}, 최대 {dies.max()}")
S = CLASSES.index("Scratch")
for name, idx in (("5% 라벨 Scratch", lab[ds.labels[lab] == S]), ("테스트 Scratch", ds.test[ds.labels[ds.test] == S]),
                  ("테스트 none", ds.test[ds.labels[ds.test] == CLASSES.index("none")])):
    print(f"{name:16s} n={len(idx):5d}  die 수 25/50/75%: {np.percentile(dies[idx], [25, 50, 75]).astype(int).tolist()}")

ts = ds.test[ds.labels[ds.test] == S]
edges = [0, 600, 1200, 2500, 10 ** 6]
bins = np.digitize(dies[ts], edges[1:-1])
print("\n테스트 Scratch 의 die 수 구간별 재현율 (맞춤/장수)")
print(f"{'die 수':>12s} {'장수':>5s} " + "".join(f"{m:>14s}" for m in ("ver3", "ver7", "ver7.7")))
pred = {m: probs(f"{m}_ssl50_ft30", ds, ts, torch.device("cpu")).argmax(1) for m in ("ver3", "ver7", "ver7.7")}
for b in range(len(edges) - 1):
    g = bins == b
    if not g.any():
        continue
    rng = f"{edges[b]}~{edges[b + 1] - 1}" if b < len(edges) - 2 else f"{edges[b]}+"
    print(f"{rng:>12s} {g.sum():5d} " + "".join(f"{(pred[m][g] == S).sum():>7d}/{g.sum():<3d}{(pred[m][g] == S).mean():>4.0%}" for m in pred))
for m in pred:
    miss_none = pred[m] == CLASSES.index("none")
    print(f"{m}: none 으로 놓친 Scratch 의 die 수 중앙값 {int(np.median(dies[ts][miss_none]))}, 맞힌 것 {int(np.median(dies[ts][pred[m] == S]))}")
