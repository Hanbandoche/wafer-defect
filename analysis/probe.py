"""고정 특징 평가: 인코더 고정 → 5% 라벨 특징으로 kNN(k=20, 거리 가중) / 선형 분류기 (seed 3개). CPU 아님, GPU."""
import os, sys
import numpy as np, torch, torch.nn.functional as F
proj = sys.argv[1]; os.chdir(proj); sys.path.insert(0, proj)
from sklearn.metrics import f1_score
from wafer.data import WM811K, CLASSES
from wafer.selfsup import _encoder, embed
dev = torch.device("cuda")
ds = WM811K(os.path.join(os.path.expanduser("~"), "wafer_cache"))
lab = ds.labeled_subset(0.05, seed=0)
SEED = sys.argv[2] if len(sys.argv) > 2 else "0"
C = os.path.join(os.path.expanduser("~"), "wafer_cache", "ver3", "runs", "resnet18", "p0.05", "seed" + SEED)
encs = [("ver3 (SimCLR)", "simclr_random_cut_e50.pt"), ("ver7 (평균→손실, 4장)", "masklat7_e50.pt"),
        ("ver7.2 (뷰별 손실, 4장)", "masklat72_e50.pt"), ("ver6.3 (평균→손실, 3장)", "masklat63_e50.pt")]
encs = [e for e in encs if os.path.exists(os.path.join(C, e[1]))]
MINOR = [CLASSES.index(c) for c in ("Near-full", "Donut", "Random", "Scratch", "Loc")]
S = CLASSES.index("Scratch")
y_l, y_v, y_t = (torch.as_tensor(ds.labels[i], device=dev) for i in (lab, ds.val, ds.test))
print(f"labeled {len(lab)} / val {len(ds.val)} / test {len(ds.test)}")

def scores(y, p):
    y, p = y.cpu().numpy(), p.cpu().numpy()
    f = f1_score(y, p, labels=range(len(CLASSES)), average=None, zero_division=0)
    return f.mean(), f[MINOR].mean(), f[S]

def knn(fl, fq, k=20, T=0.07):
    out = []
    for i in range(0, len(fq), 2048):
        s, j = (fq[i:i + 2048] @ fl.T).topk(k, dim=1)
        v = torch.zeros(len(s), len(CLASSES), device=dev).scatter_add_(1, y_l[j], (s / T).exp())
        out.append(v.argmax(1))
    return torch.cat(out)

def linear(fl, fv, ft, seed, epochs=200):
    torch.manual_seed(seed)
    W = torch.nn.Linear(fl.shape[1], len(CLASSES)).to(dev)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-2, weight_decay=1e-4)
    cw = 1.0 / torch.bincount(y_l, minlength=len(CLASSES)).float().clamp(min=1); cw = cw / cw.mean()
    best = (-1, None)
    for ep in range(epochs):
        for b in torch.randperm(len(fl), device=dev).split(512):
            loss = F.cross_entropy(W(fl[b]), y_l[b], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = scores(y_v, W(fv).argmax(1))[0]
            if v > best[0]: best = (v, W(ft).argmax(1))
    return best[0], scores(y_t, best[1])

for name, f in encs:
    enc, _ = _encoder("resnet18")
    enc.load_state_dict({k: v for k, v in torch.load(os.path.join(C, f), map_location="cpu", weights_only=True).items() if not k.startswith("fc.")})
    enc = enc.to(dev)
    fl, fv, ft = (embed(enc, ds, i, dev) for i in (lab, ds.val, ds.test))
    km, kmi, ks = scores(y_t, knn(fl, ft))
    lin = [linear(fl, fv, ft, s) for s in range(3)]
    lm = np.array([r[1] for r in lin])
    print(f"{name:26s} kNN macro {km:.3f} minor {kmi:.3f} Scratch {ks:.3f} | linear macro {lm[:, 0].mean():.3f}±{lm[:, 0].std():.3f} "
          f"minor {lm[:, 1].mean():.3f}±{lm[:, 1].std():.3f} Scratch {lm[:, 2].mean():.3f}±{lm[:, 2].std():.3f} (val {np.mean([r[0] for r in lin]):.3f})", flush=True)
