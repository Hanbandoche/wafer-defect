import json, os, sys
import numpy as np
proj, run = sys.argv[1], sys.argv[2]
os.chdir(proj)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(os.path.join("runs", "resnet18", "p0.05", "seed0", f"{run}_ssl50_ft30", "metrics.json"), encoding="utf-8"))
t = d["test"]
cls = list(t["per_class_f1"])
cm = np.array(t["confusion"])
rec = cm / cm.sum(1, keepdims=True)   # 행 정규화 = 실제 클래스 중 각 예측의 비율

print("rows = 실제, cols = 예측")
print(f"{'':>10s}" + "".join(f"{c[:9]:>10s}" for c in cls) + f"{'합계':>8s}{'재현율':>8s}")
for i, c in enumerate(cls):
    print(f"{c:>10s}" + "".join(f"{v:10d}" for v in cm[i]) + f"{cm[i].sum():8d}{rec[i, i]:8.3f}")
prec = cm.diagonal() / cm.sum(0)
print(f"{'정밀도':>9s}" + "".join(f"{v:10.3f}" for v in prec))

plt.rcParams["font.family"] = "Malgun Gothic"
fig, ax = plt.subplots(figsize=(9.5, 8))
ax.imshow(rec, cmap="Blues", vmin=0, vmax=1)
for i in range(len(cls)):
    for j in range(len(cls)):
        if cm[i, j]:
            ax.text(j, i, f"{cm[i, j]}\n{rec[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if rec[i, j] > 0.5 else "black")
ax.set_xticks(range(len(cls)), [f"{c}\n(정밀 {p:.2f})" for c, p in zip(cls, prec)], rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(cls)), [f"{c} ({cm[i].sum()})" for i, c in enumerate(cls)], fontsize=9)
ax.set_xlabel("예측"); ax.set_ylabel("실제 (테스트 장수)")
ax.set_title(f"{run} 테스트 혼동 행렬 — macro-F1 {t['macro_f1']:.3f}\n칸: 장수 / 실제 클래스 중 비율 (색 = 비율)", fontsize=12)
plt.tight_layout()
out = os.path.join("results", f"confusion_{run}.png")
plt.savefig(out, dpi=110)
print("saved", out)
