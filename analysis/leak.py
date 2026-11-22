"""마스킹본끼리 정보가 새는 경로 검증.
1) 구조: 각 마스킹본은 인코더를 따로 통과 (배치 차원으로 펼침) → 픽셀을 서로 볼 수 없음. 유일한 연결 = 학습 모드 BatchNorm 의 배치 통계.
2) BN 누출 측정: 마스킹본 1 의 특징을 (a) 같은 웨이퍼의 다른 마스킹본 3장과 한 배치 (사전학습과 동일) vs
   (b) 다른 3장을 '다른 웨이퍼'의 마스킹본으로 바꾼 배치에서 계산 → 차이가 크면 BN 으로 옆 마스킹본 정보가 들어온 것.
3) 4장 합치면 원본의 몇 %가 보이는지 (회전 없이 바둑판만, 근사)."""
import copy, os, sys
import numpy as np, torch, torch.nn.functional as F
proj = sys.argv[1]; os.chdir(proj); sys.path.insert(0, proj)
from wafer import augment as A
from wafer.data import WM811K
from wafer.models import to_input
from wafer.selfsup import _encoder
dev = torch.device("cuda")
cache = os.path.join(os.path.expanduser("~"), "wafer_cache")
ds = WM811K(cache)
enc, _ = _encoder("resnet18")
enc.load_state_dict({k: v for k, v in torch.load(os.path.join(cache, "ver3/runs/resnet18/p0.05/seed0/masklat7_e50.pt"), map_location="cpu", weights_only=True).items() if not k.startswith("fc.")})
enc = enc.to(dev).train()                     # 사전학습과 같은 학습 모드 BN (배치 통계)
for m in enc.modules():
    if isinstance(m, torch.nn.BatchNorm2d): m.momentum = 0.0   # running 통계는 건드리지 않음
B = 128
rng = np.random.default_rng(0)
idx = rng.choice(ds.test, 3 * B, replace=False)
torch.manual_seed(0)
x = ds.fetch(idx, dev).float()
views = torch.stack([A.checker_mask(A.rotate(x), 6, 0.5) for _ in range(4)])   # (4, 2B, 1, H, W)
own = views[:, :B]                               # 웨이퍼 0..B-1 의 마스킹본 4장
other = views[:, B:2 * B]; other2 = views[:, 2 * B:]   # 다른 웨이퍼 두 묶음
torch.manual_seed(1); tgt = A.rotate(x[:B])                # 타깃 (다른 회전, 안 가림)
with torch.no_grad():
    fa = F.normalize(enc(to_input(own.flatten(0, 1))), dim=1).view(4, B, -1)[0]                       # (a) 사전학습 그대로
    fb = F.normalize(enc(to_input(torch.cat([own[:1], other[1:]]).flatten(0, 1))), dim=1).view(4, B, -1)[0]  # (b) 옆 3장을 남의 것으로
    fc = F.normalize(enc(to_input(torch.cat([own[:1], other2[1:]]).flatten(0, 1))), dim=1).view(4, B, -1)[0]  # (c) 다른 남의 것으로
    ft = F.normalize(enc(to_input(torch.cat([tgt, x[B:2 * B], x[B:2 * B], x[B:2 * B]]))), dim=1)[:B]       # 타깃 특징 (같은 BN 조건)
    enc.eval()
    fe = F.normalize(enc(to_input(own[0])), dim=1)                                                     # 참고: BN 고정 (혼자)
    fo = F.normalize(enc(to_input(other[0])), dim=1)
print(f"마스킹본1 특징 cos (a) vs (b) [옆 3장 교체]: {F.cosine_similarity(fa, fb).mean():.6f}  (1 = 전혀 영향 없음)")
print(f"대조군 (b) vs (c) [남의 것 두 묶음끼리]: {F.cosine_similarity(fb, fc).mean():.6f}")
print(f"마스킹본1 ↔ 자기 타깃 cos: (a) 형제와 한 배치 {F.cosine_similarity(fa, ft).mean():.4f} / (b) 남과 한 배치 {F.cosine_similarity(fb, ft).mean():.4f} / (c) {F.cosine_similarity(fc, ft).mean():.4f}")
print(f"자기 타깃 찾기 top-1 (128장 중): (a) {((fa @ ft.T).argmax(1) == torch.arange(B, device=dev)).float().mean():.1%} / (b) {((fb @ ft.T).argmax(1) == torch.arange(B, device=dev)).float().mean():.1%}")
print(f"참고 - 같은 마스킹본 학습모드 vs 고정 BN: {F.cosine_similarity(fa, fe).mean():.4f} / 서로 다른 웨이퍼끼리 평균 cos: {(fe @ fo.T).mean():.4f}")
# 3) 바둑판 4장 합집합 커버리지 (회전 없이, 시작 위치만 무작위)
cov = []
for _ in range(2000):
    vis = np.zeros((96, 96), bool)
    for _ in range(4):
        oy, ox = rng.integers(0, 12, 2)
        yy, xx = np.meshgrid(np.arange(96) + oy, np.arange(96) + ox, indexing="ij")
        vis |= ((yy // 6 + xx // 6) % 2) == 1
    cov.append(vis.mean())
print(f"마스킹본 1장이 보는 비율 50% / 4장 합치면 평균 {np.mean(cov):.1%} 보임")

