"""Детектор приступов: компактный 1D-CNN классификатор окон (приступ/межприступ)."""
import numpy as np
import torch, torch.nn as nn
from . import config as C

class Detector(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1)).squeeze(-1)

def train_detector(X, y, device="cpu", epochs=C.DET_EPOCHS, seed=C.SEED, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    n_ch = X.shape[1]
    det = Detector(n_ch).to(device)
    opt = torch.optim.Adam(det.parameters(), lr=C.DET_LR)
    # взвешивание классов (сильный дисбаланс приступ/межприступ)
    pos = max(int(y.sum()), 1); neg = max(len(y) - pos, 1)
    pw = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).float()
    idx = np.arange(len(X)); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        det.train(); rng.shuffle(idx); ep_loss = 0; nb = 0
        for k in range(0, len(idx), C.DET_BATCH):
            b = idx[k:k + C.DET_BATCH]
            logit = det(Xt[b].to(device))
            loss = lossf(logit, yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        if log: log(f"  det epoch {ep+1}/{epochs} loss={ep_loss/max(nb,1):.4f}")
    return det

@torch.no_grad()
def predict(det, X, device="cpu", bs=512):
    det.eval(); Xt = torch.from_numpy(X).float(); out = []
    for k in range(0, len(Xt), bs):
        out.append(torch.sigmoid(det(Xt[k:k+bs].to(device))).cpu().numpy())
    return np.concatenate(out)
