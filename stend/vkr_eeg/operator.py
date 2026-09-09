"""Метрический оператор идентификации субъекта.
Контрастивный 1D-CNN энкодер Φ_ϑ окна сырого сигнала (формула 2.8),
контрастивная потеря (2.9), калибровка порога по chb01/chb21 (2.10)."""
import numpy as np
import torch, torch.nn as nn
from . import config as C

class Encoder(nn.Module):
    """Φ_ϑ: окно (n_ch, win) -> эмбеддинг R^d. Компактный, под CPU."""
    def __init__(self, n_ch, emb_dim=C.EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, emb_dim)
    def forward(self, x):
        h = self.net(x).squeeze(-1)
        z = self.fc(h)
        return z

def contrastive_loss(z_i, z_j, y_ij, margin=C.CONTRASTIVE_MARGIN):
    """Формула (2.9): y*||z_i-z_j||^2 + (1-y)*[m-||z_i-z_j||]_+^2."""
    d = torch.norm(z_i - z_j, dim=1)
    pos = y_ij * d.pow(2)
    neg = (1 - y_ij) * torch.clamp(margin - d, min=0).pow(2)
    return (pos + neg).mean()

def make_pairs(emb_idx, subj_labels, n_pairs, rng):
    """Позитивные пары — окна одного субъекта, негативные — разных."""
    subj_labels = np.asarray(subj_labels)
    idx_by_s = {s: np.where(subj_labels == s)[0] for s in np.unique(subj_labels)}
    subs = list(idx_by_s.keys())
    pairs, ys = [], []
    for _ in range(n_pairs):
        if rng.random() < 0.5:  # позитив
            s = subs[rng.integers(len(subs))]
            pool = idx_by_s[s]
            if len(pool) < 2: continue
            a, b = rng.choice(pool, 2, replace=False)
            pairs.append((a, b)); ys.append(1.0)
        else:                    # негатив
            s1, s2 = rng.choice(len(subs), 2, replace=False)
            a = rng.choice(idx_by_s[subs[s1]]); b = rng.choice(idx_by_s[subs[s2]])
            pairs.append((a, b)); ys.append(0.0)
    return np.array(pairs), np.array(ys, dtype=np.float32)

def train_encoder(X, subj_labels, device="cpu", epochs=C.OP_EPOCHS, seed=C.SEED, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n_ch = X.shape[1]
    enc = Encoder(n_ch).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=C.OP_LR)
    Xt = torch.from_numpy(X).float()
    history = []
    for ep in range(epochs):
        enc.train()
        pairs, ys = make_pairs(np.arange(len(X)), subj_labels, C.OP_BATCH * 20, rng)
        ep_loss = 0.0; nb = 0
        for k in range(0, len(pairs), C.OP_BATCH):
            pb = pairs[k:k + C.OP_BATCH]; yb = ys[k:k + C.OP_BATCH]
            zi = enc(Xt[pb[:, 0]].to(device)); zj = enc(Xt[pb[:, 1]].to(device))
            loss = contrastive_loss(zi, zj, torch.from_numpy(yb).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        history.append(ep_loss / max(nb, 1))
        if log: log(f"  epoch {ep+1}/{epochs} loss={history[-1]:.4f}")
    return enc, history

@torch.no_grad()
def embed(enc, X, device="cpu", bs=512):
    enc.eval()
    Xt = torch.from_numpy(X).float()
    out = []
    for k in range(0, len(Xt), bs):
        out.append(enc(Xt[k:k+bs].to(device)).cpu().numpy())
    return np.concatenate(out)
