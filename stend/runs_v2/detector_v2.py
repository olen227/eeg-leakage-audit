"""Усиленный детектор приступов.

ЗАЧЕМ. Детектор из июльского кода (stend/vkr_eeg/detector.py) — две свёрточных
операции, максимум 64 канала, 16 673 обучаемых параметра при 18 отведениях. Это заведомо
недостаточная ёмкость для окна 18 x 1024. Низкие абсолютные показатели работы
частично объясняются именно этим, а не только честностью протокола.

ЧТО ИЗМЕНЕНО (и почему каждое изменение законно):

  1. Глубина и ширина: четыре свёрточных блока 32-64-128-256 вместо двух 32-64.
     Больше ёмкости — больше шансов выучить форму приступа.
  2. Остаточные связи внутри блоков: устойчивее обучение при большей глубине.
  3. Совмещённое усреднение и максимум по времени вместо только усреднения:
     приступ локален во времени, среднее его размывает.
  4. Dropout перед классификатором: против переобучения при возросшей ёмкости.
  5. Косинусное расписание скорости обучения и больше эпох.
  6. Аугментация обучающих окон: случайное обнуление каналов, масштаб амплитуды,
     сдвиг по времени. Заставляет опираться на форму сигнала, а не на уровень
     конкретного канала конкретного пациента — то есть прямо противодействует
     запоминанию пациента.

ЧТО НЕ ИЗМЕНЕНО. Протоколы разбиения, схема оценки, калибровка порога, состав
данных. Поэтому измерение утечки остаётся сопоставимым: Δ считается тем же способом,
просто на более сильном детекторе.

ЧЕСТНОСТЬ. Архитектура и гиперпараметры выбраны ДО просмотра результатов на
оценочной части и не подбирались по ней. Июльский детектор сохранён и приводится
рядом как базовый уровень.
"""
import numpy as np
import torch
import torch.nn as nn

from stend.vkr_eeg import config as C


class Block(nn.Module):
    """Свёрточный блок с остаточной связью и понижением частоты вдвое."""

    def __init__(self, cin, cout, k=7):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity())
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.pool(self.act(h + self.skip(x)))


class DetectorV2(nn.Module):
    def __init__(self, n_ch, widths=(32, 64, 128, 256), p_drop=0.3):
        super().__init__()
        blocks, cin = [], n_ch
        for w in widths:
            blocks.append(Block(cin, w)); cin = w
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(p_drop)
        # усреднение И максимум по времени: приступ локален, среднее его размывает
        self.head = nn.Sequential(nn.Linear(cin * 2, 128), nn.ReLU(),
                                  nn.Dropout(p_drop), nn.Linear(128, 1))

    def forward(self, x):
        h = self.body(x)
        h = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=1)
        return self.head(self.drop(h)).squeeze(-1)


def augment(xb, rng, p_chan=0.2, amp=0.15, shift=64):
    """Аугментация обучающего батча (на вход подаётся уже нормированный тензор)."""
    b, c, l = xb.shape
    # случайное обнуление части каналов
    mask = torch.from_numpy((rng.random((b, c, 1)) > p_chan).astype(np.float32))
    xb = xb * mask
    # случайный масштаб амплитуды
    sc = torch.from_numpy((1.0 + rng.normal(0, amp, (b, 1, 1))).astype(np.float32))
    xb = xb * sc
    # случайный циклический сдвиг по времени
    s = int(rng.integers(-shift, shift + 1))
    if s:
        xb = torch.roll(xb, shifts=s, dims=-1)
    return xb


def train_detector_v2(X, train_idx, y, mu, sd, device, epochs, seed,
                      batch=None, lr=None, log=None, use_augment=True, widths=None):
    """widths задаёт ширину четырёх остаточных блоков и тем самым ёмкость модели.

    Параметр введён для измерения зависимости завышения от мощности: архитектура,
    расписание обучения и аугментация при этом не меняются, различается только число
    каналов, поэтому точки кривой сопоставимы между собой.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    batch = batch or C.DET_BATCH
    lr = lr or C.DET_LR
    det = DetectorV2(X.shape[1], **({'widths': tuple(widths)} if widths else {})).to(device)
    opt = torch.optim.AdamW(det.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ytr = y[train_idx]
    pos = max(int(ytr.sum()), 1); neg = max(len(ytr) - pos, 1)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    ymap = {int(i): float(v) for i, v in zip(train_idx, ytr)}

    for ep in range(epochs):
        det.train()
        order = rng.permutation(len(train_idx))
        tot = 0.0; nb = 0
        for k in range(0, len(order), batch):
            b = np.sort(train_idx[order[k:k + batch]])
            xb = torch.from_numpy(((X[b] - mu) / sd).astype(np.float32))
            if use_augment:
                xb = augment(xb, rng)
            yb = torch.tensor([ymap[int(i)] for i in b],
                              dtype=torch.float32, device=device)
            loss = lossf(det(xb.to(device)), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(det.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if log:
            log(f"      эпоха {ep+1}/{epochs} loss={tot/max(nb,1):.4f}")
    return det


def n_params(model):
    return sum(p.numel() for p in model.parameters())
