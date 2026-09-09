"""Сертифицированный разделитель выборки: протоколы P0, P1, P2.
P0 — оконное разбиение (перемешанные окна, УТЕЧКА).
P1 — субъектное разбиение + удаление дублей (без перекрытия пациентов).
P2 — субъектное + временное эмбарго + изоляция предобработки (без утечки).
"""
import numpy as np
from . import config as C

def window_split(n, seed=C.SEED, test_frac=0.3):
    """P0: перемешать все окна и разбить безотносительно субъекта."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    return idx[:cut], idx[cut:]

def subject_split(subjects_unique, seed=C.SEED, test_frac=0.3):
    """Разбить множество субъектов на train/eval (единица разбиения — субъект)."""
    rng = np.random.default_rng(seed)
    su = np.array(sorted(subjects_unique))
    perm = rng.permutation(len(su))
    cut = max(1, int(len(su) * (1 - test_frac)))
    return set(su[perm[:cut]].tolist()), set(su[perm[cut:]].tolist())

def embargo_mask(times, labels, train_mask, embargo_sec=C.EMBARGO_SEC):
    """Убрать из train окна, отстоящие от eval-окон менее чем на эмбарго
    (в пределах одного файла/субъекта). Здесь применяется по временной оси."""
    # times, labels — по одному субъекту; train_mask — булев
    keep = train_mask.copy()
    eval_t = times[~train_mask]
    if len(eval_t) == 0:
        return keep
    for i in np.where(train_mask)[0]:
        if np.any(np.abs(times[i] - eval_t) < embargo_sec):
            keep[i] = False
    return keep
