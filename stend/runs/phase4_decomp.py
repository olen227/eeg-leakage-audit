"""
Фаза 4 — аддитивное разложение завышения + бутстрэп по идентичностям
Выполнено на стенде Главы 3, env=eeg, seed=20260706.
Код восстановлен дословно из происхождения (lineage) артефакта-результата.
Зависит от модулей stend/vkr_eeg/*.py и кэша outputs/master_dataset.npz.
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import sys
sys.path.insert(0, os.getcwd())

import json
import time
import gc
import numpy as np
import torch
import mne

from stend.vkr_eeg import data, dataset, splitter, config as C
from stend.vkr_eeg import operator as OP, metrics as MET, detector as DET

mne.set_log_level("ERROR")
torch.set_num_threads(8)

subjects = ["chb01", "chb02", "chb03", "chb05", "chb08", "chb21"]

man = json.load(open(os.path.join(C.DATA_ROOT, "chbmit", "download_manifest.json")))

def subject_files(s):
    return man[s]["seizure"] + man[s]["interictal"]

# Subject-level split
train_subj, eval_subj = splitter.subject_split(subjects, seed=C.SEED, test_frac=0.34)

# Fit train stats
train_stats = dataset.fit_train_stats(train_subj, man)

# Build master dataset
X, y_sz, subj, fileid, wtime = dataset.build(subjects, train_stats, man)
dhash = dataset.data_hash(subjects, man)

# File-level split flags
rng_files = np.random.default_rng(C.SEED)
train_files = set()
eval_files = set()
for s in subjects:
    fs = sorted(set(fileid[subj == s]))
    perm = rng_files.permutation(len(fs))
    k = max(1, int(len(fs) * 0.6))
    for i in perm[:k]:
        train_files.add((s, fs[i]))
    for i in perm[k:]:
        eval_files.add((s, fs[i]))

is_train_file = np.array([(subj[i], fileid[i]) in train_files for i in range(len(X))])

def eval_f1(det, Xe, ye, thr=0.5):
    p = DET.predict(det, Xe, device="cpu")
    yhat = (p >= thr).astype(int)
    return MET.f1_binary(ye, yhat), p

# ---- P0: shuffled window split (LEAKY) ----
def run_P0(seed=C.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, ev = splitter.window_split(len(X), seed=seed, test_frac=0.3)
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr, ev), p

f1_p0, det0, (tr0, ev0), p0 = run_P0()

# ---- P1: file-level split ----
def run_P1(seed=C.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    tr_files = set()
    ev_files = set()
    for s in subjects:
        fs = sorted(set(fileid[subj == s]))
        perm = rng.permutation(len(fs))
        k = max(1, int(round(len(fs) * 0.7)))
        for i in perm[:k]:
            tr_files.add((s, fs[i]))
        for i in perm[k:]:
            ev_files.add((s, fs[i]))
    is_tr = np.array([(subj[i], fileid[i]) in tr_files for i in range(len(X))])
    tr = np.where(is_tr)[0]
    ev = np.where(~is_tr)[0]
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr, ev), p

f1_p1, det1, (tr1, ev1), p1_ = run_P1()

d_dubli = f1_p0 - f1_p1

# ---- P2: subject split (HONEST) ----
def run_P2(seed=C.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_s, ev_s = splitter.subject_split(subjects, seed=seed, test_frac=0.34)
    tr = np.where(np.isin(subj, sorted(tr_s)))[0]
    ev = np.where(np.isin(subj, sorted(ev_s)))[0]
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr, ev), p, (tr_s, ev_s)

f1_p2, det2, (tr2, ev2), p2, (trs, evs) = run_P2()

d_utech = f1_p1 - f1_p2

# Bootstrap CI for f1_p2
pred_prob_full = np.full(len(X), np.nan)
pred_prob_full[ev2] = p2

def bootstrap_ci(pred_prob, ev_idx, n=C.BOOTSTRAP_N, gamma=C.BOOTSTRAP_GAMMA, seed=C.SEED):
    rng = np.random.default_rng(seed)
    ev_subj = subj[ev_idx]
    uniq = np.unique(ev_subj)
    vals = []
    for _ in range(n):
        chosen = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([ev_idx[ev_subj == s] for s in chosen])
        yhat = (pred_prob[idx] >= 0.5).astype(int)
        if y_sz[idx].sum() == 0:
            continue
        vals.append(MET.f1_binary(y_sz[idx], yhat))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [100 * gamma / 2, 100 * (1 - gamma / 2)])
    return float(vals.mean()), float(lo), float(hi), len(vals)

mean_f1, lo, hi, nboot = bootstrap_ci(pred_prob_full, ev2, n=C.BOOTSTRAP_N, gamma=C.BOOTSTRAP_GAMMA)

# Bootstrap decomposition
prob_p0 = np.full(len(X), np.nan)
prob_p0[ev0] = p0
prob_p1 = np.full(len(X), np.nan)
prob_p1[ev1] = p1_
prob_p2 = np.full(len(X), np.nan)
prob_p2[ev2] = p2

ev2_subs = np.unique(subj[ev2])

def f1_boot_protocol(prob, ev_idx, chosen):
    parts = [ev_idx[subj[ev_idx] == s] for s in chosen if (subj[ev_idx] == s).any()]
    if not parts:
        return np.nan
    idx = np.concatenate(parts)
    if y_sz[idx].sum() == 0:
        return np.nan
    return MET.f1_binary(y_sz[idx], (prob[idx] >= 0.5).astype(int))

rng = np.random.default_rng(C.SEED)
dd = []
du = []
for _ in range(C.BOOTSTRAP_N):
    chosen = rng.choice(ev2_subs, len(ev2_subs), replace=True)
    a = f1_boot_protocol(prob_p0, ev0, chosen)
    b = f1_boot_protocol(prob_p1, ev1, chosen)
    c = f1_boot_protocol(prob_p2, ev2, chosen)
    if np.isnan([a, b, c]).any():
        continue
    dd.append(a - b)
    du.append(b - c)

dd = np.array(dd)
du = np.array(du)

def ci(v):
    return round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)

dd_m, dd_lo, dd_hi = ci(dd)
du_m, du_lo, du_hi = ci(du)

decomp = {
    "f1_zayavlennyy": round(float(f1_p0), 4),
    "f1_P1_file_split": round(float(f1_p1), 4),
    "f1_bez_utechki": round(float(f1_p2), 4),
    "vklad_dubl_perekrytiya": round(float(d_dubli), 4),
    "vklad_ostatochnoj_utechki": round(float(d_utech), 4),
    "delta_total": round(float(f1_p0 - f1_p2), 4),
    "additivity_check_ok": bool(abs((d_dubli + d_utech) - (f1_p0 - f1_p2)) < 1e-9),
    "bootstrap_ci_subjekty": {
        "f1_bez_utechki": {"mean": round(mean_f1, 4), "lo": round(lo, 4), "hi": round(hi, 4)},
        "vklad_dubl_perekrytiya": {"mean": dd_m, "lo": dd_lo, "hi": dd_hi},
        "vklad_ostatochnoj_utechki": {"mean": du_m, "lo": du_lo, "hi": du_hi},
        "n_bootstrap": C.BOOTSTRAP_N,
        "gamma": C.BOOTSTRAP_GAMMA,
        "n_eval_subjects": int(len(ev2_subs)),
        "eval_subjects": list(map(str, ev2_subs)),
        "note": "Широкие интервалы обусловлены малым числом оценочных субъектов (3) — ограничение подмножества без GPU."
    },
    "formulas": {
        "delta_dubli": "M(P0)-M(P1) (2.11)",
        "delta_utechka": "M(P1)-M(P2) (2.12)",
        "delta": "M(P0)-M(P2) (2.13)"
    },
    "seed": C.SEED,
    "data_sha256": dhash,
}

os.makedirs("outputs", exist_ok=True)
json.dump(decomp, open("outputs/decomposition.json", "w"), ensure_ascii=False, indent=2)
print("saved decomposition.json")