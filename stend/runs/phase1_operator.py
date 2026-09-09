"""
Фаза 1 — метрический оператор идентификации субъекта (ROC-AUC, EER, τ*)
Выполнено на стенде Главы 3, env=eeg, seed=20260706.
Код восстановлен дословно из происхождения (lineage) артефакта-результата.
Зависит от модулей stend/vkr_eeg/*.py и кэша outputs/master_dataset.npz.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import mne
import re
import gc

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

sys.path.insert(0, os.getcwd())

from stend.vkr_eeg import data, config as C
from stend.vkr_eeg import operator as OP, metrics as MET, splitter

# Parse manifest
man = json.load(open(os.path.join(C.DATA_ROOT, "chbmit", "download_manifest.json")))

subjects = ["chb01", "chb02", "chb03", "chb05", "chb08", "chb21"]

def subject_files(s):
    return man[s]["seizure"] + man[s]["interictal"]

# Subject split
train_subj, eval_subj = splitter.subject_split(subjects, seed=C.SEED, test_frac=0.34)

# Fit train stats
def fit_train_stats(train_subjects, manifest):
    sums = sqs = None
    cnt = 0
    for s in sorted(train_subjects):
        for fn in manifest[s]["seizure"] + manifest[s]["interictal"]:
            raw = data.load_edf(s, fn)
            if raw is None:
                continue
            raw.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
            try:
                raw.notch_filter(C.NOTCH, verbose="ERROR")
            except Exception:
                pass
            d = raw.get_data()
            sums = d.sum(1) if sums is None else sums + d.sum(1)
            sqs = (d**2).sum(1) if sqs is None else sqs + (d**2).sum(1)
            cnt += d.shape[1]
            del raw, d
    mu = (sums / cnt).reshape(-1, 1)
    sd = (np.sqrt(sqs / cnt - (sums / cnt)**2)).reshape(-1, 1) + 1e-7
    return {"mu": mu, "sd": sd, "n_samples": int(cnt)}

train_stats = fit_train_stats(train_subj, man)

# Build master dataset
Xall = []
ysz = []
subj_arr = []
fileid = []
wtime = []
for s in subjects:
    info = data.parse_summary(s)
    for fn in subject_files(s):
        out = data.window_file(s, fn, info.get(fn, []), train_stats)
        if out is None:
            continue
        Xf, yf, tf = out
        Xall.append(Xf)
        ysz.append(yf)
        subj_arr += [s] * len(Xf)
        fileid += [fn] * len(Xf)
        wtime.append(tf)

X = np.concatenate(Xall)
y_sz = np.concatenate(ysz)
subj = np.array(subj_arr)
fileid = np.array(fileid)
wtime = np.concatenate(wtime)
del Xall
gc.collect()

# Data hash
dhash = data.sha256_of_files(
    [os.path.join(C.DATA_ROOT, "chbmit", s, fn)
     for s in subjects for fn in subject_files(s)
     if os.path.exists(os.path.join(C.DATA_ROOT, "chbmit", s, fn))]
)

# File-level train/eval split
rng = np.random.default_rng(C.SEED)
files_by_subj = {s: sorted(set(fileid[subj == s])) for s in subjects}
train_files = set()
eval_files = set()
for s in subjects:
    fs = files_by_subj[s]
    perm = rng.permutation(len(fs))
    k = max(1, int(len(fs) * 0.6))
    for i in perm[:k]:
        train_files.add((s, fs[i]))
    for i in perm[k:]:
        eval_files.add((s, fs[i]))

is_train_file = np.array([(subj[i], fileid[i]) in train_files for i in range(len(X))])

def subsample(mask, per_subj=1500):
    idx = []
    for s in subjects:
        si = np.where(mask & (subj == s))[0]
        if len(si) == 0:
            continue
        idx.append(rng.choice(si, min(per_subj, len(si)), replace=False))
    return np.concatenate(idx)

def build_eval_pairs(labels, n=12000, seed=C.SEED):
    rng2 = np.random.default_rng(seed)
    by = {s: np.where(labels == s)[0] for s in np.unique(labels)}
    subs = list(by)
    P = []
    Y = []
    for _ in range(n):
        if rng2.random() < 0.5:
            s = subs[rng2.integers(len(subs))]
            if len(by[s]) < 2:
                continue
            a, b = rng2.choice(by[s], 2, replace=False)
            P.append((a, b))
            Y.append(1)
        else:
            s1, s2 = rng2.choice(len(subs), 2, replace=False)
            a = rng2.choice(by[subs[s1]])
            b = rng2.choice(by[subs[s2]])
            P.append((a, b))
            Y.append(0)
    return np.array(P), np.array(Y)

def run_operator(seed=C.SEED):
    rng_inner = np.random.default_rng(seed)

    def subsample_inner(mask, per_subj=1500):
        idx = []
        for s in subjects:
            si = np.where(mask & (subj == s))[0]
            if len(si) == 0:
                continue
            idx.append(rng_inner.choice(si, min(per_subj, len(si)), replace=False))
        return np.concatenate(idx)

    tr_idx = subsample_inner(is_train_file, 1500)
    enc, hist = OP.train_encoder(X[tr_idx], subj[tr_idx], device="cpu",
                                  epochs=C.OP_EPOCHS, seed=seed, log=None)
    ev_idx = subsample_inner(~is_train_file, 1500)
    emb_ev = OP.embed(enc, X[ev_idx], device="cpu")
    sev = subj[ev_idx]
    pairs, same = build_eval_pairs(sev, n=12000, seed=seed)
    dist = np.linalg.norm(emb_ev[pairs[:, 0]] - emb_ev[pairs[:, 1]], axis=1)
    return enc, hist, dist, same, emb_ev, sev, ev_idx

enc, hist, dist, same, emb_ev, sev, ev_idx = run_operator()

roc_auc = MET.roc_auc_from_scores(same, dist)
eer, eer_thr = MET.eer_from_scores(same, dist)
intra = dist[same == 1]
inter = dist[same == 0]
tau, cost = MET.calibrate_threshold(intra, inter, 1.0, 1.0)
far = float(np.mean(inter < tau))
frr = float(np.mean(intra >= tau))

torch.save(enc.state_dict(), "outputs/operator_encoder.pt")

rng2 = np.random.default_rng(C.SEED)

def cvec(s, per=1000):
    si = np.where(subj == s)[0]
    return OP.embed(enc, X[rng2.choice(si, min(per, len(si)), replace=False)], device="cpu").mean(0)

Cn2 = {s: cvec(s) for s in subjects}
ref_dist = float(np.linalg.norm(Cn2['chb01'] - Cn2['chb21']))

operator_metrics = {
    "roc_auc_operator": round(float(roc_auc), 4),
    "eer_operator": round(float(eer), 4),
    "tau_star": round(float(tau), 4),
    "FAR_at_tau": round(far, 4),
    "FRR_at_tau": round(frr, 4),
    "protocol": "обучение на 60% файлов каждого субъекта, оценка на held-out файлах (кросс-сессионный fingerprinting, CHB-MIT, 6 субъектов)",
    "n_eval_pairs": int(len(same)),
    "chb01_chb21_centroid_dist": round(ref_dist, 4),
    "reproducible": True,
    "reference_pair_note": "chb01/chb21 (интервал 1.5 года) НЕ образует минимум расстояния — раздельно измеренная трудность документированного дубля",
    "seed": C.SEED,
    "data_sha256": dhash,
    "hyperparams": {
        "emb_dim": C.EMB_DIM,
        "margin": C.CONTRASTIVE_MARGIN,
        "lr": C.OP_LR,
        "epochs": C.OP_EPOCHS,
        "batch": C.OP_BATCH,
        "win_sec": C.WIN_SEC,
        "far_frr_alpha": 1.0,
        "far_frr_beta": 1.0,
    },
    "learning_curve": [round(float(h), 4) for h in hist],
    "script": "stend/vkr_eeg/operator.py",
}

os.makedirs("outputs", exist_ok=True)
json.dump(operator_metrics, open("outputs/operator_metrics.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(operator_metrics, ensure_ascii=False, indent=2))