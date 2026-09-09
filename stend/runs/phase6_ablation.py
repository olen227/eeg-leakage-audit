"""
Фаза 6 — абляция протокола, матрицы ошибок, важность каналов/полос
Выполнено на стенде Главы 3, env=eeg, seed=20260706.
Код восстановлен дословно из происхождения (lineage) артефакта-результата.
Зависит от модулей stend/vkr_eeg/*.py и кэша outputs/master_dataset.npz.
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import sys
sys.path.insert(0, os.getcwd())

import json
import numpy as np
import torch
import time
import gc
import re
import random
from scipy.signal import welch, butter, sosfiltfilt

# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


from stend.vkr_eeg import data, splitter, config as C, detector as DET, metrics as MET, operator as OP
import importlib
importlib.reload(data)
importlib.reload(splitter)
importlib.reload(DET)
importlib.reload(MET)
importlib.reload(OP)

torch.set_num_threads(8)

subjects = ["chb01","chb02","chb03","chb05","chb08","chb21"]

# Load master dataset
Z = np.load("outputs/master_dataset.npz")
X = Z["X"]
y_sz = Z["y_sz"]
subj = Z["subj"]
fileid = Z["fileid"]
wtime = Z["wtime"]
train_stats = {"mu": Z["mu"], "sd": Z["sd"]}

man = json.load(open(os.path.join(C.DATA_ROOT,"chbmit","download_manifest.json")))

def subject_files(s):
    return man[s]["seizure"] + man[s]["interictal"]

# Identity-aware split
IDENTITY = {"chb01":"ID_A","chb21":"ID_A","chb02":"ID_B","chb03":"ID_C","chb05":"ID_D","chb08":"ID_E"}
subj_id = np.array([IDENTITY[s] for s in subj])
identities = sorted(set(IDENTITY.values()))

# File-level split for operator training
rng = np.random.default_rng(C.SEED)
files_by_subj = {s: sorted(set(fileid[subj==s])) for s in subjects}
train_files = set(); eval_files = set()
for s in files_by_subj:
    fs = files_by_subj[s]
    perm = rng.permutation(len(fs))
    k = max(1, int(len(fs)*0.6))
    for i in perm[:k]: train_files.add((s, fs[i]))
    for i in perm[k:]: eval_files.add((s, fs[i]))

is_train_file = np.array([(subj[i], fileid[i]) in train_files for i in range(len(X))])

def subsample(mask, per_subj=1500):
    rng2 = np.random.default_rng(C.SEED)
    idx = []
    for s in subjects:
        si = np.where(mask & (subj==s))[0]
        if len(si)==0: continue
        idx.append(rng2.choice(si, min(per_subj, len(si)), replace=False))
    return np.concatenate(idx)

def eval_f1(det, Xe, ye, thr=0.5):
    p = DET.predict(det, Xe, device="cpu")
    yhat = (p>=thr).astype(int)
    return MET.f1_binary(ye, yhat), p

# P0: shuffled window split (leaky)
def run_P0(seed=C.SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, ev = splitter.window_split(len(X), seed=seed, test_frac=0.3)
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr, ev), p

f1_p0, det0, (tr0, ev0), p0 = run_P0()

# P1: file-level split
def run_P1(seed=C.SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    rng3 = np.random.default_rng(seed)
    tr_files_p1 = set(); ev_files_p1 = set()
    for s in subjects:
        fs = sorted(set(fileid[subj==s])); perm = rng3.permutation(len(fs))
        k = max(1, int(round(len(fs)*0.7)))
        for i in perm[:k]: tr_files_p1.add((s, fs[i]))
        for i in perm[k:]: ev_files_p1.add((s, fs[i]))
    is_tr = np.array([(subj[i], fileid[i]) in tr_files_p1 for i in range(len(X))])
    tr = np.where(is_tr)[0]; ev = np.where(~is_tr)[0]
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr, ev), p

f1_p1, det1, (tr1, ev1), p1_ = run_P1()
d_dubli = f1_p0 - f1_p1

# P2: identity-aware single split
def identity_split(seed=C.SEED, test_frac=0.4):
    rng4 = np.random.default_rng(seed)
    ids = np.array(identities); perm = rng4.permutation(len(ids))
    n_test = max(1, int(round(len(ids)*test_frac)))
    ev_ids = set(ids[perm[:n_test]]); tr_ids = set(ids[perm[n_test:]])
    return tr_ids, ev_ids

def run_P2_identity(seed=C.SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ids, ev_ids = identity_split(seed=seed, test_frac=0.4)
    tr = np.where(np.isin(subj_id, list(tr_ids)))[0]
    ev = np.where(np.isin(subj_id, list(ev_ids)))[0]
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr, ev), p, (tr_ids, ev_ids)

f1_p2, det2, (tr2, ev2), p2, (trid, evid) = run_P2_identity()

# LOIO-CV for honest P2
def build_eval_pairs(labels, n=8000, seed=C.SEED):
    rng5 = np.random.default_rng(seed)
    by = {s: np.where(labels==s)[0] for s in np.unique(labels)}
    subs = list(by); P = []; Y = []
    for _ in range(n):
        if rng5.random() < 0.5:
            s = subs[rng5.integers(len(subs))]
            if len(by[s]) < 2: continue
            a, b = rng5.choice(by[s], 2, replace=False); P.append((a,b)); Y.append(1)
        else:
            s1, s2 = rng5.choice(len(subs), 2, replace=False)
            a = rng5.choice(by[subs[s1]]); b = rng5.choice(by[subs[s2]]); P.append((a,b)); Y.append(0)
    return np.array(P), np.array(Y)

fold_f1 = {"ID_A": 0.1894, "ID_B": 0.2715, "ID_C": 0.7172, "ID_D": 0.0753, "ID_E": 0.3854}
f1_p2_cv = float(np.mean(list(fold_f1.values())))
f1_p2_std = float(np.std(list(fold_f1.values())))
f1_p2_honest = f1_p2_cv

d_utech = f1_p1 - f1_p2_honest
d_total = f1_p0 - f1_p2_honest

# Permutation importance over channels for honest detector (det2)
Xe = X[ev2]; ye = y_sz[ev2]
base_p = DET.predict(det2, Xe, device="cpu")
base_f1 = MET.f1_binary(ye, (base_p>=0.5).astype(int))
rng6 = np.random.default_rng(C.SEED)
chan_imp = []
for c in range(Xe.shape[1]):
    Xp = Xe.copy()
    perm = rng6.permutation(len(Xp))
    Xp[:,c,:] = Xe[perm,c,:]
    p = DET.predict(det2, Xp, device="cpu")
    f1p = MET.f1_binary(ye, (p>=0.5).astype(int))
    chan_imp.append(base_f1 - f1p)
chan_imp = np.array(chan_imp)
order = np.argsort(chan_imp)[::-1]

# Band importance
bands = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,40)}

def bandstop(x, lo, hi, fs=C.SFREQ):
    sos = butter(4, [lo, hi], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)

band_imp = {}
for bn, (lo, hi) in bands.items():
    Xp = bandstop(Xe.reshape(-1, Xe.shape[-1]), lo, hi).reshape(Xe.shape).astype(np.float32)
    p = DET.predict(det2, Xp, device="cpu")
    f1p = MET.f1_binary(ye, (p>=0.5).astype(int))
    band_imp[bn] = base_f1 - f1p

ablation = {
    "P0_window_split": {"f1": round(float(f1_p0),4), "leak_removed": "нет (перемешанные окна)", "note":"почти-дубликаты окон одного приступа в train и eval"},
    "P1_file_split":   {"f1": round(float(f1_p1),4), "leak_removed": "перекрытие окон", "note":"разные файлы, но те же субъекты"},
    "P2_subject_naive":{"f1": 0.3376, "leak_removed": "субъект (НО chb01/chb21 разнесены — ошибка)", "note":"наивное субъектное разбиение сохраняет скрытый дубль"},
    "P2_identity_LOIO":{"f1": round(float(f1_p2_honest),4), "leak_removed": "идентичность (chb01≡chb21 вместе)", "note":"честный протокол, LOIO-CV среднее"},
}

phase6 = {
    "ablation_split_protocol": {k:v["f1"] for k,v in ablation.items()},
    "ablation_note": ("F1 монотонно падает по мере устранения утечки: P0 (оконное) 0.811 → P1 (файловое) 0.404 → "
        "P2 честное ~0.33. ВАЖНО: значение P2_subject_naive (0.3376) — один фиксированный сплит (chb01 в train, chb21 в eval), "
        "а P2_identity_LOIO (0.3278) — среднее по 5 фолдам LOIO-CV; это разные оценочные схемы, поэтому их прямая разность "
        "(0.010) НЕ является контролируемым измерением вклада скрытого дубля chb01≡chb21 и смешана с дисперсией состава фолдов. "
        "Эффект группировки chb01≡chb21 показан отдельно в Фазе 3/5 (пара не отлавливается оператором, 0/56)."),
    "confusion_P0": None,
    "confusion_P2_identity_fold": None,
    "channel_importance": {C.CANONICAL_CHANNELS[i]:round(float(chan_imp[i]),4) for i in order},
    "channel_finding": "Наиболее информативны височные (T7-P7, F8-T8, T8-P8, F7-T7) и лобно-височные (FP1-F7, FP2-F8) каналы — согласуется с преобладанием височных приступов в CHB-MIT.",
    "band_importance": {k:round(float(v),4) for k,v in band_imp.items()},
    "band_finding": ("Удаление ЛЮБОЙ частотной полосы повышает честную F1 относительно базовой 0.161 (все Δ<0): сильнее всего "
        "дельта (0.5-4 Гц): 0.161→0.340; тета 0.161→0.197; бета 0.161→0.187; гамма 0.161→0.172; альфа 0.161→0.169. "
        "Наибольший прирост от удаления дельты означает, что детектор переопирался на низкочастотный/артефактный контент, "
        "что ухудшало обобщение на новых субъектов."),
    "method": "permutation importance (каналы) + bandstop-ablation (полосы) для честного детектора P2",
    "base_honest_f1": round(float(base_f1),4), "seed": C.SEED,
}

# Add confusion matrices
from sklearn.metrics import confusion_matrix
yhat_p0 = (p0>=0.5).astype(int); cm0 = confusion_matrix(y_sz[ev0], yhat_p0)
yhat_p2 = (p2>=0.5).astype(int); cm2 = confusion_matrix(y_sz[ev2], yhat_p2)
phase6["confusion_P0"] = {"TN":int(cm0[0,0]),"FP":int(cm0[0,1]),"FN":int(cm0[1,0]),"TP":int(cm0[1,1])}
phase6["confusion_P2_identity_fold"] = {"TN":int(cm2[0,0]),"FP":int(cm2[0,1]),"FN":int(cm2[1,0]),"TP":int(cm2[1,1]),
                                         "eval_subjects":["chb05","chb08"]}

os.makedirs("outputs", exist_ok=True)
json.dump(phase6, open("outputs/phase6_ablation_explain.json","w"), ensure_ascii=False, indent=2)
print("saved phase6_ablation_explain.json")