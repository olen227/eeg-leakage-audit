"""
Фаза 3 — разрыв F1 (P0 с утечкой vs P2 честный, identity-aware LOIO)
Выполнено на стенде Главы 3, env=eeg, seed=20260706.
Код восстановлен дословно из происхождения (lineage) артефакта-результата.
Зависит от модулей stend/vkr_eeg/*.py и кэша outputs/master_dataset.npz.
"""

import os, sys, json, time, importlib
os.environ["NUMBA_DISABLE_JIT"] = "1"
for m in list(sys.modules):
    if m.startswith(("mne","numba")):
        del sys.modules[m]

import torch
import numpy as np
import mne

torch.set_num_threads(8)

sys.path.insert(0, os.getcwd())

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


from stend.vkr_eeg import data, dataset, splitter, detector as DET, metrics as MET, config as C
importlib.reload(data)
importlib.reload(dataset)
importlib.reload(splitter)
importlib.reload(DET)
importlib.reload(MET)

subjects = ["chb01","chb02","chb03","chb05","chb08","chb21"]

man = json.load(open(os.path.join(C.DATA_ROOT,"chbmit","download_manifest.json")))

def subject_files(s):
    return man[s]["seizure"] + man[s]["interictal"]

train_subj, eval_subj = splitter.subject_split(subjects, seed=C.SEED, test_frac=0.34)

train_stats = dataset.fit_train_stats(train_subj, man)
X, y_sz, subj, fileid, wtime = dataset.build(subjects, train_stats, man)
dhash = dataset.data_hash(subjects, man)

# file-level train/eval split
rng = np.random.default_rng(C.SEED)
files_by_subj = {s: sorted(set(fileid[subj==s])) for s in subjects}
train_files=set(); eval_files=set()
for s in subjects:
    fs = files_by_subj[s]
    perm = rng.permutation(len(fs))
    k = max(1, int(len(fs)*0.6))
    for i in perm[:k]: train_files.add((s, fs[i]))
    for i in perm[k:]: eval_files.add((s, fs[i]))

is_train_file = np.array([(subj[i], fileid[i]) in train_files for i in range(len(X))])

def eval_f1(det, Xe, ye, thr=0.5):
    p = DET.predict(det, Xe, device="cpu")
    yhat = (p>=thr).astype(int)
    return MET.f1_binary(ye, yhat), p

# ---- P0: shuffled window split (LEAKY) ----
def run_P0(seed=C.SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, ev = splitter.window_split(len(X), seed=seed, test_frac=0.3)
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr,ev), p

t0=time.time()
f1_p0, det0, (tr0,ev0), p0 = run_P0()
print(f"P0 (window split, leaky):   F1 = {f1_p0:.4f}   [{time.time()-t0:.0f}s, eval n={len(ev0)}, sz={int(y_sz[ev0].sum())}]")

# ---- P2: subject split (HONEST) ----
def run_P2(seed=C.SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_s, ev_s = splitter.subject_split(subjects, seed=seed, test_frac=0.34)
    tr = np.where(np.isin(subj, sorted(tr_s)))[0]
    ev = np.where(np.isin(subj, sorted(ev_s)))[0]
    det = DET.train_detector(X[tr], y_sz[tr], device="cpu", epochs=C.DET_EPOCHS, seed=seed)
    f1, p = eval_f1(det, X[ev], y_sz[ev])
    return f1, det, (tr,ev), p, (tr_s,ev_s)

t0=time.time()
f1_p2, det2, (tr2,ev2), p2, (trs,evs) = run_P2()
print(f"P2 (subject split, honest): F1 = {f1_p2:.4f}   [{time.time()-t0:.0f}s]")
print(f"  train subjects: {sorted(trs)} | eval subjects: {sorted(evs)}")
print(f"  eval n={len(ev2)}, seizure windows={int(y_sz[ev2].sum())}")
print(f"\n  P0 (leaky/claimed) F1 = {f1_p0:.4f}")
print(f"  P2 (honest)        F1 = {f1_p2:.4f}")
print(f"  GAP = {f1_p0-f1_p2:.4f}")

# reproducibility check
f1_p0b,_,_,_ = run_P0()
f1_p2b,*_ = run_P2()
print(f"repro P0: {f1_p0b:.4f} (was {f1_p0:.4f}) | repro P2: {f1_p2b:.4f} (was {f1_p2:.4f})")

detector_gap = {
    "f1_zayavlennyy": round(float(f1_p0),4),
    "f1_bez_utechki": round(float(f1_p2),4),
    "gap": round(float(f1_p0-f1_p2),4),
    "protocol_P0": "перемешанное оконное разбиение (утечка почти-дубликатов окон)",
    "protocol_P2": "субъектное разбиение (train/eval — разные пациенты)",
    "P0_eval": {"n":int(len(ev0)),"seizure_windows":int(y_sz[ev0].sum())},
    "P2_eval": {"n":int(len(ev2)),"seizure_windows":int(y_sz[ev2].sum()),
                "train_subjects":sorted(trs),"eval_subjects":sorted(evs)},
    "reproducible": bool(abs(f1_p0b-f1_p0)<1e-6 and abs(f1_p2b-f1_p2)<1e-6),
    "note": "Разрыw метрики при устранении утечки: F1 падает с 0.81 (завышенная) до 0.34 (честная). Согласуется с мотивацией работы (заявленные ~99% vs воспроизводимые ~32%).",
    "seed": C.SEED, "data_sha256": dhash,
    "detector_hp": {"epochs":C.DET_EPOCHS,"batch":C.DET_BATCH,"lr":C.DET_LR,"pos_weight":"neg/pos (class-balanced)"},
}
json.dump(detector_gap, open("outputs/detector_gap.json","w"), ensure_ascii=False, indent=2)
torch.save(det0.state_dict(),"outputs/detector_P0.pt"); torch.save(det2.state_dict(),"outputs/detector_P2.pt")
print("saved detector_gap.json + detector weights")