"""
Фаза 0 — сборка master-датасета (субъектное разбиение, предобработка на train, хэш)
Выполнено на стенде Главы 3, env=eeg, seed=20260706.
Код восстановлен дословно из происхождения (lineage) артефакта-результата.
Зависит от модулей stend/vkr_eeg/*.py и кэша outputs/master_dataset.npz.
"""

import os, re, json, hashlib, time, gc
import numpy as np
import mne

os.environ["NUMBA_DISABLE_JIT"] = "1"

# Config
SEED = 20260706
SFREQ = 256
WIN_SEC = 4.0
WIN_STEP_SEC = 4.0
EMBARGO_SEC = 60.0
BANDPASS = (0.5, 40.0)
NOTCH = 60.0

CANONICAL_CHANNELS = [
    "FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ",
]

DATA_ROOT = os.path.join(os.getcwd(), "eeg_data")

mne.set_log_level("ERROR")


def parse_summary(subject):
    path = os.path.join(DATA_ROOT, "chbmit", subject, f"{subject}-summary.txt")
    if not os.path.exists(path):
        path = os.path.join(DATA_ROOT, "chbmit", f"{subject}-summary.txt")
    txt = open(path).read()
    info = {}
    for b in re.split(r"File Name:\s*", txt)[1:]:
        name = b.split("\n", 1)[0].strip()
        starts = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)", b)]
        ends   = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? End Time:\s*(\d+)", b)]
        info[name] = list(zip(starts, ends))
    return info


def load_edf(subject, fname):
    path = os.path.join(DATA_ROOT, "chbmit", subject, fname)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    ren = {}
    seen = set()
    for ch in raw.ch_names:
        norm = re.sub(r"-[01]$", "", ch.upper().replace(" ", ""))
        if norm in seen:
            norm = ch.upper().replace(" ", "")
        ren[ch] = norm
        seen.add(norm)
    raw.rename_channels(ren)
    keep = [ch for ch in CANONICAL_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(keep, ordered=True)
    if len(raw.ch_names) != len(CANONICAL_CHANNELS):
        return None
    if int(round(raw.info["sfreq"])) != SFREQ:
        raw.resample(SFREQ)
    return raw


def preprocess(raw, fit_stats=None):
    raw = raw.copy()
    raw.filter(BANDPASS[0], BANDPASS[1], verbose="ERROR")
    try:
        raw.notch_filter(NOTCH, verbose="ERROR")
    except Exception:
        pass
    data = raw.get_data()
    if fit_stats is None:
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True) + 1e-7
        return {"mu": mu, "sd": sd}
    data = (data - fit_stats["mu"]) / fit_stats["sd"]
    return data


def window_file(subject, fname, seizures, fit_stats):
    raw = load_edf(subject, fname)
    if raw is None:
        return None
    data_arr = preprocess(raw, fit_stats)
    win = int(WIN_SEC * SFREQ)
    step = int(WIN_STEP_SEC * SFREQ)
    n = data_arr.shape[1]
    X, y, ts = [], [], []
    for start in range(0, n - win + 1, step):
        seg = data_arr[:, start:start + win]
        t0 = start / SFREQ
        t1 = (start + win) / SFREQ
        lab = 0
        for (ss, se) in seizures:
            if t0 < se and t1 > ss:
                lab = 1; break
        X.append(seg.astype(np.float32)); y.append(lab); ts.append(t0)
    if not X:
        return None
    return np.stack(X), np.array(y, dtype=np.int64), np.array(ts, dtype=np.float32)


def sha256_of_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def subject_split(subjects_unique, seed=SEED, test_frac=0.3):
    rng = np.random.default_rng(seed)
    su = np.array(sorted(subjects_unique))
    perm = rng.permutation(len(su))
    cut = max(1, int(len(su) * (1 - test_frac)))
    return set(su[perm[:cut]].tolist()), set(su[perm[cut:]].tolist())


def fit_train_stats(train_subjects, manifest):
    sums = sqs = None; cnt = 0
    for s in sorted(train_subjects):
        for fn in manifest[s]["seizure"] + manifest[s]["interictal"]:
            raw = load_edf(s, fn)
            if raw is None: continue
            raw.filter(BANDPASS[0], BANDPASS[1], verbose="ERROR")
            try: raw.notch_filter(NOTCH, verbose="ERROR")
            except Exception: pass
            d = raw.get_data()
            sums = d.sum(1) if sums is None else sums + d.sum(1)
            sqs  = (d**2).sum(1) if sqs is None else sqs + (d**2).sum(1)
            cnt += d.shape[1]
            del raw, d
    mu = (sums / cnt).reshape(-1, 1)
    sd = (np.sqrt(sqs / cnt - (sums / cnt)**2)).reshape(-1, 1) + 1e-7
    return {"mu": mu, "sd": sd, "n_samples": int(cnt)}


def build(subjects, train_stats, manifest):
    Xall = []; ysz = []; subj = []; fid = []; wt = []
    for s in subjects:
        info = parse_summary(s)
        for fn in manifest[s]["seizure"] + manifest[s]["interictal"]:
            out = window_file(s, fn, info.get(fn, []), train_stats)
            if out is None: continue
            Xf, yf, tf = out
            Xall.append(Xf); ysz.append(yf)
            subj += [s] * len(Xf); fid += [fn] * len(Xf); wt.append(tf)
    return (np.concatenate(Xall), np.concatenate(ysz),
            np.array(subj), np.array(fid), np.concatenate(wt))


def data_hash(subjects, manifest):
    paths = []
    for s in subjects:
        for fn in manifest[s]["seizure"] + manifest[s]["interictal"]:
            paths.append(os.path.join(DATA_ROOT, "chbmit", s, fn))
    return sha256_of_files([p for p in paths if os.path.exists(p)])


subjects = ["chb01", "chb02", "chb03", "chb05", "chb08", "chb21"]

train_subj, eval_subj = subject_split(subjects, seed=SEED, test_frac=0.34)

man = json.load(open(os.path.join(DATA_ROOT, "chbmit", "download_manifest.json")))

t0 = time.time()
train_stats = fit_train_stats(train_subj, man)
X, y_sz, subj, fileid, wtime = build(subjects, train_stats, man)
dhash = data_hash(subjects, man)
print(f"MASTER: X={X.shape} ({X.nbytes/1e9:.2f}GB) seizure_win={int(y_sz.sum())} total={len(X)} | {time.time()-t0:.0f}s")
print("data SHA-256:", dhash[:16], "...")

os.makedirs("outputs", exist_ok=True)
np.savez_compressed("master_dataset.npz",
    X=X, y_sz=y_sz, subj=subj, fileid=fileid, wtime=wtime,
    mu=train_stats["mu"], sd=train_stats["sd"])
print("saved master_dataset.npz")