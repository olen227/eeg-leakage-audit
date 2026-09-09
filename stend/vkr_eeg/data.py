"""Загрузка EDF, разбор аннотаций приступов, нарезка окон.
Предобработка (фильтры) настраивается ТОЛЬКО на train (изоляция предобработки)."""
import os, re, hashlib
import numpy as np
import mne
from . import config as C

mne.set_log_level("ERROR")

def parse_summary(subject):
    path = os.path.join(C.DATA_ROOT, "chbmit", subject, f"{subject}-summary.txt")
    if not os.path.exists(path):
        path = os.path.join(C.DATA_ROOT, "chbmit", f"{subject}-summary.txt")
    txt = open(path).read()
    info = {}
    for b in re.split(r"File Name:\s*", txt)[1:]:
        name = b.split("\n", 1)[0].strip()
        starts = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)", b)]
        ends   = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? End Time:\s*(\d+)", b)]
        info[name] = list(zip(starts, ends))
    return info

def load_edf(subject, fname):
    path = os.path.join(C.DATA_ROOT, "chbmit", subject, fname)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    # унификация имён каналов -> верхний регистр, без пробелов;
    # снятие суффикса дубликата монтажа CHB-MIT (T8-P8-0 / T8-P8-1 -> T8-P8)
    ren = {}
    seen = set()
    for ch in raw.ch_names:
        norm = re.sub(r"-[01]$", "", ch.upper().replace(" ", ""))
        if norm in seen:
            norm = ch.upper().replace(" ", "")  # оставить исходное, чтобы не столкнуть
        ren[ch] = norm
        seen.add(norm)
    raw.rename_channels(ren)
    keep = [ch for ch in C.CANONICAL_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(keep, ordered=True)
    if len(raw.ch_names) != len(C.CANONICAL_CHANNELS):
        return None  # неполный монтаж — пропускаем
    if int(round(raw.info["sfreq"])) != C.SFREQ:
        raw.resample(C.SFREQ)
    return raw

def preprocess(raw, fit_stats=None):
    """Фильтрация. Нормировка по статистикам train (fit_stats).
    Если fit_stats=None — возвращает статистики (режим fit на train)."""
    raw = raw.copy()
    raw.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
    # Режекция сетевой наводки избыточна: полосовой 0,5-40 Гц уже подавляет и 50,
    # и 60 Гц, поэтому отказ режекторного фильтра на содержимое окон не влияет.
    try:
        raw.notch_filter(C.NOTCH, verbose="ERROR")
    except Exception:
        pass
    data = raw.get_data()  # (n_ch, n_samples)
    if fit_stats is None:
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True) + 1e-7
        return {"mu": mu, "sd": sd}
    data = (data - fit_stats["mu"]) / fit_stats["sd"]
    return data

def window_file(subject, fname, seizures, fit_stats):
    """Нарезка одного файла на окна. Возвращает (X, y_seizure, t_start).
    y_seizure=1, если окно пересекается с интервалом приступа."""
    raw = load_edf(subject, fname)
    if raw is None:
        return None
    data = preprocess(raw, fit_stats)          # (n_ch, n_samples), нормировано
    win = int(C.WIN_SEC * C.SFREQ)
    step = int(C.WIN_STEP_SEC * C.SFREQ)
    n = data.shape[1]
    X, y, ts = [], [], []
    for start in range(0, n - win + 1, step):
        seg = data[:, start:start + win]
        t0 = start / C.SFREQ
        t1 = (start + win) / C.SFREQ
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
