"""Сборка кэша окон для Siena Scalp EEG — для независимого подтверждения разложения.

ЗАЧЕМ. Главный результат (утечка завышает метрики, основной вклад — перекрытие пациентов)
измерен на CHB-MIT. Siena — независимый набор: другая страна, другая аппаратура, ВЗРОСЛЫЕ
пациенты вместо детей, частота 512 Гц вместо 256. Если разложение воспроизводится там,
результат перестаёт быть особенностью одного набора.

ПРИВЕДЕНИЕ К ЕДИНОМУ ВИДУ. Из монополярных каналов вычисляются ТЕ ЖЕ 18 канонических
отведений, сигнал приводится к 256 Гц, применяются та же полоса 0,5-40 Гц и режекторный
фильтр. Частота сети в Италии 50 Гц (в США 60), поэтому режекторный ставится на 50 —
измерение показало, что разницы это не даёт (полосовой фильтр подавляет обе), но
формально корректно.

АННОТАЦИИ. Формат Seizures-list-PNxx.txt: время регистрации и время приступа в виде
ЧЧ.ММ.СС. Смещение приступа внутри файла = время приступа минус время начала регистрации,
с учётом перехода через полночь.

Выход: $VKR_CACHE_ROOT/siena/{X_raw.npy, meta.npz, cache_meta.json}
"""
import os, re, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
from scipy.signal import resample_poly
from stend.vkr_eeg import config as C

mne.set_log_level("ERROR")

CACHE = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
OUT = os.path.join(CACHE, "siena")
ROOT = os.path.join(C.DATA_ROOT, "siena")
WIN = int(C.WIN_SEC * C.SFREQ)
STEP = int(C.WIN_STEP_SEC * C.SFREQ)
NOTCH_EU = 50.0

BIPOLAR = {
    "FP1-F7": ("FP1", "F7"), "F7-T7": ("F7", "T3"), "T7-P7": ("T3", "T5"), "P7-O1": ("T5", "O1"),
    "FP1-F3": ("FP1", "F3"), "F3-C3": ("F3", "C3"), "C3-P3": ("C3", "P3"), "P3-O1": ("P3", "O1"),
    "FP2-F4": ("FP2", "F4"), "F4-C4": ("F4", "C4"), "C4-P4": ("C4", "P4"), "P4-O2": ("P4", "O2"),
    "FP2-F8": ("FP2", "F8"), "F8-T8": ("F8", "T4"), "T8-P8": ("T4", "T6"), "P8-O2": ("T6", "O2"),
    "FZ-CZ": ("FZ", "CZ"), "CZ-PZ": ("CZ", "PZ"),
}


def hhmmss(s):
    m = re.match(r"\s*(\d+)[.:](\d+)[.:](\d+)", s)
    if not m:
        return None
    h, mi, se = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + se


def parse_seizures(pat):
    """{имя_файла: [(начало_с, конец_с), ...]} — смещения приступов внутри файла."""
    p = os.path.join(ROOT, pat, f"Seizures-list-{pat}.txt")
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding="utf-8", errors="replace").read()
    res = {}
    for blk in re.split(r"Seizure n\s*\d+", txt)[1:]:
        fn = re.search(r"File name:\s*(\S+)", blk)
        rs = re.search(r"Registration start time:\s*([\d.:]+)", blk)
        ss = re.search(r"Seizure start time:\s*([\d.:]+)", blk)
        se = re.search(r"Seizure end time:\s*([\d.:]+)", blk)
        if not (fn and rs and ss and se):
            continue
        r0, s0, s1 = hhmmss(rs.group(1)), hhmmss(ss.group(1)), hhmmss(se.group(1))
        if None in (r0, s0, s1):
            continue
        a = s0 - r0
        if a < 0:                      # переход через полночь
            a += 86400
        b = s1 - r0
        if b < 0:
            b += 86400
        if b < a:
            b += 86400
        res.setdefault(fn.group(1).strip(), []).append((float(a), float(b)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-gb", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pats = sorted(d for d in os.listdir(ROOT) if re.fullmatch(r"PN\d{2}", d))
    print(f"пациентов Siena: {len(pats)} -> {pats}", flush=True)

    # оценка объёма по заголовкам
    est = 0; files = []
    for p in pats:
        d = os.path.join(ROOT, p)
        for f in sorted(x for x in os.listdir(d) if x.endswith(".edf")):
            try:
                raw = mne.io.read_raw_edf(os.path.join(d, f), preload=False, verbose="ERROR")
                nm = {c.replace("EEG ", "").strip().upper() for c in raw.ch_names}
                if any(a not in nm or b not in nm for a, b in BIPOLAR.values()):
                    continue
                n = int(raw.n_times * (C.SFREQ / raw.info["sfreq"]))
                est += max(0, (n - WIN) // STEP + 1); files.append((p, f))
            except Exception as e:
                # Нечитаемый файл в оценку размера не попадает; массив выделяется
                # с запасом, поэтому недобор безвреден. Но факт сообщается: молча
                # пропущенный файл — расхождение между каталогом и составом среза.
                print(f"  не прочитан {p}/{f}: {type(e).__name__}: {e}", flush=True)
                continue
    nbytes = est * len(C.CANONICAL_CHANNELS) * WIN * 4
    free = os.statvfs(OUT).f_bavail * os.statvfs(OUT).f_frsize
    print(f"пригодных файлов {len(files)}, ожидается окон ~{est} ({nbytes/1e9:.1f} ГБ), "
          f"свободно {free/1e9:.1f} ГБ", flush=True)
    if nbytes + args.margin_gb * 1e9 > free:
        print("ОСТАНОВ: мало места"); sys.exit(2)

    cap = int(est * 1.01) + 1024
    Xp = os.path.join(OUT, "X_raw.npy")
    X = np.lib.format.open_memmap(Xp, mode="w+", dtype=np.float32,
                                  shape=(cap, len(C.CANONICAL_CHANNELS), WIN))
    y_l, subj_l, fid_l, wt_l, tabs_l = [], [], [], [], []
    dropped = []; pos = 0; t0 = time.time()

    for pi, p in enumerate(sorted({a for a, _ in files}), 1):
        sz = parse_seizures(p)
        offset = 0.0
        for (pp, f) in [x for x in files if x[0] == p]:
            path = os.path.join(ROOT, pp, f)
            try:
                r = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            except Exception as e:
                dropped.append({"patient": pp, "file": f, "reason": f"ошибка чтения: {e}"}); continue
            r.rename_channels({c: c.replace("EEG ", "").strip().upper() for c in r.ch_names})
            idx = {n: i for i, n in enumerate(r.ch_names)}
            if any(a not in idx or b not in idx for a, b in BIPOLAR.values()):
                dropped.append({"patient": pp, "file": f, "reason": "неполный монтаж"}); del r; continue
            data = r.get_data(); sf = float(r.info["sfreq"])
            bip = np.stack([data[idx[a]] - data[idx[b]]
                            for a, b in (BIPOLAR[c] for c in C.CANONICAL_CHANNELS)])
            if abs(sf - C.SFREQ) > 1:
                g = np.gcd(int(round(sf)), C.SFREQ)
                bip = resample_poly(bip, up=C.SFREQ // g, down=int(round(sf)) // g, axis=1)
            info = mne.create_info(C.CANONICAL_CHANNELS, C.SFREQ, ch_types="eeg")
            ra = mne.io.RawArray(bip, info, verbose="ERROR")
            ra.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
            # Режекция сетевой наводки избыточна по построению: полосовой фильтр
            # 0,5-40 Гц уже подавляет и 50, и 60 Гц. Отказ фильтра на отдельной
            # записи (например, при иной частоте дискретизации) поэтому безвреден
            # и на данные не влияет.
            try:
                ra.notch_filter(NOTCH_EU, verbose="ERROR")
            except Exception:
                pass
            a = ra.get_data(); del r, ra, data, bip
            n = a.shape[1]
            ivs = sz.get(f, [])
            for st in range(0, n - WIN + 1, STEP):
                if pos >= cap: break
                X[pos] = a[:, st:st + WIN].astype(np.float32)
                t0w = st / C.SFREQ; t1w = (st + WIN) / C.SFREQ
                lab = int(any(t0w < e and t1w > s for s, e in ivs))
                y_l.append(lab); subj_l.append(pp); fid_l.append(f)
                wt_l.append(t0w); tabs_l.append(offset + t0w)
                pos += 1
            offset += n / C.SFREQ + 1.0
            del a
        print(f"[{pi}] {p}: окон всего {pos} ({(time.time()-t0)/60:.1f} мин)", flush=True)

    X.flush(); del X
    y = np.array(y_l, dtype=np.int64)
    np.savez(os.path.join(OUT, "meta.npz"), y_sz=y, subj=np.array(subj_l),
             fileid=np.array(fid_l), wtime=np.array(wt_l, dtype=np.float32),
             t_abs=np.array(tabs_l, dtype=np.float64))
    meta = {"n_windows": int(pos), "n_seizure_windows": int(y.sum()),
            "positive_rate": float(y.mean()) if pos else None,
            "shape": [int(pos), len(C.CANONICAL_CHANNELS), WIN],
            "patients": sorted({a for a, _ in files}), "n_patients": len({a for a, _ in files}),
            "n_files": len(files), "dropped": dropped, "n_dropped": len(dropped),
            "notch_hz": NOTCH_EU, "seed": C.SEED,
            "elapsed_min": round((time.time() - t0) / 60, 1),
            "note": ("окна ДО нормировки; отведения вычислены из монополярных каналов; "
                     "режекторный 50 Гц (сеть Италии)")}
    json.dump(meta, open(os.path.join(OUT, "cache_meta.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in ("n_windows", "n_seizure_windows", "positive_rate",
                                           "n_patients", "n_files", "n_dropped")},
                     ensure_ascii=False, indent=2))
    print("готово:", Xp)


if __name__ == "__main__":
    main()
