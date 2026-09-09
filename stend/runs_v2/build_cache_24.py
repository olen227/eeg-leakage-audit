"""Сборка кэша окон для полного набора CHB-MIT (24 идентификатора) — без переполнения памяти.

ЗАЧЕМ ОТДЕЛЬНЫЙ СБОРЩИК. Июльский phase0_dataset.py накапливает все окна в списке
и делает np.concatenate — для 6 субъектов это 13,6 ГБ в оперативной памяти, для 24
получится около 59 ГБ, что не помещается в 32 ГБ машины. Здесь окна пишутся
непосредственно в memmap на диск, в оперативной памяти держится один файл за раз.

ЧТО ПИШЕТСЯ. Окна ДО нормировки (как и в кэше на 6 субъектов): нормировка в конвейере v2
применяется пофолдово по обучающей части, поэтому глобальная нормировка кэшу не нужна.
Дополнительно восстанавливается абсолютное время окна по 'File Start Time'.

Предобработка идентична модулю stend/vkr_eeg/data.py: те же 18 канонических отведений,
полоса 0,5-40 Гц, режекторный 60 Гц, окна 4 с при 256 Гц, шаг 4 с.

Выход: $VKR_CACHE_ROOT/full24/{X_raw.npy, meta.npz, cache_meta.json}
"""
import os, re, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
from stend.vkr_eeg import data as D, config as C

mne.set_log_level("ERROR")

CACHE = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
OUT = os.path.join(CACHE, "full24")
WIN = int(C.WIN_SEC * C.SFREQ)
STEP = int(C.WIN_STEP_SEC * C.SFREQ)


def parse_start_times(subject):
    path = os.path.join(C.DATA_ROOT, "chbmit", subject, f"{subject}-summary.txt")
    txt = open(path).read()
    starts, prev, day = {}, None, 0
    for b in re.split(r"File Name:\s*", txt)[1:]:
        name = b.split("\n", 1)[0].strip()
        m = re.search(r"File Start Time:\s*(\d+):(\d+):(\d+)", b)
        if not m:
            starts[name] = None; continue
        h, mi, s = (int(x) for x in m.groups())
        sec = h * 3600 + mi * 60 + s
        if prev is not None and sec < prev:
            day += 1
        prev = sec
        starts[name] = sec + day * 86400
    vals = [v for v in starts.values() if v is not None]
    base = min(vals) if vals else 0
    return {k: (None if v is None else v - base) for k, v in starts.items()}


def estimate_windows(subjects, man):
    """Число окон по заголовкам EDF, без чтения сигнала.

    Файлы с неполным каноническим монтажом отбрасываются здесь так же, как их
    отбросит load_edf, — иначе оценка завышается и место резервируется впустую.
    Имена каналов доступны из заголовка при preload=False.
    """
    import re as _re
    need = set(C.CANONICAL_CHANNELS)
    total = 0
    for s in subjects:
        for fn in man[s]["seizure"] + man[s]["interictal"]:
            p = os.path.join(C.DATA_ROOT, "chbmit", s, fn)
            if not os.path.exists(p):
                continue
            try:
                raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
                seen, norm = set(), []
                for ch in raw.ch_names:
                    v = _re.sub(r"-[01]$", "", ch.upper().replace(" ", ""))
                    if v in seen:
                        v = ch.upper().replace(" ", "")
                    seen.add(v); norm.append(v)
                if not need.issubset(set(norm)):
                    continue                      # будет отброшен load_edf
                n = int(raw.n_times * (C.SFREQ / raw.info["sfreq"]))
                total += max(0, (n - WIN) // STEP + 1)
            except Exception:
                continue
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="")
    ap.add_argument("--margin-gb", type=float, default=10.0, help="запас места на диске, ГБ")
    args = ap.parse_args()

    man_path = os.path.join(C.DATA_ROOT, "chbmit", "download_manifest.json")
    man = json.load(open(man_path))
    subjects = (args.subjects.split(",") if args.subjects
                else sorted(k for k in man if re.fullmatch(r"chb\d{2}", k)))
    subjects = [s for s in subjects if os.path.isdir(os.path.join(C.DATA_ROOT, "chbmit", s))]
    print(f"субъектов: {len(subjects)} -> {subjects}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    print("оцениваю объём по заголовкам EDF ...", flush=True)
    t0 = time.time()
    est = estimate_windows(subjects, man)
    nbytes = est * len(C.CANONICAL_CHANNELS) * WIN * 4
    free = os.statvfs(OUT).f_bavail * os.statvfs(OUT).f_frsize
    print(f"  ожидается окон ~{est}, объём ~{nbytes/1e9:.1f} ГБ "
          f"(оценка за {time.time()-t0:.0f} с)", flush=True)
    print(f"  свободно на диске {free/1e9:.1f} ГБ", flush=True)
    if nbytes + args.margin_gb * 1e9 > free:
        print("ОСТАНОВ: недостаточно места на диске. Освободите место или сократите набор.",
              flush=True)
        sys.exit(2)

    cap = int(est * 1.005) + 1024          # запас на округление при передискретизации
    Xp = os.path.join(OUT, "X_raw.npy")
    X = np.lib.format.open_memmap(Xp, mode="w+", dtype=np.float32,
                                  shape=(cap, len(C.CANONICAL_CHANNELS), WIN))
    y_l, subj_l, fid_l, wt_l, tabs_l = [], [], [], [], []
    dropped = []
    pos = 0
    t0 = time.time()

    for si, s in enumerate(subjects, 1):
        starts = parse_start_times(s)
        info = D.parse_summary(s)
        files = man[s]["seizure"] + man[s]["interictal"]
        for fn in files:
            p = os.path.join(C.DATA_ROOT, "chbmit", s, fn)
            if not os.path.exists(p):
                dropped.append({"subject": s, "file": fn, "reason": "файла нет на диске"})
                continue
            raw = D.load_edf(s, fn)
            if raw is None:
                dropped.append({"subject": s, "file": fn,
                                "reason": "неполный канонический монтаж (18 отведений)"})
                continue
            r = raw.copy()
            r.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
            try:
                r.notch_filter(C.NOTCH, verbose="ERROR")
            except Exception:
                pass
            d = r.get_data()
            del raw, r
            n = d.shape[1]
            sz = info.get(fn, [])
            off = starts.get(fn)
            k = 0
            for st in range(0, n - WIN + 1, STEP):
                if pos >= cap:
                    break
                X[pos] = d[:, st:st + WIN].astype(np.float32)
                t_start = st / C.SFREQ
                t_end = (st + WIN) / C.SFREQ
                lab = 0
                for (ss_, se_) in sz:
                    if t_start < se_ and t_end > ss_:
                        lab = 1; break
                y_l.append(lab); subj_l.append(s); fid_l.append(fn)
                wt_l.append(t_start)
                tabs_l.append(np.nan if off is None else off + t_start)
                pos += 1; k += 1
            del d
        print(f"[{si}/{len(subjects)}] {s}: окон всего {pos} "
              f"({(time.time()-t0)/60:.1f} мин)", flush=True)

    X.flush()
    del X
    # Усечения через копию НЕ делаем: оно требует временно удвоенного места на диске
    # (именно на этом шаге сборка падала с ENOSPC). Фактическое число окон пишется
    # в метаданные; хвост массива за пределами n_windows просто никогда не индексируется,
    # поскольку все индексы конвейера строятся из meta.npz.
    y = np.array(y_l, dtype=np.int64)
    if pos < cap:
        print(f"выделено {cap} строк, заполнено {pos}; "
              f"хвост {(cap-pos)*len(C.CANONICAL_CHANNELS)*WIN*4/1e9:.1f} ГБ не используется "
              f"(усечение копией не выполняется намеренно)", flush=True)
    np.savez(os.path.join(OUT, "meta.npz"),
             y_sz=y, subj=np.array(subj_l), fileid=np.array(fid_l),
             wtime=np.array(wt_l, dtype=np.float32),
             t_abs=np.array(tabs_l, dtype=np.float64))
    meta = {
        "n_windows": int(pos), "n_seizure_windows": int(y.sum()),
        "positive_rate": float(y.mean()) if pos else None,
        "shape": [int(pos), len(C.CANONICAL_CHANNELS), WIN],
        "subjects": subjects, "n_subjects": len(subjects),
        "dropped_files": dropped, "n_dropped": len(dropped),
        "seed": C.SEED, "manifest": man_path,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "note": "окна ДО нормировки; нормировка применяется пофолдово по обучающей части",
    }
    json.dump(meta, open(os.path.join(OUT, "cache_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    # требование П4 исходного задания: учёт отброшенных файлов
    json.dump({"n_dropped": len(dropped), "dropped": dropped,
               "subjects_scanned": subjects},
              open(os.path.join("outputs", "dropped_files.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in
                      ("n_windows", "n_seizure_windows", "positive_rate",
                       "n_subjects", "n_dropped", "elapsed_min")},
                     ensure_ascii=False, indent=2))
    print("готово:", Xp)


if __name__ == "__main__":
    main()
