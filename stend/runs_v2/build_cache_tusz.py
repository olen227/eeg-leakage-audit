"""Сборка кэша окон для TUH EEG Seizure Corpus (TUSZ) — взрослые пациенты.

ЗАЧЕМ. Проверка на взрослых (Siena, 14 пациентов, 47 приступов) мощности не набрала:
интервал завышения [−0,0156; 0,1414] включает ноль. TUSZ v2.0.6 — 675 пациентов и
2 664 размеченных приступа, на порядок больше всех трёх прежних наборов вместе. Если
эффект воспроизводится здесь, вывод о взрослых перестаёт быть неопределённым.

УСТРОЙСТВО НАБОРА. edf/{train,dev,eval}/<пациент>/<сеанс>/<монтаж>/<запись>.edf; к
каждой записи файл .csv_bi — разметка одной дорожкой на всю запись, строки
`TERM,начало,конец,seiz|bckg,уверенность`. Авторское разбиение по пациентам не
пересекается; оно записывается в meta.npz (массив split), но фолды стенда строятся
поверх всего корпуса своим разбиением по идентичностям.

ПРИВЕДЕНИЕ К ЕДИНОМУ ВИДУ. В EDF лежат референтные каналы (`EEG FP1-REF` или `-LE`);
18 канонических биполярных отведений вычисляются вычитанием, опорный электрод при
этом сокращается, поэтому монтажи AR и LE обрабатываются одинаково. Электроды T3/T4/
T5/T6 набора — те же, что T7/T8/P7/P8 в CHB-MIT (старая и новая номенклатура 10-20).
У 625 записей из 8 140 нет электродов FZ и PZ; такие записи отбрасываются и
перечисляются в паспорте среза, контракт «18 каналов» сохраняется. Частота приводится
к 256 Гц (в наборе 250/256/400/512/1000). Полоса 0,5-40 Гц, режекторный 60 Гц (США).

ХРАНЕНИЕ В FLOAT16. Кэш в float32 занял бы около 90 ГБ и на диск не помещается; по
контракту данных допускается float16 (относительная точность ~0,05 %). Чтобы значения
не попадали в субнормальный диапазон float16, окна хранятся В МИКРОВОЛЬТАХ (MNE
отдаёт вольты). Масштаб на измерение не влияет: окна нормируются пофолдово.

ЕДИНАЯ ШКАЛА ВРЕМЕНИ. Абсолютных меток времени в TUSZ нет (даты обезличены до года,
время суток отсутствует). Для эмбарго строится ВИРТУАЛЬНАЯ шкала пациента: записи
одного сеанса (t000, t001, …) — это последовательные отрезки одной регистрации и
кладутся встык; между сеансами вставляется зазор в сутки. Эмбарго 60 с при этом
работает точно там, где соседство реально, — на стыках отрезков одного сеанса.

Выход: $VKR_CACHE_ROOT/tusz/{X_raw.npy, meta.npz, cache_meta.json}

ЗАПУСК:
    $PY stend/runs_v2/build_cache_tusz.py                 # весь корпус
    $PY stend/runs_v2/build_cache_tusz.py --limit 20 --out /tmp/tusz_smoke   # проверка
"""
import os, re, sys, glob, json, time, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
from scipy.signal import resample_poly
from stend.vkr_eeg import config as C

mne.set_log_level("ERROR")

CACHE = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
ROOT = os.path.join(C.DATA_ROOT, "tusz", "v2.0.6")
WIN = int(C.WIN_SEC * C.SFREQ)
STEP = int(C.WIN_STEP_SEC * C.SFREQ)
NOTCH_US = 60.0
SESSION_GAP_SEC = 86400.0        # зазор виртуальной шкалы между сеансами
TO_MICROVOLT = 1e6

BIPOLAR = {
    "FP1-F7": ("FP1", "F7"), "F7-T7": ("F7", "T3"), "T7-P7": ("T3", "T5"), "P7-O1": ("T5", "O1"),
    "FP1-F3": ("FP1", "F3"), "F3-C3": ("F3", "C3"), "C3-P3": ("C3", "P3"), "P3-O1": ("P3", "O1"),
    "FP2-F4": ("FP2", "F4"), "F4-C4": ("F4", "C4"), "C4-P4": ("C4", "P4"), "P4-O2": ("P4", "O2"),
    "FP2-F8": ("FP2", "F8"), "F8-T8": ("F8", "T4"), "T8-P8": ("T4", "T6"), "P8-O2": ("T6", "O2"),
    "FZ-CZ": ("FZ", "CZ"), "CZ-PZ": ("CZ", "PZ"),
}
NEEDED = sorted({e for pair in BIPOLAR.values() for e in pair})


def norm_ch(name):
    """'EEG FP1-REF' / 'EEG Fp1-LE' -> 'FP1'."""
    s = name.upper().replace("EEG ", "").strip()
    return re.sub(r"-\s*(REF|LE)$", "", s).strip()


def edf_header(path):
    """Имена каналов, частота и длительность из заголовка EDF без чтения сигнала."""
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256]); nrec = int(head[236:244]); rec_dur = float(head[244:252])
        rest = f.read(ns * 256)
    labels = [rest[i * 16:(i + 1) * 16].decode("latin1").strip() for i in range(ns)]
    nsamp = int(rest[ns * 216:ns * 216 + 8])
    return labels, nsamp / rec_dur, nrec * rec_dur


def parse_csv_bi(path):
    """[(начало_с, конец_с), ...] приступов и длительность записи."""
    seiz, dur = [], None
    for ln in open(path):
        if ln.startswith("# duration"):
            dur = float(ln.split("=")[1].split()[0])
        elif ln.startswith("TERM"):
            _, a, b, lab, _ = ln.strip().split(",")
            if lab == "seiz":
                seiz.append((float(a), float(b)))
    return seiz, dur


def list_records():
    """Все записи корпуса в порядке пациент → сеанс → отрезок."""
    recs = []
    for edf in sorted(glob.glob(os.path.join(ROOT, "edf", "*", "*", "*", "*", "*.edf"))):
        rel = os.path.relpath(edf, os.path.join(ROOT, "edf")).split(os.sep)
        split, subj, session, montage, fname = rel
        recs.append({"split": split, "subj": subj, "session": session, "montage": montage,
                     "file": fname, "edf": edf, "csv_bi": edf[:-4] + ".csv_bi"})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CACHE, "tusz"))
    ap.add_argument("--limit", type=int, default=0, help="взять первые N записей (проверка)")
    ap.add_argument("--margin-gb", type=float, default=5.0)
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    recs = list_records()
    n_source = len(recs)
    if args.limit:
        recs = recs[:args.limit]
    print(f"записей в корпусе: {n_source}, в сборке: {len(recs)}", flush=True)

    # --- предварительный проход по заголовкам: пригодность и объём ---
    usable, skipped, est = [], [], 0
    for r in recs:
        try:
            labels, sf, dur = edf_header(r["edf"])
        except Exception as e:
            skipped.append({**{k: r[k] for k in ("split", "subj", "file")},
                            "reason": f"заголовок не прочитан ({type(e).__name__}: {e})"})
            continue
        have = {norm_ch(l) for l in labels}
        miss = [e for e in NEEDED if e not in have]
        if miss:
            skipped.append({**{k: r[k] for k in ("split", "subj", "file")},
                            "reason": f"нет электродов {','.join(miss)}"})
            continue
        if not os.path.exists(r["csv_bi"]):
            skipped.append({**{k: r[k] for k in ("split", "subj", "file")},
                            "reason": "нет файла разметки .csv_bi"})
            continue
        r["sfreq"], r["dur"] = sf, dur
        r["exclude"] = [l for l in labels if norm_ch(l) not in NEEDED]
        est += max(0, (int(dur * C.SFREQ) - WIN) // STEP + 1)
        usable.append(r)
    nbytes = est * len(C.CANONICAL_CHANNELS) * WIN * 2          # float16
    free = os.statvfs(OUT).f_bavail * os.statvfs(OUT).f_frsize
    print(f"пригодных: {len(usable)}, отброшено на предпросмотре: {len(skipped)}, "
          f"ожидается окон ~{est} ({nbytes/1e9:.1f} ГБ float16), свободно {free/1e9:.1f} ГБ",
          flush=True)
    if nbytes + args.margin_gb * 1e9 > free:
        print("ОСТАНОВ: мало места"); sys.exit(2)

    cap = int(est * 1.01) + 1024
    Xp = os.path.join(OUT, "X_raw.npy")
    X = np.lib.format.open_memmap(Xp, mode="w+", dtype=np.float16,
                                  shape=(cap, len(C.CANONICAL_CHANNELS), WIN))
    y_l, subj_l, fid_l, wt_l, tabs_l, split_l = [], [], [], [], [], []
    dropped, pos, t0 = [], 0, time.time()
    n_seiz_events = 0
    # виртуальная шкала: смещение текущей позиции по каждому пациенту и последний сеанс
    offset, last_session = {}, {}
    sf_hist = {}

    for k, r in enumerate(usable, 1):
        sid = r["subj"]
        if sid not in offset:
            offset[sid], last_session[sid] = 0.0, r["session"]
        elif r["session"] != last_session[sid]:
            offset[sid] += SESSION_GAP_SEC
            last_session[sid] = r["session"]
        try:
            raw = mne.io.read_raw_edf(r["edf"], preload=True, exclude=r["exclude"],
                                      verbose="ERROR")
        except Exception as e:
            dropped.append({"subj": sid, "file": r["file"], "reason": f"{type(e).__name__}: {e}"})
            continue
        raw.rename_channels({c: norm_ch(c) for c in raw.ch_names})
        idx = {n_: i for i, n_ in enumerate(raw.ch_names)}
        data = raw.get_data(); sf = float(raw.info["sfreq"])
        sf_hist[str(int(round(sf)))] = sf_hist.get(str(int(round(sf))), 0) + 1
        bip = np.stack([data[idx[a]] - data[idx[b]]
                        for a, b in (BIPOLAR[c] for c in C.CANONICAL_CHANNELS)])
        if abs(sf - C.SFREQ) > 1:
            g = np.gcd(int(round(sf)), C.SFREQ)
            bip = resample_poly(bip, up=C.SFREQ // g, down=int(round(sf)) // g, axis=1)
        info = mne.create_info(C.CANONICAL_CHANNELS, C.SFREQ, ch_types="eeg")
        ra = mne.io.RawArray(bip, info, verbose="ERROR")
        ra.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
        # Полоса 0,5-40 Гц уже подавляет сетевые 60 Гц; режекторный фильтр — формальное
        # соответствие остальным наборам, и его отказ на отдельной записи безвреден.
        try:
            ra.notch_filter(NOTCH_US, verbose="ERROR")
        except Exception:
            pass
        a = ra.get_data() * TO_MICROVOLT
        del raw, ra, data, bip
        n = a.shape[1]
        seiz, dur_ann = parse_csv_bi(r["csv_bi"])
        n_seiz_events += len(seiz)

        for st in range(0, n - WIN + 1, STEP):
            if pos >= cap:
                break
            X[pos] = a[:, st:st + WIN].astype(np.float16)
            t0w = st / C.SFREQ; t1w = (st + WIN) / C.SFREQ
            y_l.append(1 if any(s0 < t1w and s1 > t0w for s0, s1 in seiz) else 0)
            subj_l.append(sid); fid_l.append(r["file"]); split_l.append(r["split"])
            wt_l.append(t0w); tabs_l.append(offset[sid] + t0w)
            pos += 1
        offset[sid] += n / C.SFREQ
        del a
        if k % 50 == 0 or k == len(usable):
            print(f"[{k}/{len(usable)}] окон {pos} ({(time.time()-t0)/60:.1f} мин)", flush=True)

    X.flush(); del X
    y = np.array(y_l, dtype=np.int64)
    np.savez(os.path.join(OUT, "meta.npz"), y_sz=y, subj=np.array(subj_l),
             fileid=np.array(fid_l), wtime=np.array(wt_l, dtype=np.float32),
             t_abs=np.array(tabs_l, dtype=np.float64), split=np.array(split_l))
    subj_arr = np.array(subj_l)
    with_sz = sorted(set(subj_arr[y == 1].tolist()))
    by_split = {}
    for s in ("train", "dev", "eval"):
        m = np.array(split_l) == s
        by_split[s] = {"n_windows": int(m.sum()), "n_seizure_windows": int(y[m].sum()),
                       "n_subjects": len(set(subj_arr[m].tolist()))}
    meta = {"n_windows": int(pos), "n_seizure_windows": int(y.sum()),
            "positive_rate": float(y.mean()) if pos else None,
            "shape": [int(pos), len(C.CANONICAL_CHANNELS), WIN],
            "dtype": "float16", "units": "мкВ",
            "n_subjects": len(set(subj_l)), "n_subjects_with_seizures": len(with_sz),
            "n_seizure_events_in_slice": n_seiz_events,
            "n_files_in_source": n_source, "n_files_in_build": len(recs),
            "n_files_used": len(usable) - len(dropped),
            "skipped_at_prescan": skipped, "n_skipped_at_prescan": len(skipped),
            "dropped": dropped, "n_dropped": len(dropped),
            "sfreq_histogram_used_files": sf_hist,
            "by_official_split": by_split,
            "notch_hz": NOTCH_US, "session_gap_sec": SESSION_GAP_SEC,
            "seed": C.SEED, "elapsed_min": round((time.time() - t0) / 60, 1),
            "source_version": "TUSZ v2.0.6 (AAREADME: 20260521)",
            "note": ("взрослые; 18 отведений вычислены из референтных каналов (AR/LE); "
                     "записи без FZ/PZ отброшены; float16 в микровольтах; t_abs — "
                     "виртуальная шкала пациента (отрезки сеанса встык, между сеансами "
                     "сутки); метка — окно пересекает интервал seiz из .csv_bi")}
    json.dump(meta, open(os.path.join(OUT, "cache_meta.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in
                      ("n_windows", "n_seizure_windows", "positive_rate", "n_subjects",
                       "n_subjects_with_seizures", "n_files_used", "n_skipped_at_prescan",
                       "n_dropped", "sfreq_histogram_used_files", "by_official_split")},
                     ensure_ascii=False, indent=2))
    print("готово:", Xp)


if __name__ == "__main__":
    main()
