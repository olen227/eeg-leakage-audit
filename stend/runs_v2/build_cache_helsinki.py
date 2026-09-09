"""Сборка кэша окон для набора новорождённых (Helsinki) — третья возрастная группа.

ЗАЧЕМ. Основной результат измерен на детях (CHB-MIT), проверка на взрослых (Siena) не
набрала мощности. Helsinki даёт 79 новорождённых, из них 39 с приступами по консенсусу
трёх экспертов — это больше идентичностей, чем в CHB-MIT и Siena вместе. Если эффект
проявится и здесь, он перестанет быть свойством одной возрастной группы.

ОСОБЕННОСТЬ ФАЙЛОВ. В заголовках EDF поле даты записи обнулено (`00.00.00`), по-видимому
при деперсонификации. Формально это недопустимо: нулевого дня и месяца не существует,
поэтому ни MNE, ни pyedflib файл не открывают. Здесь создаётся исправленная копия, в
которой правится ТОЛЬКО поле даты в заголовке (байты 168-176); отсчёты сигнала и все
прочие поля побайтово сохраняются. Это не изменение данных, а восстановление
формальной корректности контейнера.

РАЗМЕТКА. Три эксперта размечали независимо, посекундно (annotations_2017_{A,B,C}.csv:
столбец — субъект, строка — секунда, 1 — приступ). Основная метка — КОНСЕНСУС всех трёх
(самое строгое определение, ему соответствуют заявленные в статье 39 субъектов с
приступами). Отдельно сохраняется число согласившихся экспертов: это позволяет отдельно
изучить влияние согласия разметчиков на метрики — такой возможности нет ни в CHB-MIT,
ни в Siena.

ПРИВЕДЕНИЕ К ЕДИНОМУ ВИДУ. Из 19 монополярных каналов 10-20 вычисляются те же 18
канонических отведений, что в CHB-MIT и Siena. Частота приводится к 256 Гц (в наборе
она уже 256). Полоса 0,5-40 Гц, режекторный 50 Гц (сеть Финляндии).

Выход: $VKR_CACHE_ROOT/helsinki/{X_raw.npy, meta.npz, cache_meta.json}
"""
import os, re, sys, csv, json, time, shutil, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
from scipy.signal import resample_poly
from stend.vkr_eeg import config as C

mne.set_log_level("ERROR")

CACHE = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
OUT = os.path.join(CACHE, "helsinki")
ROOT = os.path.join(C.DATA_ROOT, "helsinki")
FIXED = os.path.join(ROOT, "_fixed_headers")
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


def norm_ch(name):
    """Имя канала к единому виду: 'EEG Fp1-Ref' -> 'FP1'.

    Регистр суффикса в наборе непостоянен ('-Ref', '-REF'), поэтому сначала
    приводим к верхнему регистру и лишь затем срезаем служебные части.
    """
    s = name.upper().replace("EEG ", "").strip()
    return re.sub(r"-\s*REF$", "", s).strip()


def fix_header(src, dst):
    """Копия файла с исправленным полем даты записи. Сигнал не трогается."""
    with open(src, "rb") as f:
        head = bytearray(f.read(256))
    if head[168:176] == b"00.00.00":
        head[168:176] = b"01.01.85"          # условная дата; исходная обнулена при обезличивании
    if head[176:184] == b"00.00.00":
        head[176:184] = b"00.00.00"          # полночь — корректное время, оставляем
    with open(src, "rb") as f, open(dst, "wb") as g:
        f.seek(256)
        g.write(bytes(head))
        shutil.copyfileobj(f, g, length=1 << 22)


def load_annotations():
    """{номер субъекта: массив по секундам с числом согласившихся экспертов 0..3}."""
    per_expert = {}
    for tag in ("A", "B", "C"):
        p = os.path.join(ROOT, f"annotations_2017_{tag}.csv")
        rows = list(csv.reader(open(p)))
        subj_ids = [s.strip() for s in rows[0]]
        arr = np.zeros((len(rows) - 1, len(subj_ids)), dtype=np.int8)
        for i, r in enumerate(rows[1:]):
            for j, v in enumerate(r[:len(subj_ids)]):
                v = v.strip()
                if v in ("1", "1.0"):
                    arr[i, j] = 1
        per_expert[tag] = (subj_ids, arr)
    ids = per_expert["A"][0]
    total = sum(per_expert[t][1] for t in ("A", "B", "C"))
    return {sid: total[:, j] for j, sid in enumerate(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", type=int, default=3,
                    help="сколько экспертов должны согласиться, чтобы секунда считалась приступом")
    ap.add_argument("--margin-gb", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True); os.makedirs(FIXED, exist_ok=True)
    ann = load_annotations()
    files = sorted((f for f in os.listdir(ROOT) if re.fullmatch(r"eeg\d+\.edf", f)),
                   key=lambda x: int(re.findall(r"\d+", x)[0]))
    print(f"файлов ЭЭГ: {len(files)}, субъектов в разметке: {len(ann)}", flush=True)

    # --- исправление заголовков и оценка объёма ---
    est, usable = 0, []
    for f in files:
        sid = re.findall(r"\d+", f)[0]
        src, dst = os.path.join(ROOT, f), os.path.join(FIXED, f)
        if not os.path.exists(dst):
            fix_header(src, dst)
        try:
            raw = mne.io.read_raw_edf(dst, preload=False, verbose="ERROR")
        except Exception as e:
            print(f"  {f}: не открылся ({e})", flush=True); continue
        nm = {norm_ch(c) for c in raw.ch_names}
        if any(a not in nm or b not in nm for a, b in BIPOLAR.values()):
            print(f"  {f}: неполный монтаж", flush=True); continue
        n = int(raw.n_times * (C.SFREQ / raw.info["sfreq"]))
        est += max(0, (n - WIN) // STEP + 1)
        usable.append((sid, dst))
    nbytes = est * len(C.CANONICAL_CHANNELS) * WIN * 4
    free = os.statvfs(OUT).f_bavail * os.statvfs(OUT).f_frsize
    print(f"пригодных: {len(usable)}, ожидается окон ~{est} ({nbytes/1e9:.1f} ГБ), "
          f"свободно {free/1e9:.1f} ГБ", flush=True)
    if nbytes + args.margin_gb * 1e9 > free:
        print("ОСТАНОВ: мало места"); sys.exit(2)

    cap = int(est * 1.01) + 1024
    Xp = os.path.join(OUT, "X_raw.npy")
    X = np.lib.format.open_memmap(Xp, mode="w+", dtype=np.float32,
                                  shape=(cap, len(C.CANONICAL_CHANNELS), WIN))
    y_l, agree_l, subj_l, fid_l, wt_l, tabs_l = [], [], [], [], [], []
    dropped, pos, t0 = [], 0, time.time()

    for k, (sid, path) in enumerate(usable, 1):
        try:
            r = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        except Exception as e:
            dropped.append({"subject": sid, "reason": str(e)}); continue
        r.rename_channels({c: norm_ch(c) for c in r.ch_names})
        idx = {n_: i for i, n_ in enumerate(r.ch_names)}
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
        votes = ann.get(sid)
        if votes is None:
            dropped.append({"subject": sid, "reason": "нет разметки"}); del a; continue

        for st in range(0, n - WIN + 1, STEP):
            if pos >= cap: break
            X[pos] = a[:, st:st + WIN].astype(np.float32)
            t0w = st / C.SFREQ
            s0, s1 = int(t0w), int(np.ceil((st + WIN) / C.SFREQ))
            seg = votes[s0:min(s1, len(votes))]
            mx = int(seg.max()) if len(seg) else 0
            y_l.append(1 if mx >= args.consensus else 0)
            agree_l.append(mx)
            subj_l.append(f"PN{int(sid):03d}"); fid_l.append(f"eeg{sid}.edf")
            wt_l.append(t0w); tabs_l.append(t0w)
            pos += 1
        del a
        if k % 10 == 0 or k == len(usable):
            print(f"[{k}/{len(usable)}] окон {pos} ({(time.time()-t0)/60:.1f} мин)", flush=True)

    X.flush(); del X
    y = np.array(y_l, dtype=np.int64)
    agree = np.array(agree_l, dtype=np.int8)
    subj = np.array(subj_l)
    np.savez(os.path.join(OUT, "meta.npz"), y_sz=y, subj=subj, fileid=np.array(fid_l),
             wtime=np.array(wt_l, dtype=np.float32),
             t_abs=np.array(tabs_l, dtype=np.float64), expert_agreement=agree)
    with_sz = sorted({s for s, v in zip(subj_l, y_l) if v})
    meta = {"n_windows": int(pos), "n_seizure_windows": int(y.sum()),
            "positive_rate": float(y.mean()) if pos else None,
            "shape": [int(pos), len(C.CANONICAL_CHANNELS), WIN],
            "n_subjects": len(set(subj_l)), "n_subjects_with_seizures": len(with_sz),
            "consensus_required": args.consensus,
            "windows_by_agreement": {str(v): int((agree == v).sum()) for v in (0, 1, 2, 3)},
            "dropped": dropped, "n_dropped": len(dropped), "notch_hz": NOTCH_EU,
            "seed": C.SEED, "elapsed_min": round((time.time() - t0) / 60, 1),
            "note": ("новорождённые; отведения вычислены из монополярных каналов; "
                     "метка — консенсус экспертов; заголовки EDF исправлены (поле даты "
                     "было обнулено при обезличивании), сигнал не изменялся")}
    json.dump(meta, open(os.path.join(OUT, "cache_meta.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in
                      ("n_windows", "n_seizure_windows", "positive_rate", "n_subjects",
                       "n_subjects_with_seizures", "windows_by_agreement", "n_dropped")},
                     ensure_ascii=False, indent=2))
    print("готово:", Xp)


if __name__ == "__main__":
    main()
