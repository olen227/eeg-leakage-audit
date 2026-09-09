"""Подготовка кэша для корректного конвейера (v2).

Что делает:
 1. Читает master_dataset.npz (окна, нормированные ГЛОБАЛЬНОЙ статистикой train-субъектов).
 2. Обращает нормировку: X_raw = X_norm * sd + mu — чтобы нормировку можно было
    пересчитывать ПОФОЛДОВО (изоляция предобработки выполняется корректно, а не один раз глобально).
 3. Восстанавливает АБСОЛЮТНОЕ время каждого окна по 'File Start Time' из summary.
    В CHB-MIT соседние файлы идут подряд (разрыв ~3 с), поэтому пофайловое разбиение
    само по себе НЕ создаёт временного зазора — эмбарго осмысленно именно на общей шкале.
 4. Сохраняет X в .npy (memmap-friendly) и метаданные отдельно.

Выход: outputs/v2/X_raw.npy, outputs/v2/meta.npz
"""
import os, re, sys, json
import numpy as np

sys.path.insert(0, os.getcwd())

OUT = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
DATA_ROOT = os.environ.get("VKR_DATA_ROOT", os.path.join(os.getcwd(), "eeg_data"))


def parse_file_start_times(subject):
    """{filename: смещение начала файла в секундах от начала записи субъекта}.
    Учитывает переход через полночь: если время меньше предыдущего, добавляем сутки."""
    path = os.path.join(DATA_ROOT, "chbmit", subject, f"{subject}-summary.txt")
    txt = open(path).read()
    starts = {}
    prev_sec = None
    day = 0
    for b in re.split(r"File Name:\s*", txt)[1:]:
        name = b.split("\n", 1)[0].strip()
        m = re.search(r"File Start Time:\s*(\d+):(\d+):(\d+)", b)
        if not m:
            starts[name] = None
            continue
        h, mi, s = (int(x) for x in m.groups())
        sec = h * 3600 + mi * 60 + s          # в CHB-MIT встречается h>=24
        if prev_sec is not None and sec < prev_sec:
            day += 1
        prev_sec = sec
        starts[name] = sec + day * 86400
    # сдвигаем к нулю
    vals = [v for v in starts.values() if v is not None]
    base = min(vals) if vals else 0
    return {k: (None if v is None else v - base) for k, v in starts.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("читаю master_dataset.npz ...", flush=True)
    Z = np.load("master_dataset.npz")
    subj = Z["subj"]; fid = Z["fileid"]; y = Z["y_sz"]; wt = Z["wtime"]
    mu = Z["mu"].astype(np.float32); sd = Z["sd"].astype(np.float32)
    n = len(y)
    print(f"окон: {n}, каналов: {mu.shape[0]}", flush=True)

    # --- абсолютное время окна ---
    t_abs = np.full(n, np.nan, dtype=np.float64)
    missing = []
    for s in sorted(set(subj.tolist())):
        starts = parse_file_start_times(s)
        m = subj == s
        for f in sorted(set(fid[m].tolist())):
            off = starts.get(f)
            sel = m & (fid == f)
            if off is None:
                missing.append((s, f))
                continue
            t_abs[sel] = off + wt[sel]
    print(f"файлов без 'File Start Time': {len(missing)}", flush=True)

    # --- обращение нормировки, поблочная запись в .npy ---
    Xn = Z["X"]                      # (n, ch, win) float32, нормированные
    shape = Xn.shape
    print(f"обращаю нормировку и пишу X_raw.npy {shape} ...", flush=True)
    out_path = os.path.join(OUT, "X_raw.npy")
    Xr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)
    B = 4096
    for k in range(0, shape[0], B):
        Xr[k:k+B] = Xn[k:k+B] * sd[None, :, :] + mu[None, :, :]
        if (k // B) % 10 == 0:
            print(f"  {k}/{shape[0]}", flush=True)
    Xr.flush(); del Xr

    np.savez(os.path.join(OUT, "meta.npz"),
             y_sz=y, subj=subj, fileid=fid, wtime=wt, t_abs=t_abs,
             mu_global=mu, sd_global=sd)
    meta = {
        "n_windows": int(n),
        "n_seizure_windows": int(y.sum()),
        "positive_rate": float(y.mean()),
        "shape": list(map(int, shape)),
        "subjects": sorted(set(subj.tolist())),
        "files_without_start_time": [list(x) for x in missing],
        "note": ("X_raw.npy — окна ДО нормировки (нормировка обращена). "
                 "Нормировка применяется пофолдово по train-части (изоляция предобработки)."),
    }
    json.dump(meta, open(os.path.join(OUT, "cache_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2)[:800])
    print("готово:", out_path)


if __name__ == "__main__":
    main()
