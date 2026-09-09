"""Контракт входных данных модуля аудита утечки и его проверка.

НАЗНАЧЕНИЕ МОДУЛЯ. Дать ответ на вопрос: насколько заявленное качество детектора
завышено утечкой данных, и из чего это завышение складывается. Модуль применим к любому
набору ЭЭГ, приведённому к описанному ниже контракту, — не только к CHB-MIT.

=============================================================================
КОНТРАКТ
=============================================================================

Набор данных подаётся каталогом, содержащим три файла.

1. X_raw.npy — массив окон, форма (N, C, L), тип float32, формат .npy (читается memmap).
   N — число окон, C — число каналов, L — отсчётов в окне.
   Окна должны быть ДО нормировки: модуль нормирует их пофолдово по обучающей части,
   иначе изоляция предобработки нарушается (это одна из измеряемых утечек).

2. meta.npz — метаданные окон, все массивы длины N:
     y_sz    int64   1 — окно пересекается с приступом, иначе 0
     subj    <U..    идентификатор записи субъекта (например 'chb01', 'PN00')
     fileid  <U..    имя файла, из которого нарезано окно
     wtime   float32 время начала окна от начала ФАЙЛА, секунды
     t_abs   float64 время начала окна по ЕДИНОЙ шкале субъекта, секунды.
                     Нужно для эмбарго: если файлы записаны подряд, пофайловое
                     разбиение само по себе не создаёт временного зазора.
                     Допускается NaN, тогда эмбарго для этих окон не применяется.

3. cache_meta.json — паспорт среза: n_windows, n_seizure_windows, positive_rate,
   shape, а также сведения о происхождении (источник, число файлов, отброшенные файлы).

ГРУППИРОВКА ИДЕНТИЧНОСТЕЙ. Если в наборе есть записи одного и того же человека под
разными идентификаторами, они ОБЯЗАНЫ быть объединены — иначе «честное» разбиение по
субъектам само содержит утечку (измерено на внедрённых дублях: завышение F1 на 0,054 [0,009; 0,109]).
Известные пары задаются в protocols.KNOWN_DUPLICATES.

=============================================================================
ТРЕБОВАНИЯ К ОБЪЁМУ (получены измерением, см. FINDINGS 3.5, 3.17, 3.19)
=============================================================================

Ограничения носят характер эксплуатационных: при их нарушении модуль выдаст числа,
но они будут определяться шумом, а не измеряемой величиной.

  идентичностей >= 6   иначе минимально достижимое p парного критерия равно 2/2^n
                       и порог 0,05 НЕДОСТИЖИМ арифметически при любом размере эффекта
                       (при 5 идентичностях предел p = 0,0625)
  идентичностей >= 12  для запаса по значимости (предел p = 0,0005)
  приступных окон в оценочной части фолда >= 30
                       при 6-29 положительных примерах F1 неустойчива: на наборе Siena
                       разброс Δ между фолдами составил от -0,22 до +0,43 при среднем 0,06,
                       и значимости получить не удалось
  прогонов с разными seed >= 3 для ПОКОМПОНЕНТНОГО разложения
                       измерено: размах Δ_окна между инициализациями составил 146 % его
                       величины, тогда как у суммарного Δ — 12 %. Суммарное завышение
                       можно измерять одним прогоном, вклады компонентов — нельзя
  файлов на идентичность >= 2 для ПОКОМПОНЕНТНОГО разложения
                       при одном файле на субъекта пофайловое разбиение внутри субъекта
                       построить не из чего, и протоколы P1, P1e, P2 становятся
                       тождественными: всё завышение попадает в Δ_окна, а Δ_эмбарго и
                       Δ_пациент равны нулю ПО ПОСТРОЕНИЮ, а не по результату измерения.
                       Полное Δ при этом измеримо. Обнаружено на наборе Helsinki
                       (медиана 1 файл на субъекта против 29 в CHB-MIT)
  приступных окон всего >= 2000
                       ориентир по CHB-MIT (2 927) против Siena (1 045)

Проверка этих условий выполняется функцией check_dataset и попадает в отчёт аудита.
"""
import os
import numpy as np

REQUIRED_META = {
    "y_sz": ("int64", "метка приступа"),
    "subj": (None, "идентификатор записи субъекта"),
    "fileid": (None, "имя исходного файла"),
    "wtime": (None, "время окна от начала файла, с"),
    "t_abs": (None, "время окна по единой шкале субъекта, с"),
}

MIN_IDENTITIES_HARD = 6
MIN_IDENTITIES_COMFORT = 12
MIN_POS_PER_FOLD = 30
MIN_POS_TOTAL = 2000
MIN_FILES_PER_IDENTITY = 2


def check_dataset(cache_dir, identity_map=None):
    """Проверить каталог на соответствие контракту и оценить пригодность по объёму.

    Возвращает словарь с полями ok (структура корректна), errors, warnings,
    stats и power (оценка достаточности объёма).
    """
    errors, warnings = [], []
    xp = os.path.join(cache_dir, "X_raw.npy")
    mp = os.path.join(cache_dir, "meta.npz")
    cp = os.path.join(cache_dir, "cache_meta.json")

    for p, name in ((xp, "X_raw.npy"), (mp, "meta.npz")):
        if not os.path.exists(p):
            errors.append(f"нет обязательного файла {name}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings,
                "stats": None, "power": None}

    X = np.load(xp, mmap_mode="r")
    M = np.load(mp, allow_pickle=True)

    if X.ndim != 3:
        errors.append(f"X_raw.npy должен быть трёхмерным (N, C, L), получено {X.shape}")
    if X.dtype != np.float32:
        warnings.append(f"тип X_raw.npy {X.dtype}, ожидается float32")

    for k, (dt, desc) in REQUIRED_META.items():
        if k not in M:
            errors.append(f"в meta.npz нет массива '{k}' ({desc})")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings,
                "stats": None, "power": None}

    y, subj = M["y_sz"], M["subj"]
    n = len(y)
    if X.shape[0] < n:
        errors.append(f"в X_raw.npy {X.shape[0]} строк, в meta.npz {n} записей")
    elif X.shape[0] > n:
        warnings.append(f"в X_raw.npy {X.shape[0]} строк против {n} в meta.npz; "
                        "хвост не индексируется — это допустимо")
    for k in ("subj", "fileid", "wtime", "t_abs"):
        if k in M and len(M[k]) != n:
            errors.append(f"длина '{k}' ({len(M[k])}) не совпадает с длиной y_sz ({n})")

    if set(np.unique(y).tolist()) - {0, 1}:
        errors.append("y_sz должен содержать только 0 и 1")

    n_nan = int(np.isnan(M["t_abs"]).sum()) if "t_abs" in M else n
    if n_nan:
        warnings.append(f"у {n_nan} окон t_abs = NaN — для них эмбарго не применяется")

    subjects = sorted(set(subj.tolist()))
    ident = ({s: identity_map.get(s, f"ID_{s}") for s in subjects}
             if identity_map else {s: f"ID_{s}" for s in subjects})
    n_ident = len(set(ident.values()))
    n_pos = int(y.sum())

    fid = M["fileid"] if "fileid" in M else None
    files_per_ident = []
    if fid is not None:
        ident_arr = np.array([ident[s] for s in subj])
        for i in sorted(set(ident.values())):
            m = ident_arr == i
            files_per_ident.append(len(set(zip(subj[m].tolist(), fid[m].tolist()))))
    med_files = float(np.median(files_per_ident)) if files_per_ident else None

    stats = {"n_windows": int(n), "n_channels": int(X.shape[1]),
             "files_per_identity_median": med_files,
             "files_per_identity_min": (min(files_per_ident) if files_per_ident else None),
             "window_len": int(X.shape[2]), "n_subjects": len(subjects),
             "n_identities": n_ident, "n_seizure_windows": n_pos,
             "positive_rate": float(y.mean()),
             "windows_per_identity_median": float(np.median(
                 [int((np.array([ident[s] for s in subj]) == i).sum())
                  for i in sorted(set(ident.values()))])),
             "cache_meta_present": os.path.exists(cp)}

    # оценка достаточности объёма
    p_floor = 2 / 2 ** n_ident if n_ident < 40 else 0.0
    power = {
        "n_identities": n_ident,
        "min_achievable_p": p_floor,
        "significance_reachable": bool(p_floor < 0.05),
        "expected_pos_per_fold": round(n_pos / max(n_ident, 1) * 0.25, 1),
        "verdict": [],
    }
    if n_ident < MIN_IDENTITIES_HARD:
        power["verdict"].append(
            f"КРИТИЧНО: идентичностей {n_ident} < {MIN_IDENTITIES_HARD}. Минимально достижимое "
            f"p парного критерия равно {p_floor:.4f}; порог 0,05 недостижим арифметически "
            "при любом размере эффекта. Выводы о значимости делать нельзя.")
    elif n_ident < MIN_IDENTITIES_COMFORT:
        power["verdict"].append(
            f"ПРЕДУПРЕЖДЕНИЕ: идентичностей {n_ident}; предел p = {p_floor:.4f}. "
            "Значимость достижима только при полном совпадении знака во всех фолдах.")
    if med_files is not None and med_files < MIN_FILES_PER_IDENTITY:
        power["verdict"].append(
            f"КРИТИЧНО ДЛЯ РАЗЛОЖЕНИЯ: файлов на идентичность в среднем {med_files:.0f}. "
            "Пофайловое разбиение внутри субъекта построить не из чего, поэтому протоколы "
            "P1, P1e и P2 совпадают: Δ_эмбарго и Δ_пациент обратятся в ноль ПО ПОСТРОЕНИЮ, "
            "а всё завышение попадёт в Δ_окна. Полное Δ измеримо, покомпонентное "
            "разложение — нет.")
    if n_pos < MIN_POS_TOTAL:
        power["verdict"].append(
            f"ПРЕДУПРЕЖДЕНИЕ: приступных окон всего {n_pos} < {MIN_POS_TOTAL}. "
            "На наборе такого объёма (Siena, 1 045) мощности не хватило: разброс Δ "
            "между фолдами от -0,22 до +0,43 при среднем 0,06.")
    if power["expected_pos_per_fold"] < MIN_POS_PER_FOLD:
        power["verdict"].append(
            f"ПРЕДУПРЕЖДЕНИЕ: ожидается около {power['expected_pos_per_fold']} приступных "
            f"окон в оценочной части фолда (< {MIN_POS_PER_FOLD}). F1 при таком объёме "
            "неустойчива; опирайтесь на AUPRC и ROC-AUC.")
    if not power["verdict"]:
        power["verdict"].append("объём достаточен для измерения с оценкой значимости")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "stats": stats, "power": power}


def print_report(res):
    if res["stats"]:
        s = res["stats"]
        print("СОСТАВ НАБОРА")
        print(f"  окон {s['n_windows']}, каналов {s['n_channels']}, длина окна {s['window_len']}")
        print(f"  субъектов {s['n_subjects']}, идентичностей {s['n_identities']}")
        print(f"  приступных окон {s['n_seizure_windows']} "
              f"({s['positive_rate']*100:.3f} %)")
        if s.get("files_per_identity_median") is not None:
            print(f"  файлов на идентичность: медиана {s['files_per_identity_median']:.0f}, "
                  f"минимум {s['files_per_identity_min']}")
    print("\nСТРУКТУРА:", "соответствует контракту" if res["ok"] else "НЕ СООТВЕТСТВУЕТ")
    for e in res["errors"]:
        print("  ОШИБКА:", e)
    for w in res["warnings"]:
        print("  замечание:", w)
    if res["power"]:
        print("\nПРИГОДНОСТЬ ПО ОБЪЁМУ")
        print(f"  минимально достижимое p: {res['power']['min_achievable_p']:.6f}")
        for v in res["power"]["verdict"]:
            print("  ", v)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VKR_CACHE_ROOT", ".")
    print_report(check_dataset(d))
