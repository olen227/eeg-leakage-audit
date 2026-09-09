"""Протоколы разбиения P0/P1/P2 — корректная реализация (v2).

ГЛАВНОЕ МЕТОДИЧЕСКОЕ ОТЛИЧИЕ ОТ ИЮЛЬСКОГО КОДА
-----------------------------------------------
В июльской реализации P0, P1 и P2 оценивались на РАЗНЫХ оценочных множествах
(P0 — случайные 30 % всех окон, P2 — окна отложенных субъектов). Их разность
поэтому смешивает два эффекта: утечку и различие оценочных популяций.
Сам июльский код это признаёт (phase6_ablation.py:244–246): «это разные
оценочные схемы, поэтому их прямая разность НЕ является контролируемым
измерением вклада».

Здесь для каждого фолда оценочное множество ФИКСИРУЕТСЯ ОДНО, а протоколы
различаются ТОЛЬКО составом обучающей части. Тогда разность метрик —
контролируемое измерение вклада конкретного вида утечки.

Схема одного фолда (отложенная идентичность I):
  E            — отложенные файлы идентичности I (доля FILE_HOLDOUT)
  eval         — случайная половина окон из E                      (одна для всех протоколов)
  P0 train     = все прочие идентичности + ВСЕ окна I кроме eval
                 (в том числе вторая половина окон из E — соседние во времени
                  окна тех же приступов: утечка почти-дубликатов)
  P1 train     = все прочие идентичности + окна I из файлов вне E
                 (перекрытия окон нет, но субъект тот же)
  P1e train    = P1 с применением временного эмбарго по АБСОЛЮТНОЙ шкале
  P2 train     = только прочие идентичности                        (честный)

Разложение (все члены на одном eval):
  Δ_окна    = M(P0)  − M(P1)
  Δ_эмбарго = M(P1)  − M(P1e)
  Δ_субъект = M(P1e) − M(P2)
  Δ         = M(P0)  − M(P2) = Δ_окна + Δ_эмбарго + Δ_субъект   (тождество по построению)

Изоляция предобработки: статистики нормировки (mu, sd) считаются ПОФОЛДОВО
только по обучающей части каждого протокола.
"""
import numpy as np

# Документированный скрытый дубль CHB-MIT: chb21 — тот же пациент, что chb01,
# запись через 1.5 года. Группировка обязательна, иначе «честное» субъектное
# разбиение само содержит утечку, которую работа устраняет.
KNOWN_DUPLICATES = [("chb01", "chb21")]

# Отбор отложенных файлов СТРАТИФИЦИРОВАН по наличию приступа в файле.
# Приступные окна в CHB-MIT крайне разрежены (0,4 %), а файлы с приступами
# составляют ~15 % записей: случайный отбор файлов оставляет в eval единицы
# положительных примеров и делает метрику шумом. Стратификация — стандартный
# приём формирования выборки, утечки не создаёт (разбиение остаётся пофайловым).
FILE_HOLDOUT_SEIZURE = 0.50   # доля отложенных файлов С приступами
FILE_HOLDOUT_PLAIN = 0.30     # доля отложенных файлов БЕЗ приступов
EVAL_FRAC_IN_E = 0.50         # доля окон отложенных файлов, попадающих в eval


def build_identity_map(subjects, extra_groups=None):
    """subject -> identity. Объединяет известные дубли и дополнительно найденные."""
    groups = [set(g) for g in KNOWN_DUPLICATES]
    for g in (extra_groups or []):
        groups.append(set(g))
    # транзитивное замыкание
    merged = []
    for g in groups:
        hit = [m for m in merged if m & g]
        for m in hit:
            merged.remove(m)
            g = g | m
        merged.append(g)
    ident = {}
    for s in sorted(subjects):
        lab = None
        for m in merged:
            if s in m:
                lab = "ID_" + "+".join(sorted(m))
                break
        ident[s] = lab or f"ID_{s}"
    return ident


def embargo_mask_abs(t_abs, subj, train_idx, eval_idx, embargo_sec):
    """Убрать из train окна, отстоящие по АБСОЛЮТНОМУ времени менее чем на
    embargo_sec от любого eval-окна ТОГО ЖЕ субъекта.

    В CHB-MIT соседние файлы записаны подряд, поэтому разделение по файлам само
    по себе не создаёт временного зазора; эмбарго считается на общей шкале.
    Возвращает подмножество train_idx, прошедшее эмбарго.

    ОКНА БЕЗ АБСОЛЮТНОГО ВРЕМЕНИ (t_abs = NaN) СОХРАНЯЮТСЯ. Контракт данных
    (dataset_contract.py) объявляет NaN допустимым значением и предписывает не
    применять к таким окнам эмбарго. Наивная запись `dist >= embargo_sec` даёт на
    NaN ложь и удаляет их все, отчего у субъекта без разметки времени (в CHB-MIT
    это chb24, 12 файлов) обучающая часть P1e совпадает с P2, и вклад перекрытия
    пациентов обращается в ноль ПО ПОСТРОЕНИЮ, а не по результату измерения.
    Такие окна и в eval, и в train исключаются из расчёта расстояний явно.
    """
    keep = np.ones(len(train_idx), dtype=bool)
    for s in np.unique(subj[eval_idx]):
        ev_sel = eval_idx[subj[eval_idx] == s]
        ev_t = np.sort(t_abs[ev_sel][~np.isnan(t_abs[ev_sel])])
        if len(ev_t) == 0:
            continue
        loc = np.where((subj[train_idx] == s) & ~np.isnan(t_abs[train_idx]))[0]
        if len(loc) == 0:
            continue
        tt = t_abs[train_idx[loc]]
        pos = np.searchsorted(ev_t, tt)
        left = np.where(pos > 0, ev_t[np.clip(pos - 1, 0, len(ev_t) - 1)], -np.inf)
        right = np.where(pos < len(ev_t), ev_t[np.clip(pos, 0, len(ev_t) - 1)], np.inf)
        dist = np.minimum(np.abs(tt - left), np.abs(tt - right))
        keep[loc] = dist >= embargo_sec
    return train_idx[keep]


def make_fold(identity_of_subject, subj, fileid, t_abs, held_identity, seed,
              embargo_sec=60.0, y=None):
    """Строит индексы eval и train-части всех протоколов для одного фолда.

    y — метки приступа по окнам; нужны для стратификации отбора отложенных файлов.
    """
    rng = np.random.default_rng(seed)
    ident = np.array([identity_of_subject[s] for s in subj])
    in_I = ident == held_identity
    others = np.where(~in_I)[0]

    # отложенные файлы идентичности I — отбор стратифицирован по наличию приступа
    files_I = sorted(set(zip(subj[in_I].tolist(), fileid[in_I].tolist())))
    if y is None:
        has_sz = {f: False for f in files_I}
    else:
        has_sz = {}
        for (a, b) in files_I:
            sel = (subj == a) & (fileid == b)
            has_sz[(a, b)] = bool(y[sel].sum() > 0)
    sz_files = [f for f in files_I if has_sz[f]]
    pl_files = [f for f in files_I if not has_sz[f]]
    E = set()
    for group, frac in ((sz_files, FILE_HOLDOUT_SEIZURE), (pl_files, FILE_HOLDOUT_PLAIN)):
        if not group:
            continue
        pm = rng.permutation(len(group))
        k = max(1, int(round(len(group) * frac)))
        E |= {group[i] for i in pm[:k]}

    key = np.array([f"{a}|{b}" for a, b in zip(subj, fileid)])
    E_key = {f"{a}|{b}" for a, b in E}
    in_E = in_I & np.isin(key, list(E_key))

    idx_E = np.where(in_E)[0]
    pe = rng.permutation(len(idx_E))
    n_eval = int(round(len(idx_E) * EVAL_FRAC_IN_E))
    eval_idx = np.sort(idx_E[pe[:n_eval]])
    restE_idx = np.sort(idx_E[pe[n_eval:]])        # соседние окна тех же файлов

    idx_I_notE = np.where(in_I & ~in_E)[0]

    P0 = np.sort(np.concatenate([others, idx_I_notE, restE_idx]))
    P1 = np.sort(np.concatenate([others, idx_I_notE]))
    P1e = embargo_mask_abs(t_abs, subj, P1, eval_idx, embargo_sec)
    P2 = np.sort(others)

    # Вырождение ступени каскада: если обучающая часть протокола совпала с соседней,
    # соответствующий вклад разложения равен нулю ПО ПОСТРОЕНИЮ и измерением не
    # является. Молча такое пропускать нельзя — ноль неотличим от измеренного нуля.
    degenerate = [f"{a} == {b}" for a, b, A, B in
                  (("P0", "P1", P0, P1), ("P1", "P1e", P1, P1e), ("P1e", "P2", P1e, P2))
                  if len(A) == len(B) and np.array_equal(A, B)]

    return {
        "eval": eval_idx,
        "train": {"P0": P0, "P1": P1, "P1e": P1e, "P2": P2},
        "held_identity": held_identity,
        "n_files_held": len(E),
        "n_files_total_I": len(files_I),
        "embargo_removed": int(len(P1) - len(P1e)),
        "n_eval_no_abs_time": int(np.isnan(t_abs[eval_idx]).sum()),
        "degenerate_steps": degenerate,
    }


def fold_norm_stats(X, train_idx, chunk=8192):
    """mu/sd по каналам ТОЛЬКО по обучающей части фолда (изоляция предобработки)."""
    n_ch = X.shape[1]
    s1 = np.zeros(n_ch, dtype=np.float64)
    s2 = np.zeros(n_ch, dtype=np.float64)
    cnt = 0
    for k in range(0, len(train_idx), chunk):
        b = X[train_idx[k:k + chunk]]
        s1 += b.sum(axis=(0, 2))
        s2 += (b.astype(np.float64) ** 2).sum(axis=(0, 2))
        cnt += b.shape[0] * b.shape[2]
    mu = s1 / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 1e-20)) + 1e-7
    return mu.astype(np.float32).reshape(1, -1, 1), sd.astype(np.float32).reshape(1, -1, 1)
