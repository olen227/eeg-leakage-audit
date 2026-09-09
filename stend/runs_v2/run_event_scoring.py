"""Событийная оценка и временная постобработка — стандарт SzCORE.

ЗАЧЕМ. Пооконная F1 при распространённости 0,34 % — самый жёсткий из возможных способов
оценки, и он СИСТЕМАТИЧЕСКИ ЗАНИЖАЕТ качество детектора:

  1. Врачу не нужно, чтобы модель пометила каждое из 15 окон приступа. Нужно, чтобы
     приступ был ЗАМЕЧЕН. Пропуск 10 окон из 15 при пооконном счёте — провал,
     при событийном — успешное обнаружение.
  2. Одиночное ложное срабатывание среди тишины считается полноценной ошибкой,
     хотя любая практическая система сглаживает выход по времени.

Стандарт SzCORE (Dan et al., Epilepsia, 2025) предписывает приводить ОБЕ оценки:
пооконную (sample-based) и событийную (event-based), с числом ложных тревог в сутки.

ЧТО ЗДЕСЬ ДЕЛАЕТСЯ. Индексы оценочных окон восстанавливаются детерминированным
повтором построения фолдов (make_fold с тем же seed), поэтому переобучение НЕ требуется —
берутся уже сохранённые предсказания. Затем:

  - временная постобработка: срабатывание засчитывается, только если подряд идёт
    не менее MIN_CONSEC окон; разрывы короче MERGE_GAP окон заклеиваются;
  - события истины и предсказания склеиваются из подряд идущих окон;
  - событие считается обнаруженным при любом пересечении с предсказанным событием.

ЭТО НЕ УЛУЧШЕНИЕ МОДЕЛИ. Веса, пороги и предсказания не меняются — меняется только
способ подсчёта, приводимый к общепринятому. Разложение Δ пересчитывается в той же
схеме, поэтому вывод об утечке остаётся сопоставимым.

Выход: outputs/track_b/event_scoring_<tag>.json
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P

OUTDIR = os.path.join("outputs", "track_b")
PROTOCOLS = ["P0", "P1", "P1e", "P2"]


def smooth(pred, min_consec, merge_gap):
    """Временная постобработка: склеить короткие разрывы, убрать короткие всплески."""
    p = pred.astype(bool).copy()
    if merge_gap > 0:                      # заклеить разрывы
        i = 0
        while i < len(p):
            if not p[i]:
                j = i
                while j < len(p) and not p[j]:
                    j += 1
                if 0 < i and j < len(p) and (j - i) <= merge_gap:
                    p[i:j] = True
                i = j
            else:
                i += 1
    if min_consec > 1:                     # убрать короткие всплески
        i = 0
        while i < len(p):
            if p[i]:
                j = i
                while j < len(p) and p[j]:
                    j += 1
                if (j - i) < min_consec:
                    p[i:j] = False
                i = j
            else:
                i += 1
    return p.astype(int)


def to_events(mask):
    """Границы подряд идущих единиц: [(начало, конец_исключительно), ...]."""
    ev, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            ev.append((i, j)); i = j
        else:
            i += 1
    return ev


def event_scores(y, pred, win_sec):
    """Событийная оценка: событие истины обнаружено при любом пересечении."""
    tev, pev = to_events(y.astype(bool)), to_events(pred.astype(bool))
    detected = sum(1 for (a, b) in tev
                   if any(pa < b and pb > a for (pa, pb) in pev))
    matched = sum(1 for (pa, pb) in pev
                  if any(pa < b and pb > a for (a, b) in tev))
    fp_events = len(pev) - matched
    hours = len(y) * win_sec / 3600.0
    sens = detected / len(tev) if tev else None
    prec = matched / len(pev) if pev else None
    f1 = (2 * sens * prec / (sens + prec)) if (sens and prec) else 0.0
    return {"n_true_events": len(tev), "n_pred_events": len(pev),
            "detected_events": detected, "false_positive_events": fp_events,
            "sensitivity": sens, "precision": prec, "f1": f1,
            "false_alarms_per_24h": round(fp_events / hours * 24, 2) if hours else None,
            "eval_hours": round(hours, 2)}


def sample_scores(y, pred, win_sec):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    hours = len(y) * win_sec / 3600.0
    sens = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    return {"TP": tp, "FP": fp, "FN": fn,
            "sensitivity": sens, "precision": prec,
            "f1": (2 * sens * prec / (sens + prec)) if (sens and prec) else 0.0,
            "false_alarms_per_24h": round(fp / hours * 24, 2) if hours else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="decomposition_24")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--min-consec", type=int, default=3,
                    help="минимум подряд идущих окон для срабатывания (3 окна = 12 с)")
    ap.add_argument("--merge-gap", type=int, default=2,
                    help="разрывы короче этого числа окон заклеиваются")
    args = ap.parse_args()

    dec = json.load(open(os.path.join(OUTDIR, f"{args.tag}.json")))
    Z = np.load(os.path.join(OUTDIR, f"{args.tag}_pooled_predictions.npz"))
    M = np.load(os.path.join(args.cache, "meta.npz"), allow_pickle=True)
    y_all, subj, fid, t_abs = M["y_sz"], M["subj"], M["fileid"], M["t_abs"]
    ident_of = P.build_identity_map(sorted(set(subj.tolist())))

    # восстановление индексов eval детерминированным повтором построения фолдов
    folds_ok = [k for k, v in dec["per_fold"].items() if "protocols" in v]
    ev_idx, per_fold_len = [], {}
    for I in folds_ok:
        f = P.make_fold(ident_of, subj, fid, t_abs, I, seed=C.SEED,
                        embargo_sec=C.EMBARGO_SEC, y=y_all)
        ev_idx.append(f["eval"]); per_fold_len[I] = len(f["eval"])
    ev_idx = np.concatenate(ev_idx)

    # проверка: восстановленные метки обязаны совпасть с сохранёнными
    if not np.array_equal(y_all[ev_idx], Z["P0_y"]):
        print("ОСТАНОВ: восстановленные индексы не совпали с сохранёнными предсказаниями.")
        print(f"  восстановлено {len(ev_idx)}, сохранено {len(Z['P0_y'])}")
        sys.exit(2)
    print(f"индексы восстановлены и сверены: {len(ev_idx)} окон, "
          f"{int(y_all[ev_idx].sum())} приступных", flush=True)

    res = {}
    for pr in PROTOCOLS:
        if f"{pr}_pred" not in Z:
            continue
        y, pred = Z[f"{pr}_y"], Z[f"{pr}_pred"]
        # постобработка применяется ВНУТРИ каждого фолда: склеивать через границу
        # между разными пациентами нельзя
        sm, off = [], 0
        for I in folds_ok:
            n = per_fold_len[I]
            sm.append(smooth(pred[off:off + n], args.min_consec, args.merge_gap))
            off += n
        sm = np.concatenate(sm)
        res[pr] = {
            "sample_raw": sample_scores(y, pred, C.WIN_SEC),
            "sample_smoothed": sample_scores(y, sm, C.WIN_SEC),
            "event_raw": event_scores(y, pred, C.WIN_SEC),
            "event_smoothed": event_scores(y, sm, C.WIN_SEC),
        }

    # разложение в каждой из четырёх схем счёта
    dec_out = {}
    for scheme in ("sample_raw", "sample_smoothed", "event_raw", "event_smoothed"):
        g = lambda pr: res[pr][scheme]["f1"]
        dec_out[scheme] = {
            "delta_okna": g("P0") - g("P1"),
            "delta_embargo": g("P1") - g("P1e"),
            "delta_subject": g("P1e") - g("P2"),
            "delta_total": g("P0") - g("P2"),
        }

    out = {"passport": {"script": "stend/runs_v2/run_event_scoring.py",
                        "source": f"{args.tag}.json", "cache": args.cache,
                        "min_consec_windows": args.min_consec,
                        "merge_gap_windows": args.merge_gap,
                        "window_sec": C.WIN_SEC, "seed": C.SEED,
                        "python": platform.python_version(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "note": ("модель, пороги и предсказания НЕ менялись; изменён только способ "
                    "подсчёта — приведён к стандарту SzCORE (пооконная и событийная оценки)"),
           "protocols": res, "decomposition_by_scheme": dec_out}
    p = os.path.join(OUTDIR, f"event_scoring_{args.tag}.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)

    print(f"\n{'протокол':6s} | {'F1 окна':>8s} {'F1 окна+сгл':>12s} | "
          f"{'F1 событ':>9s} {'чувств':>7s} {'ложн/сут':>9s}")
    print("-" * 66)
    for pr in PROTOCOLS:
        if pr not in res: continue
        r = res[pr]
        print(f"{pr:6s} | {r['sample_raw']['f1']:8.4f} {r['sample_smoothed']['f1']:12.4f} | "
              f"{r['event_smoothed']['f1']:9.4f} {r['event_smoothed']['sensitivity']:7.3f} "
              f"{r['event_smoothed']['false_alarms_per_24h']:9.1f}")
    print("\nразложение Δ полное в разных схемах счёта:")
    for k, v in dec_out.items():
        print(f"  {k:18s} {v['delta_total']:+.4f}")
    print("\nсохранено:", p)


if __name__ == "__main__":
    main()
