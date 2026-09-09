"""Вклад скрытого дубля субъекта — контролируемое измерение на документированной паре.

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ. Вклад скрытого дубля измерялся вариантом
`без_удаления_дублей` в run_ablation_components.py, и это измерение было НЕВЕРНЫМ.
Оценочное множество фолда ID_chb01+chb21 составлено окнами ОБЕИХ записей — 6 660
окон chb01 и 4 514 окон chb21. Вариант оставлял в обучении партнёра по дублю,
выбирая его так: `keep_partner = "chb01" if "chb21" in ev_subj else "chb21"`.
Поскольку в оценке присутствуют оба, партнёром оказывался chb01 — тот самый
субъект, чьи окна составляют 60 % оценочного множества. Тем самым в обучение
попадали СОСЕДНИЕ ПО ВРЕМЕНИ окна тех же файлов, то есть перекрытие почти-дублей
окон (канал утечки протокола P0), а вовсе не скрытая повторная регистрация.
Измеренная величина +0,1683 смешивала два разных источника завышения.

ЧТО ДЕЛАЕТ НАСТОЯЩИЙ СКРИПТ. Оценочное множество сужается до окон ОДНОЙ записи
пары, а партнёром берётся ВТОРАЯ запись, в оценке не представленная вовсе. Тогда
между обучающей и оценочной частями нет ни одного общего файла и ни одного
соседнего во времени окна: единственное, что добавляет наивный вариант, — это
сигнал того же человека, зарегистрированный полутора годами позже (или раньше)
под другим идентификатором. Разность и есть вклад скрытого дубля.

  eval        = окна записи D из отложенных файлов идентичности
  честно      = обучение на прочих идентичностях (ни chb01, ни chb21)
  наивно      = то же обучение плюс ВСЕ окна записи-партнёра P (P != D)
  Δ_дубль     = F1(наивно) − F1(честно) на одном и том же eval

Измерение выполняется в ОБЕ стороны (D = chb21 при партнёре chb01 и наоборот):
записи различаются длиной и числом приступов, и одностороннее измерение зависело
бы от того, какую из них выбрали оценочной.

ОГРАНИЧЕНИЕ, КОТОРОЕ ОБЯЗАНО ПРИВОДИТЬСЯ. Приступных окон в оценочной части
здесь десятки, что ниже собственного порога модуля (30 окон на фолд), поэтому
величина приводится вместе с числом приступных окон и не заменяет интервальную
оценку на внедрённых дублях (run_injected_duplicates.py).

ЗАПУСК:
    python stend/runs_v2/run_hidden_duplicate.py --cache <каталог кэша>

Выход: outputs/track_b/hidden_duplicate.json
"""
import os, sys, json, time, platform, argparse

import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_decomposition import (
    get_device, train_detector_mm, predict_mm, metrics_of,
    calibrate_threshold_on_train, slice_provenance, OUTDIR)

PAIR = ("chb01", "chb21")
IDENTITY = "ID_chb01+chb21"


def run_variant(X, y, tr, ev, device, epochs, seed):
    """Обучить, откалибровать порог по обучающей части, оценить на ev."""
    mu, sd = P.fold_norm_stats(X, tr)
    det = train_detector_mm(X, tr, y, mu, sd, device, epochs, seed)
    thr, _ = calibrate_threshold_on_train(det, X, tr, y, mu, sd, device, seed)
    m = metrics_of(y[ev], predict_mm(det, X, ev, mu, sd, device), thr=thr)
    m["train_n"] = int(len(tr))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device()
    t0 = time.time()

    X = np.load(os.path.join(args.cache, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(args.cache, "meta.npz"), allow_pickle=True)
    y, subj, fid, t_abs = M["y_sz"], M["subj"], M["fileid"], M["t_abs"]
    subjects = sorted(set(subj.tolist()))
    for s in PAIR:
        if s not in subjects:
            raise SystemExit(f"ОСТАНОВ: в наборе нет записи {s} — измерять нечего")

    ident_of = P.build_identity_map(subjects)
    fold = P.make_fold(ident_of, subj, fid, t_abs, IDENTITY, seed=C.SEED,
                       embargo_sec=C.EMBARGO_SEC, y=y)
    ev_all = fold["eval"]
    train_honest = fold["train"]["P2"]        # ни chb01, ни chb21

    print("=" * 78)
    print("ВКЛАД СКРЫТОГО ДУБЛЯ СУБЪЕКТА (контролируемое измерение)")
    print("=" * 78)
    print(f"устройство: {device} | кэш: {args.cache}")
    print(f"оценочное множество фолда {IDENTITY}: {len(ev_all)} окон, "
          f"{int(y[ev_all].sum())} приступных")

    # честный вариант обучается ОДИН раз: обучающая часть у обеих сторон одна и та же,
    # различаются только оценочные подмножества
    directions = {}
    for d in PAIR:
        p = PAIR[1] if d == PAIR[0] else PAIR[0]      # партнёр
        ev = ev_all[subj[ev_all] == d]
        n_pos = int(y[ev].sum())
        print(f"\n--- оценка на {d}, партнёр в обучении {p} ---")
        print(f"  окон {len(ev)}, приступных {n_pos}")
        if n_pos < 3:
            directions[d] = {"skipped": f"приступных окон {n_pos} (< 3)"}
            print("  ПРОПУСК: приступных окон слишком мало")
            continue

        honest = run_variant(X, y, train_honest, ev, device, args.epochs, C.SEED)
        # наивный вариант: партнёр целиком в обучении. В оценке его окон нет,
        # поэтому общих файлов и смежных во времени окон между частями не возникает.
        partner_idx = np.where(subj == p)[0]
        assert not np.intersect1d(partner_idx, ev).size, "партнёр не должен встречаться в eval"
        naive_tr = np.sort(np.concatenate([train_honest, partner_idx]))
        naive = run_variant(X, y, naive_tr, ev, device, args.epochs, C.SEED)

        directions[d] = {
            "eval_subject": d, "partner_in_train": p,
            "eval_n": int(len(ev)), "eval_pos": n_pos,
            "honest": honest, "naive": naive,
            "delta_f1": naive["f1"] - honest["f1"],
            "delta_auprc": ((naive["auprc"] - honest["auprc"])
                            if (naive["auprc"] is not None and honest["auprc"] is not None)
                            else None),
            "n_partner_windows_added": int(len(partner_idx)),
        }
        print(f"  честно  F1={honest['f1']:.4f} AUPRC={honest['auprc']:.4f} train={honest['train_n']}")
        print(f"  наивно  F1={naive['f1']:.4f} AUPRC={naive['auprc']:.4f} train={naive['train_n']}")
        print(f"  Δ_дубль F1={directions[d]['delta_f1']:+.4f}")

    got = [v for v in directions.values() if "delta_f1" in v]
    mean_f1 = float(np.mean([v["delta_f1"] for v in got])) if got else None
    note, h = slice_provenance(args.cache)

    out = {
        "passport": {"seed": C.SEED, "device": device, "python": platform.python_version(),
                     "torch": torch.__version__, "numpy": np.__version__,
                     "script": "stend/runs_v2/run_hidden_duplicate.py",
                     "epochs": args.epochs, "cache_dir": args.cache,
                     "data_slice": note, "data_sha256": h,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "elapsed_sec": round(time.time() - t0, 1)},
        "scheme": ("оценка на окнах ОДНОЙ записи пары; партнёр по дублю в оценке "
                   "не представлен вовсе, поэтому наивный вариант добавляет в обучение "
                   "только сигнал того же человека под другим идентификатором, "
                   "а не соседние во времени окна тех же файлов"),
        "supersedes": ("ablation_components.json -> без_удаления_дублей: там оценочное "
                       "множество содержало окна ОБЕИХ записей пары, и партнёром "
                       "оказывалась запись, представленная в оценке, отчего величина "
                       "смешивала вклад скрытого дубля с вкладом почти-дублей смежных окон"),
        "directions": directions,
        "mean_delta_f1": mean_f1,
        "caveat": ("приступных окон в оценочной части десятки, что ниже порога "
                   "применимости модуля (30 на фолд); величина приводится вместе "
                   "с числом приступных окон, а интервальная оценка вклада строится "
                   "на внедрённых дублях (injected_duplicates.json)"),
    }
    path = os.path.join(OUTDIR, "hidden_duplicate.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    if mean_f1 is not None:
        print(f"вклад скрытого дубля, среднее по двум направлениям: {mean_f1:+.4f}")
    print("сохранено:", path)


if __name__ == "__main__":
    main()
