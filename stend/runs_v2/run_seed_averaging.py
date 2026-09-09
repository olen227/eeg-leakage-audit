"""Усреднение покомпонентного разложения по нескольким инициализациям весов.

ЗАЧЕМ. Контракт модуля требует усреднять покомпонентное разложение не менее чем по трём
инициализациям: одиночный прогон давал вклад, размах которого между запусками оказывался
сопоставим с самой величиной. Настоящий скрипт закрывает это требование — собирает
разложения прогонов, различающихся ТОЛЬКО начальным значением генератора, и даёт
усреднённые вклады с характеристикой разброса.

ЧТО ПОКАЗАЛО ПРИМЕНЕНИЕ (4 инициализации, 12 общих фолдов, см. seed_averaging.json).
Вклад перекрытия пациентов воспроизводится хорошо (СКО между запусками 0,0085 при
величине 0,0875) и остаётся главным источником во всех прогонах. Вклад временной
смежности МЕНЯЕТ ЗНАК между запусками и отдельной величиной приводиться не может.
Существеннее же то, что разброс между инициализациями в 4-10 раз МЕНЬШЕ разброса между
пациентами: определяющим источником неопределённости служит состав выборки, поэтому
доверительные интервалы правильно строить ресэмплингом по идентичностям, как в работе
и сделано, а усреднение по seed уточняет оценку, но не заменяет его.

ЧТО СУЩЕСТВЕННО. Прогоны сопоставляются на ОДНОМ И ТОМ ЖЕ множестве фолдов: если у
разных seed посчитаны разные идентичности, разность вкладов смешивает влияние
инициализации с влиянием состава выборки. Берётся пересечение фолдов, и его размер
приводится в отчёте.

Разброс по инициализациям и разброс между фолдами — разные величины, и смешивать их
нельзя. Первый отвечает на вопрос «воспроизводится ли вклад при перезапуске», второй —
«одинаков ли вклад у разных пациентов». Приводятся оба.

Выход: outputs/track_b/seed_averaging.json
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
OUTDIR = os.path.join("outputs", "track_b")
COMP = {"delta_okna": ("P0", "P1", "вклад почти-дублей смежных окон"),
        "delta_embargo": ("P1", "P1e", "вклад временной смежности"),
        "delta_subject": ("P1e", "P2", "вклад перекрытия пациентов"),
        "delta_total": ("P0", "P2", "полное завышение")}


def load_run(path):
    """Пофолдовые значения метрики одного прогона. Промежуточные файлы допускаются:
    для разложения нужны только пофолдовые величины, объединённые предсказания — нет."""
    d = json.load(open(path))
    folds = {k: v for k, v in d.get("per_fold", {}).items() if "protocols" in v}
    seed = d.get("passport", {}).get("seed")
    if seed is None:                      # промежуточный файл паспорта не содержит
        m = os.path.basename(path)
        seed = "".join(ch for ch in m.split("seed")[-1] if ch.isdigit()) or "?"
    return {"path": path, "seed": seed, "folds": folds}


def contributions(folds, keys, met="f1"):
    """Вклады по фолдам: разности соседних протоколов каскада."""
    out = {}
    for name, (a, b, _) in COMP.items():
        out[name] = np.array([folds[k]["protocols"][a][met] - folds[k]["protocols"][b][met]
                              for k in keys], dtype=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="файлы разложений, различающиеся только seed")
    ap.add_argument("--metric", default="f1")
    args = ap.parse_args()

    runs = [load_run(p) for p in args.runs]
    runs = [r for r in runs if r["folds"]]
    if len(runs) < 3:
        print(f"ОСТАНОВ: прогонов с результатами {len(runs)}, для усреднения нужно >= 3")
        sys.exit(2)

    common = set(runs[0]["folds"])
    for r in runs[1:]:
        common &= set(r["folds"])
    keys = sorted(common)
    if not keys:
        print("ОСТАНОВ: у прогонов нет общих фолдов")
        sys.exit(2)

    print("=" * 70)
    print("УСРЕДНЕНИЕ РАЗЛОЖЕНИЯ ПО ИНИЦИАЛИЗАЦИЯМ")
    print("=" * 70)
    for r in runs:
        print(f"  seed {str(r['seed']):>9s}: фолдов {len(r['folds']):2d}  "
              f"{os.path.basename(r['path'])}")
    print(f"\nобщих фолдов: {len(keys)}")
    for r in runs:
        extra = len(r["folds"]) - len(keys)
        if extra:
            print(f"  seed {r['seed']}: {extra} фолдов не вошли — нет у остальных прогонов")

    per_run = {str(r["seed"]): contributions(r["folds"], keys, args.metric) for r in runs}
    res = {}
    print(f"\n{'компонент':34s} {'среднее':>9s} {'разброс по seed':>17s} "
          f"{'размах, % величины':>19s} {'знак':>7s}")
    print("-" * 92)
    for name, (_, _, human) in COMP.items():
        means = np.array([per_run[s][name].mean() for s in per_run])   # по одному числу на seed
        m = float(means.mean())
        rng = float(means.max() - means.min())
        rel = abs(rng / m) if m else float("nan")
        same = bool(np.all(means > 0) or np.all(means < 0))
        res[name] = {
            "human": human,
            "mean_over_seeds": m,
            "std_over_seeds": float(means.std(ddof=1)),
            "min": float(means.min()), "max": float(means.max()),
            "range": rng, "relative_range": rel,
            "sign_stable": same,
            "per_seed": {s: float(per_run[s][name].mean()) for s in per_run},
            "std_between_folds_mean": float(np.mean(
                [per_run[s][name].std(ddof=1) for s in per_run])),
        }
        print(f"{human:34s} {m:+9.4f} {means.std(ddof=1):17.4f} {rel*100:18.0f} % "
              f"{'устойчив' if same else 'МЕНЯЕТСЯ':>9s}")

    # Вердикт строится не по произвольному порогу на относительный размах, а по двум
    # содержательным признакам: сохраняется ли ЗНАК вклада при перезапуске и как разброс
    # между инициализациями соотносится с разбросом между пациентами. Второе существенно:
    # если инициализация вносит много меньше разнобоя, чем состав выборки, то усреднение
    # по seed не главный источник неопределённости, и интервалы следует строить
    # ресэмплингом по идентичностям, как и сделано в работе.
    verdict = []
    tot = res["delta_total"]
    verdict.append(
        f"Полное завышение усреднено по {len(runs)} инициализациям: "
        f"{tot['mean_over_seeds']:+.4f} (СКО между запусками {tot['std_over_seeds']:.4f}, "
        f"пределы {tot['min']:+.4f}…{tot['max']:+.4f}); знак "
        + ("сохраняется во всех прогонах." if tot["sign_stable"] else "МЕНЯЕТСЯ между прогонами."))
    flips = [v["human"] for k, v in res.items() if not v["sign_stable"]]
    verdict.append("Меняют знак между запусками: " + (", ".join(flips) if flips else "нет"))
    ratios = {k: (v["std_over_seeds"] / v["std_between_folds_mean"])
              for k, v in res.items() if v["std_between_folds_mean"]}
    verdict.append(
        "Разброс между инициализациями меньше разброса между пациентами в "
        + ", ".join(f"{1/r:.0f} раз ({res[k]['human']})" for k, r in sorted(ratios.items(),
                                                                           key=lambda x: x[1]))
        + " — определяющим источником неопределённости служит состав выборки, "
          "а не инициализация весов.")
    main_src = max((k for k in res if k != "delta_total"),
                   key=lambda k: abs(res[k]["mean_over_seeds"]))
    verdict.append(f"Главный источник по усреднённой оценке: {res[main_src]['human']} "
                   f"({res[main_src]['mean_over_seeds']:+.4f}).")

    out = {"passport": {"script": "stend/runs_v2/run_seed_averaging.py",
                        "metric": args.metric, "n_runs": len(runs),
                        "seeds": [str(r["seed"]) for r in runs],
                        "n_common_folds": len(keys), "common_folds": keys,
                        "sources": [r["path"] for r in runs],
                        "python": platform.python_version(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "note": ("разброс по инициализациям и разброс между фолдами — разные величины; "
                    "первый отвечает на вопрос о воспроизводимости при перезапуске, "
                    "второй — о различии между пациентами"),
           "components": res, "verdict": verdict}
    p = os.path.join(OUTDIR, "seed_averaging.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print("\n" + "=" * 70)
    for v in verdict:
        print(" ", v)
    print("\nсохранено:", p)


if __name__ == "__main__":
    main()
