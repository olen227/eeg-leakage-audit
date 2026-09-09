"""Зависимость измеряемого завышения от ёмкости модели: кривая по четырём архитектурам.

ЗАЧЕМ. Сравнение двух архитектур показало, что разрыв между наивным и честным протоколами
у более мощной модели ШИРЕ. Но две точки не отличают тренд от случайности: могло совпасть.
Кривая по четырём ёмкостям, различающимся только шириной свёрточных блоков, отвечает на
вопрос, растёт ли завышение С МОЩНОСТЬЮ, а не «у этой модели больше, чем у той».

ЧТО СУЩЕСТВЕННО. Точки различаются ТОЛЬКО числом каналов: архитектура, расписание обучения,
аугментация, число эпох, подвыборка и состав фолдов одинаковы. Иначе рост разрыва можно
было бы отнести на счёт смены обучения, а не ёмкости.

О ЗНАЧИМОСТИ ПРИ ЧЕТЫРЁХ ТОЧКАХ. Коэффициент ранговой корреляции при четырёх наблюдениях
даёт значение p по приближению, непригодному на таких объёмах (при идеальной монотонности
оно вырождается в ноль). Поэтому значимость считается точно: перебираются все 4! = 24
перестановки порядка моделей, и p есть доля перестановок, дающих корреляцию не хуже
наблюдённой. Минимально достижимое значение при четырёх точках равно 1/24 = 0,0417,
и это обстоятельство приводится в отчёте, чтобы p не читали как «сколь угодно малое».

Выход: outputs/track_b/capacity_curve.json
"""
import os, sys, json, time, platform, itertools, argparse
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.getcwd())

OUTDIR = os.path.join("outputs", "track_b")
# (подпись, файл разложения, число параметров если его нет в паспорте)
RUNS = [("компактная", "decomposition_24.json", 16673),
        ("78 тыс.", "decomposition_24_cap078.json", None),
        ("275 тыс.", "decomposition_24_cap275.json", None),
        ("1,03 млн", "decomposition_24_detv2.json", 1027105)]


def load(name, fname, hint):
    d = json.load(open(os.path.join(OUTDIR, fname)))
    n = d.get("passport", {}).get("detector_n_params") or hint
    if n is None:
        raise SystemExit(f"{fname}: число параметров неизвестно, точка кривой бесполезна")
    folds = {k: v for k, v in d["per_fold"].items() if "protocols" in v}
    return {"name": name, "file": fname, "n_params": int(n), "folds": folds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="f1")
    args = ap.parse_args()

    pts = [load(*r) for r in RUNS]
    keys = sorted(set.intersection(*[set(p["folds"]) for p in pts]))
    if len(keys) < 6:
        raise SystemExit(f"общих фолдов {len(keys)} — для выводов недостаточно")
    g = lambda p, k, pr: p["folds"][k]["protocols"][pr][args.metric]
    for p in pts:
        p["P0"] = np.array([g(p, k, "P0") for k in keys])
        p["P2"] = np.array([g(p, k, "P2") for k in keys])
        p["D"] = p["P0"] - p["P2"]
    pts.sort(key=lambda p: p["n_params"])

    y = np.array([p["D"].mean() for p in pts])
    x = np.log10([p["n_params"] for p in pts])
    obs = float(np.corrcoef(x, y)[0, 1])
    perms = [np.corrcoef(x, np.array(q))[0, 1] for q in itertools.permutations(y)]
    p_exact = sum(1 for r in perms if r >= obs) / len(perms)
    slope = float(np.polyfit(x, y, 1)[0])
    strict = bool(all(y[i] < y[i + 1] for i in range(len(y) - 1)))

    gain0 = float(pts[-1]["P0"].mean() - pts[0]["P0"].mean())
    gain2 = float(pts[-1]["P2"].mean() - pts[0]["P2"].mean())
    i_best = int(np.argmax([p["P2"].mean() for p in pts]))

    print("=" * 78)
    print("ЗАВИСИМОСТЬ ЗАВЫШЕНИЯ ОТ ЁМКОСТИ МОДЕЛИ")
    print("=" * 78)
    print(f"общих фолдов: {len(keys)}, метрика: {args.metric}\n")
    print(f"{'модель':12s} {'параметров':>11s} {'P0':>8s} {'P2':>8s} {'Δ':>9s}  прирост к предыдущей")
    print("-" * 78)
    for i, p in enumerate(pts):
        ex = ("" if i == 0 else
              f"  P0 {p['P0'].mean()-pts[i-1]['P0'].mean():+.4f} / "
              f"P2 {p['P2'].mean()-pts[i-1]['P2'].mean():+.4f}")
        print(f"{p['name']:12s} {p['n_params']:>11,} {p['P0'].mean():8.4f} "
              f"{p['P2'].mean():8.4f} {p['D'].mean():+9.4f}{ex}")
    print("-" * 78)
    print(f"\nстрого возрастает: {'ДА' if strict else 'НЕТ'}")
    print(f"корреляция log10(параметры) с Δ: {obs:.4f}")
    print(f"точное перестановочное p: {p_exact:.4f} "
          f"(минимально достижимое при четырёх точках {1/24:.4f})")
    print(f"наклон: {slope:+.4f} Δ на десятикратный рост ёмкости")
    print(f"\nот {pts[0]['n_params']:,} до {pts[-1]['n_params']:,} параметров: "
          f"P0 {gain0:+.4f}, P2 {gain2:+.4f}")
    print(f"доля прироста ёмкости, ушедшая в утечку: {(gain0-gain2)/gain0*100:.0f} %")
    print(f"честная F1 максимальна при {pts[i_best]['n_params']:,} параметрах"
          + ("" if i_best == len(pts) - 1 else " и далее СНИЖАЕТСЯ"))
    print("\nпарные сравнения с наименьшей моделью (критерий Уилкоксона):")
    wil = {}
    for p in pts[1:]:
        w = float(wilcoxon(p["D"], pts[0]["D"], alternative="greater").pvalue)
        wil[p["name"]] = w
        print(f"  {p['name']:12s} Δ {pts[0]['D'].mean():+.4f} → {p['D'].mean():+.4f}, "
              f"шире в {int((p['D']>pts[0]['D']).sum())}/{len(keys)} фолдах, p = {w:.4f}")

    # Попарное сравнение крайних архитектур выделяется отдельным артефактом: на него
    # ссылается текст главы, и без паспорта он был бы единственным результатом работы
    # без провенанса.
    a, b = pts[0], pts[-1]
    diff = b["D"] - a["D"]
    rng = np.random.default_rng(20260706)
    bs = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    pair = {"passport": {"script": "stend/runs_v2/run_capacity_curve.py",
                         "derived_from": "capacity_curve.json",
                         "n_folds": len(keys), "folds": keys,
                         "sources": [a["file"], b["file"]],
                         "bootstrap_reps": 10000, "bootstrap_seed": 20260706,
                         "python": platform.python_version(),
                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "note": "сравнение крайних точек кривой: компактная против усиленной архитектуры",
            "n_folds": len(keys), "folds": keys,
            "delta_v1_mean": float(a["D"].mean()), "delta_v2_mean": float(b["D"].mean()),
            "delta_v1_std": float(a["D"].std(ddof=1)), "delta_v2_std": float(b["D"].std(ddof=1)),
            "ratio": float(b["D"].mean() / a["D"].mean()),
            "increased_in": int((b["D"] > a["D"]).sum()),
            "wilcoxon_p": float(wilcoxon(b["D"], a["D"], alternative="greater").pvalue),
            "gain_diff_mean": float(diff.mean()),
            "gain_diff_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "gain_P0": gain0, "gain_P2": gain2,
            "per_protocol": {pr: {"v1": float(np.mean([a["folds"][k]["protocols"][pr][args.metric]
                                                       for k in keys])),
                                  "v2": float(np.mean([b["folds"][k]["protocols"][pr][args.metric]
                                                       for k in keys]))}
                             for pr in ("P0", "P1", "P1e", "P2")}}
    json.dump(pair, open(os.path.join(OUTDIR, "capacity_vs_leakage.json"), "w"),
              ensure_ascii=False, indent=2)

    # Клинический счёт по каждой точке кривой: пооконная F1 и событийная оценка
    # отвечают на разные вопросы и могут указывать на разные архитектуры как лучшие,
    # поэтому обе приводятся, а «лучшая конфигурация» определяется отдельно для каждой.
    clin = {}
    for pt in pts:
        tag = os.path.splitext(pt["file"])[0]
        ev = os.path.join(OUTDIR, f"event_scoring_{tag}.json")
        if not os.path.exists(ev):
            continue
        pr = json.load(open(ev))["protocols"]
        clin[pt["n_params"]] = {
            "P2_sensitivity": pr["P2"]["event_smoothed"]["sensitivity"],
            "P2_false_alarms_per_24h": pr["P2"]["event_smoothed"]["false_alarms_per_24h"],
            "P0_sensitivity": pr["P0"]["event_smoothed"]["sensitivity"],
            "P0_false_alarms_per_24h": pr["P0"]["event_smoothed"]["false_alarms_per_24h"]}
    best_clin = None
    if clin:
        # лучшей клинической конфигурацией считается наибольшая чувствительность:
        # пропуск приступа для оператора дороже лишней тревоги
        best_clin = max(clin, key=lambda k: clin[k]["P2_sensitivity"])
        print("\nклинический счёт по стандарту SzCORE (честный протокол P2):")
        for k in sorted(clin):
            m = "  <- наибольшая чувствительность" if k == best_clin else ""
            print(f"  {k:>9,} параметров: чувствительность {clin[k]['P2_sensitivity']:.3f}, "
                  f"ложных тревог в сутки {clin[k]['P2_false_alarms_per_24h']:.0f}{m}")
        print(f"  по пооконной F1 лучшей оказывается модель на {pts[i_best]['n_params']:,} "
              f"параметров — выбор метрики меняет, какая конфигурация считается лучшей")

    out = {"passport": {"script": "stend/runs_v2/run_capacity_curve.py",
                        "metric": args.metric, "n_folds": len(keys), "folds": keys,
                        "sources": [p["file"] for p in pts],
                        "python": platform.python_version(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "note": ("точки различаются только шириной свёрточных блоков; архитектура, "
                    "расписание обучения, аугментация, эпохи, подвыборка и состав фолдов "
                    "одинаковы"),
           "points": [{"name": p["name"], "n_params": p["n_params"],
                       "P0_mean": float(p["P0"].mean()), "P2_mean": float(p["P2"].mean()),
                       "delta_mean": float(p["D"].mean()),
                       "delta_std": float(p["D"].std(ddof=1)),
                       "delta_per_fold": {k: float(v) for k, v in zip(keys, p["D"])}}
                      for p in pts],
           "monotone_strict": strict, "pearson_log": obs,
           "permutation_p_exact": p_exact, "min_achievable_p": 1 / 24,
           "slope_per_decade": slope,
           "gain_P0_total": gain0, "gain_P2_total": gain2,
           "share_to_leakage": (gain0 - gain2) / gain0,
           "best_P2_at_n_params": pts[i_best]["n_params"],
           "clinical_by_capacity": clin,
           "best_clinical_at_n_params": best_clin,
           "P2_declines_after_peak": i_best != len(pts) - 1,
           "wilcoxon_vs_smallest": wil}
    path = os.path.join(OUTDIR, "capacity_curve.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print("\nсохранено:", path)
    print("сохранено:", os.path.join(OUTDIR, "capacity_vs_leakage.json"))


if __name__ == "__main__":
    main()
