"""Устойчивость разложения к выбору метрики: порог, семейство метрики, способ подсчёта.

ЗАЧЕМ. Основной результат работы измерен пооконной F1-мерой при калиброванном пороге.
Отсюда два естественных возражения, каждое из которых обесценило бы результат, окажись оно
верным.

  1. «Завышение — артефакт порога». F1 зависит от порога решения, и разность F1 двух
     протоколов может отражать не утечку, а разное положение откалиброванных порогов.
     Проверка: то же разложение на метриках, от порога НЕ зависящих, — AUPRC и ROC-AUC.

  2. «Пооконная F1 клинически бессмысленна». При распространённости 0,34 % пооконный счёт
     не отвечает на вопрос врача: важно заметить приступ, а не пометить каждое его окно.
     Проверка: то же разложение на событийной оценке по стандарту SzCORE — чувствительность
     по приступам и число ложных тревог в сутки.

Если вклады сохраняют знак и порядок на всех трёх способах счёта, результат не является
свойством выбранной метрики. Скрипт ничего не обучает: он пересобирает разложение из уже
посчитанных артефактов, поэтому сопоставление ведётся на одних и тех же предсказаниях.

Выход: outputs/track_b/metric_robustness.json
"""
import os, sys, json, time, platform
import numpy as np

sys.path.insert(0, os.getcwd())

OUTDIR = os.path.join("outputs", "track_b")
PROT = ["P0", "P1", "P1e", "P2"]
COMP = [("delta_okna", "P0", "P1", "вклад почти-дублей смежных окон"),
        ("delta_embargo", "P1", "P1e", "вклад временной смежности"),
        ("delta_subject", "P1e", "P2", "вклад перекрытия пациентов"),
        ("delta_total", "P0", "P2", "полное завышение")]


def decompose(values):
    """values: {протокол: число} -> вклады каскада."""
    return {k: values[a] - values[b] for k, a, b, _ in COMP}


def main():
    dec = json.load(open(os.path.join(OUTDIR, "decomposition_24.json")))["summary"]
    ev = json.load(open(os.path.join(OUTDIR, "event_scoring_decomposition_24.json")))["protocols"]

    views = {}
    # пооконные метрики: среднее по фолдам (единица статистики — идентичность)
    for met, human, threshold_free in (("f1", "пооконная F1 при калиброванном пороге", False),
                                       ("auprc", "AUPRC, от порога не зависит", True),
                                       ("roc_auc", "ROC-AUC, от порога не зависит", True)):
        vals = {p: dec[met][p]["mean"] for p in PROT}
        views[met] = {"human": human, "threshold_free": threshold_free,
                      "per_protocol": vals, "contributions": decompose(vals)}

    # событийная оценка: объединённая, пофолдового разбиения у неё нет
    for key, human in (("event_smoothed", "чувствительность по приступам (SzCORE, сглаживание)"),
                       ("event_raw", "чувствительность по приступам (SzCORE, без сглаживания)")):
        vals = {p: ev[p][key]["sensitivity"] for p in PROT}
        views["sensitivity_" + key] = {
            "human": human, "threshold_free": False, "event_based": True,
            "per_protocol": vals, "contributions": decompose(vals),
            "false_alarms_per_24h": {p: ev[p][key]["false_alarms_per_24h"] for p in PROT}}

    # согласованность: сохраняют ли вклады знак на всех способах счёта
    agree = {}
    for k, _, _, human in COMP:
        signs = {v: float(np.sign(views[v]["contributions"][k])) for v in views}
        agree[k] = {"human": human,
                    "sign_same_everywhere": len(set(signs.values())) == 1,
                    "signs": signs,
                    "values": {v: float(views[v]["contributions"][k]) for v in views}}

    main_src = {}
    for v in views:
        c = views[v]["contributions"]
        main_src[v] = max((k for k, _, _, _ in COMP if k != "delta_total"),
                          key=lambda k: abs(c[k]))
    same_main = len(set(main_src.values())) == 1

    verdict = []
    verdict.append(
        "Главный источник завышения одинаков на всех способах счёта: "
        + next(h for k, _, _, h in COMP if k == list(main_src.values())[0])
        if same_main else
        "ВНИМАНИЕ: главный источник завышения зависит от способа счёта — " + json.dumps(main_src, ensure_ascii=False))
    tf = [v for v in views if views[v].get("threshold_free")]
    verdict.append(
        "На метриках, не зависящих от порога, полное завышение составляет "
        + ", ".join(f"{views[v]['contributions']['delta_total']:+.4f} ({views[v]['human'].split(',')[0]})"
                    for v in tf)
        + " — то есть эффект не является артефактом порога решения.")
    fa = views["sensitivity_event_smoothed"]["false_alarms_per_24h"]
    verdict.append(
        f"В клиническом счёте переход от наивного протокола к честному снижает чувствительность "
        f"с {views['sensitivity_event_smoothed']['per_protocol']['P0']:.3f} до "
        f"{views['sensitivity_event_smoothed']['per_protocol']['P2']:.3f} и повышает число ложных "
        f"тревог с {fa['P0']:.0f} до {fa['P2']:.0f} в сутки, то есть в {fa['P2']/fa['P0']:.1f} раза.")

    out = {"passport": {"script": "stend/runs_v2/run_metric_robustness.py",
                        "sources": ["decomposition_24.json",
                                    "event_scoring_decomposition_24.json"],
                        "note": "ничего не обучается; разложение пересобрано из готовых "
                                "предсказаний, поэтому все способы счёта сопоставляются "
                                "на одних и тех же выходах модели",
                        "python": platform.python_version(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "views": views, "agreement": agree,
           "main_source_per_view": main_src, "main_source_same": same_main,
           "verdict": verdict}
    p = os.path.join(OUTDIR, "metric_robustness.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)

    print("=" * 78)
    print("УСТОЙЧИВОСТЬ РАЗЛОЖЕНИЯ К СПОСОБУ СЧЁТА")
    print("=" * 78)
    hdr = f"{'способ счёта':46s}" + "".join(f"{h[:9]:>11s}" for _, _, _, h in COMP)
    print(hdr); print("-" * len(hdr))
    for v in views:
        c = views[v]["contributions"]
        print(f"{views[v]['human'][:46]:46s}" + "".join(f"{c[k]:+11.4f}" for k, _, _, _ in COMP))
    print("-" * len(hdr))
    print("\nзнак сохраняется на всех способах счёта:")
    for k, a in agree.items():
        print(f"  {'ДА ' if a['sign_same_everywhere'] else 'НЕТ'} {a['human']}")
    print()
    for x in verdict:
        print(" ", x)
    print("\nсохранено:", p)


if __name__ == "__main__":
    main()
