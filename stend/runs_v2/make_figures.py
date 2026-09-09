#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Построение рисунков для ВКР по артефактам outputs/track_b/*.json.

Правила проекта:
  * только matplotlib (без seaborn);
  * все числа читаются из JSON-артефактов, ничего не задаётся константами;
  * подписи по-русски, шрифт 8-9 pt, dpi=300, bbox_inches='tight';
  * различимость в Ч/Б печати (штриховка + типы линий, не только цвет);
  * сохранение и .png, и .pdf в figures_out/.

Каждая функция make_fig_* самодостаточна: читает свои артефакты и пишет свои файлы.
Дописывать новые функции В КОНЕЦ файла, не ломая существующие,
и добавлять их вызов в блок __main__.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Общие пути и настройки
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACK_B_DIR = os.path.join(PROJECT_ROOT, "outputs", "track_b")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures_out")


# Действующие источники. Прогон на 6 субъектах опровергнут прогоном на полном наборе
# (см. FINDINGS 3.0), поэтому рисунки строятся по артефактам 23 идентичностей.
# Для абляции и частотных полос версии на полном наборе нет — эти рисунки строятся
# по шестисубъектным данным и ЯВНО ПОМЕЧЕНЫ на самом рисунке.
CURRENT = {
    "decomposition_v2.json": "decomposition_24.json",
    "bootstrap_contributions.json": "bootstrap_contributions_24embargofix.json",
}
SLICE_NOTE = "полный набор CHB-MIT, 23 идентичности"
SLICE_NOTE_OLD = "6 субъектов, 5 идентичностей (прогон на полном наборе не выполнялся)"


def _operator_current() -> dict:
    """Действующие данные оператора в старой схеме.

    operator_v2.json (21 обучающая идентичность, 15 фолдов) имеет иную структуру,
    чем operator_subject_cv.json (3 идентичности, 10 фолдов). Приводим к общему виду,
    чтобы рисунки не переписывать. Конфигурация «исходная» — та же архитектура, что
    в operator_subject_cv, поэтому величины сопоставимы.
    """
    v2 = _load_json_raw("operator_v2.json")
    cfg = "исходная (как в июле)"
    a = v2["aggregate"][cfg]
    folds = v2["per_config"][cfg]
    vals = [f["eval"]["roc_auc"] for f in folds]
    eers = [f["eval"]["eer"] for f in folds]
    return {
        "aggregate": {
            "roc_auc_identity": {"mean": a["roc_auc_mean"], "std": a["roc_auc_std"],
                                 "min": min(vals), "max": max(vals)},
            "eer_identity": {"mean": a["eer_mean"], "std": a["eer_std"],
                             "min": min(eers), "max": max(eers)},
        },
        "folds": [{"held": f["held"], "roc_auc_identity": f["eval"]["roc_auc"],
                   "eer_identity": f["eval"]["eer"]} for f in folds],
        "n_train_identities": v2["n_train_identities_per_fold"],
    }


def _load_json_raw(name: str) -> dict:
    path = os.path.join(TRACK_B_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Артефакт не найден: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_json(name: str) -> dict:
    """Прочитать артефакт из outputs/track_b/, подставив действующую версию."""
    if name == "operator_subject_cv.json":
        return _operator_current()
    name = CURRENT.get(name, name)
    path = os.path.join(TRACK_B_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Артефакт не найден: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _apply_rc() -> None:
    """Типографика под печать ВКР: 8-9 pt, без семейства seaborn."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "hatch.linewidth": 0.5,
        }
    )


def _save(fig, stem: str) -> list:
    """Сохранить фигуру в .png и .pdf, вернуть список путей."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Рисунок 3.1 «Каскад протоколов»
# --------------------------------------------------------------------------- #

def make_fig_3_1() -> list:
    """Каскад протоколов P0 -> P1 -> P1e -> P2: F1 и AUPRC (объединённая оценка).

    Источник: outputs/track_b/decomposition_v2.json, summary.pooled.metrics.
    """
    _apply_rc()
    data = _load_json("decomposition_v2.json")
    metrics = data["summary"]["pooled"]["metrics"]

    keys = ["P0", "P1", "P1e", "P2"]
    labels = [
        "P0\nоконное\n(утечка)",
        "P1\nпофайловое",
        "P1e\n+ эмбарго",
        "P2\nчестное\n(новый пациент)",
    ]
    # Штриховка вместо цвета — для различимости в Ч/Б печати.
    hatches = ["////", "\\\\\\\\", "xxxx", ""]
    facecolors = ["0.85", "0.75", "0.65", "0.35"]

    panels = [("f1", "F1-мера"), ("auprc", "AUPRC")]

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.1))
    for ax, (mkey, mtitle) in zip(axes, panels):
        values = [float(metrics[k][mkey]) for k in keys]
        xs = list(range(len(keys)))
        bars = ax.bar(
            xs,
            values,
            width=0.66,
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        for bar, hatch, fc in zip(bars, hatches, facecolors):
            bar.set_facecolor(fc)
            bar.set_hatch(hatch)

        top = max(values)
        for x, v in zip(xs, values):
            ax.text(
                x,
                v + top * 0.035,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=4,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, top * 1.28)
        ax.set_ylabel(mtitle)
        ax.set_title(mtitle, pad=4)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.7", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    n = int(metrics["P0"]["n"])
    n_pos = int(metrics["P0"]["n_pos"])
    fig.suptitle(
        "Рисунок 3.4 — Каскад протоколов: деградация метрик при устранении утечки\n"
        f"({SLICE_NOTE}; объединённая оценка, окон {n}, приступных {n_pos})",
        fontsize=9,
        y=1.06,
    )
    fig.tight_layout()
    return _save(fig, "fig-3-4-kaskad")


# --------------------------------------------------------------------------- #
# Рисунок 3.2 «Вклады в завышение метрик»
# --------------------------------------------------------------------------- #

def make_fig_3_2() -> list:
    """Вклады компонентов утечки в завышение F1 с бутстрэп-ДИ 95 %.

    Источник: outputs/track_b/bootstrap_contributions.json, contributions.f1.
    Точка — point_estimate, усы — [lo; hi]. Штриховка = ДИ накрывает ноль.
    """
    _apply_rc()
    data = _load_json("bootstrap_contributions.json")
    contrib = data["contributions"]["f1"]

    keys = ["delta_okna_pochti_dubli", "delta_embargo", "delta_subject", "delta_total"]
    labels = [
        "Δ_окна\n(почти-дубли)",
        "Δ_эмбарго",
        "Δ_субъект",
        "Δ_итог\n(P0 → P2)",
    ]

    points, los, his, sig = [], [], [], []
    for k in keys:
        rec = contrib[k]
        points.append(float(rec["point_estimate"]))
        los.append(float(rec["lo"]))
        his.append(float(rec["hi"]))
        sig.append(bool(rec["excludes_zero"]))

    # Снизу вверх: последний ключ (Δ_итог) окажется наверху.
    ys = list(range(len(keys)))[::-1]

    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="-", zorder=2)

    for y, p, lo, hi, is_sig in zip(ys, points, los, his, sig):
        if is_sig:
            ax.barh(
                y, p, height=0.55, facecolor="0.45", edgecolor="black",
                linewidth=0.8, zorder=3,
            )
        else:
            ax.barh(
                y, p, height=0.55, facecolor="white", edgecolor="black",
                linewidth=0.8, hatch="////", zorder=3,
            )
        ax.errorbar(
            p, y,
            xerr=[[p - lo], [hi - p]],
            fmt="none", ecolor="black", elinewidth=0.9, capsize=3,
            capthick=0.9, zorder=4,
        )

    xmin = min(los + points + [0.0])
    xmax = max(his + points + [0.0])
    span = xmax - xmin
    ax.set_xlim(xmin - 0.10 * span, xmax + 0.60 * span)

    for y, p, lo, hi, is_sig in zip(ys, points, los, his, sig):
        _c = lambda v: f"{v:+.4f}".replace(".", ",")
        txt = f"{_c(p)}  [{_c(lo)}; {_c(hi)}]"
        if not is_sig:
            txt += "  — неотличим от нуля"
        # подпись ставится правее ВСЕГО столбца: у отрицательного вклада правый
        # конец интервала лежит около нуля, и подпись налезала бы на сам столбец
        ax.text(max(hi, p, 0.0) + 0.02 * span, y, txt, va="center", ha="left",
                fontsize=7.5, zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel("Вклад в завышение F1-меры (Δ F1)")
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.45", edgecolor="black", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      linewidth=0.8, hatch="////"),
    ]
    ax.legend(
        handles,
        ["ДИ 95 % не накрывает ноль", "ДИ 95 % накрывает ноль (неотличим от нуля)"],
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
        frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=1.4,
    )

    n_ident = int(data["n_identities"])
    unit = data["unit_of_resampling"].split(":")[0].strip()
    n_boot = int(data["passport"]["n_bootstrap"])
    fig.suptitle(
        "Рисунок 3.5 — Вклады компонентов утечки в завышение F1-меры\n"
        f"(бутстрэп {n_boot} повторов, единица ресэмплинга — {unit}, "
        f"идентичностей {n_ident})",
        fontsize=9,
        y=1.045,
    )
    fig.tight_layout()
    return _save(fig, "fig-3-5-vklady")


# --------------------------------------------------------------------------- #
# Рисунки 3.3-3.5 «Оператор идентификации»
# --------------------------------------------------------------------------- #

def _load_json_root(name: str) -> dict:
    """Прочитать артефакт ранней редакции стенда (каталог _июль_2026)."""
    path = os.path.join(PROJECT_ROOT, "_июль_2026", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Артефакт не найден: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ru(value: float, nd: int = 4, sign: bool = False) -> str:
    """Число в русской типографике: десятичная запятая."""
    fmt = f"{{:{'+' if sign else ''}.{nd}f}}"
    return fmt.format(float(value)).replace(".", ",")


def _held_label(held: list, short: bool = False) -> str:
    """Метка фолда: отложенные идентичности без префикса ID_, по одной в строку."""
    names = [str(h).replace("ID_", "") for h in held]
    if short:
        names = [n.replace("chb01+chb21", "chb01+21") for n in names]
    return "\n".join(names)


def make_fig_3_3() -> list:
    """Сравнение двух схем оценки оператора идентификации по ROC-AUC.

    Источники:
      * «измерено» — outputs/track_b/operator_subject_cv.json,
        aggregate.roc_auc_identity (mean, std): кросс-валидация по невиданным
        идентичностям;
      * схема разнесения файлов одного субъекта — glava3_results_summary.json,
        res_001_operator.roc_auc_operator (file-holdout: те же субъекты
        присутствуют в обучении). В operator_subject_cv.json этого значения нет —
        артефакт содержит только схему с вынесением целых идентичностей, поэтому
        значение читается из сводки ранней редакции стенда.
    """
    _apply_rc()
    cv = _load_json("operator_subject_cv.json")
    july = _load_json_root("glava3_results_summary.json")

    agg = cv["aggregate"]["roc_auc_identity"]
    cv_mean = float(agg["mean"])
    cv_std = float(agg["std"])
    n_folds = len(cv["folds"])

    july_auc = float(july["res_001_operator"]["roc_auc_operator"])
    july_eer = float(july["res_001_operator"]["eer_operator"])
    cv_eer = float(cv["aggregate"]["eer_identity"]["mean"])
    cv_eer_std = float(cv["aggregate"]["eer_identity"]["std"])

    labels = [
        "file-holdout,\nте же субъекты в обучении\n(схема с разнесением файлов)",
        "кросс-валидация по\nневиданным идентичностям\n(измерено)",
    ]
    values = [july_auc, cv_mean]
    errs = [None, cv_std]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = [0, 1]
    bars = ax.bar(
        xs, values, width=0.52, edgecolor="black", linewidth=0.9, zorder=3,
    )
    bars[0].set_facecolor("white")
    bars[0].set_hatch("////")
    bars[1].set_facecolor("0.45")

    ax.errorbar(
        xs[1], values[1], yerr=errs[1], fmt="none", ecolor="black",
        elinewidth=1.0, capsize=4, capthick=1.0, zorder=4,
    )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=2)
    ax.text(
        1.46, 0.5, "случайное\nугадывание 0,5", fontsize=7.5,
        va="center", ha="left",
    )

    ax.text(
        xs[0], july_auc + 0.02, _ru(july_auc), ha="center", va="bottom",
        fontsize=8.5, zorder=5,
    )
    ax.text(
        xs[1], cv_mean + cv_std + 0.02, f"{_ru(cv_mean)} ± {_ru(cv_std)}",
        ha="center", va="bottom", fontsize=8.5, zorder=5,
    )
    ax.text(
        xs[0], 0.04, f"EER = {_ru(july_eer)}", ha="center", va="bottom",
        fontsize=7.5, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )
    ax.text(
        xs[1], 0.04, f"EER = {_ru(cv_eer)} ± {_ru(cv_eer_std)}", ha="center",
        va="bottom", fontsize=7.5, color="white", zorder=5,
    )

    # Стрелка разрыва между схемами.
    gap = july_auc - cv_mean
    x_arrow = 0.5
    ax.annotate(
        "", xy=(x_arrow, july_auc), xytext=(x_arrow, cv_mean),
        arrowprops=dict(arrowstyle="<->", linewidth=1.0, color="black",
                        shrinkA=0, shrinkB=0),
        zorder=6,
    )
    ax.plot([xs[0], x_arrow], [july_auc, july_auc], linestyle=":",
            linewidth=0.7, color="black", zorder=2)
    ax.plot([x_arrow, xs[1]], [cv_mean, cv_mean], linestyle=":",
            linewidth=0.7, color="black", zorder=2)
    ax.text(
        x_arrow + 0.04, (july_auc + cv_mean) / 2.0,
        f"разрыв\n{_ru(gap, sign=True)}", fontsize=8, va="center", ha="left",
        zorder=6,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.55, 1.85)
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("ROC-AUC оператора идентификации")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        "Рисунок 3.1 — Сравнение схем оценки оператора идентификации\n"
        f"(усы — стандартное отклонение по {n_folds} фолдам кросс-валидации; "
        "для file-holdout разброс не сообщался)",
        fontsize=9,
        y=1.055,
    )
    fig.tight_layout()
    return _save(fig, "fig-3-1-shemy-ocenki")


def make_fig_3_4() -> list:
    """Разброс ROC-AUC оператора по всем фолдам кросс-валидации.

    Источник: outputs/track_b/operator_subject_cv.json,
    folds[].roc_auc_identity и folds[].held; среднее — aggregate.roc_auc_identity.
    """
    _apply_rc()
    cv = _load_json("operator_subject_cv.json")
    folds = cv["folds"]

    values = [float(f["roc_auc_identity"]) for f in folds]
    labels = [_held_label(f["held"], short=True) for f in folds]
    mean = float(cv["aggregate"]["roc_auc_identity"]["mean"])
    std = float(cv["aggregate"]["roc_auc_identity"]["std"])
    xs = list(range(len(folds)))

    fig, ax = plt.subplots(figsize=(6.9, 3.7))

    bars = ax.bar(
        xs, values, width=0.6, facecolor="0.82", edgecolor="black",
        linewidth=0.8, zorder=3,
    )
    # Фолды, где отложена пара-дубль chb01+chb21, помечены штриховкой.
    for bar, fold in zip(bars, folds):
        if any("chb01+chb21" in str(h) for h in fold["held"]):
            bar.set_facecolor("white")
            bar.set_hatch("////")

    ax.plot(xs, values, linestyle="none", marker="o", markersize=4.5,
            markerfacecolor="black", markeredgecolor="black", zorder=5)

    line_mean = ax.axhline(mean, color="black", linestyle="-", linewidth=1.0, zorder=4)
    line_rand = ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, zorder=4)

    for x, v in zip(xs, values):
        ax.text(x, v + 0.018, _ru(v, 3), ha="center", va="bottom",
                fontsize=7.5, zorder=6)

    n_low = sum(1 for v in values if v < 0.60)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_xlim(-0.7, len(folds) - 0.3)
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("ROC-AUC оператора идентификации")
    ax.set_xlabel("Фолд: отложенные (невиданные при обучении) идентичности")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      linewidth=0.8, hatch="////"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.82", edgecolor="black",
                      linewidth=0.8),
    ]
    ax.legend(
        handles + [line_mean, line_rand],
        [
            "отложена пара-дубль chb01+chb21 (в подписях chb01+21)",
            "остальные фолды",
            f"среднее по фолдам {_ru(mean)} (СКО {_ru(std)})",
            "уровень случайного угадывания 0,5",
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
        frameon=False, fontsize=7.5, handlelength=2.0, columnspacing=1.4,
    )

    fig.suptitle(
        "Рисунок 3.2 — Разброс качества оператора по фолдам "
        "кросс-валидации по идентичностям\n"
        f"({len(folds)} фолдов; в {n_low} фолдах ROC-AUC ниже 0,60, "
        "то есть у уровня случайного угадывания)",
        fontsize=9,
        y=1.055,
    )
    fig.tight_layout()
    return _save(fig, "fig-3-2-operator-foldy")


def make_fig_3_5() -> list:
    """Средние расстояния трёх типов пар в фолдах с отложенной парой-дублём.

    Источник: outputs/track_b/operator_subject_cv.json,
    folds[].duplicate_probe: dist_within_subject_mean, dist_chb01_chb21_mean,
    dist_different_identity_mean (и auc_chb01chb21_vs_different для подписи).
    """
    _apply_rc()
    # Проверка дубля проводилась только в первом прогоне оператора;
    # в operator_v2 её нет, поэтому рисунок строится по нему и помечен.
    cv = _load_json_raw("operator_subject_cv.json")
    folds = [f for f in cv["folds"] if "duplicate_probe" in f]
    if not folds:
        raise ValueError("В operator_subject_cv.json нет фолдов с duplicate_probe")

    series = [
        ("dist_within_subject_mean", "внутри одного субъекта", "0.80", ""),
        ("dist_chb01_chb21_mean", "пара chb01-chb21 (скрытый дубль)", "white", "////"),
        ("dist_different_identity_mean", "разные пациенты", "0.40", ""),
    ]

    n = len(folds)
    xs = list(range(n))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    for j, (key, label, fc, hatch) in enumerate(series):
        vals = [float(f["duplicate_probe"][key]) for f in folds]
        offs = [x + (j - 1) * width for x in xs]
        ax.bar(
            offs, vals, width=width, label=label, facecolor=fc,
            edgecolor="black", linewidth=0.8, hatch=hatch, zorder=3,
        )
        for x, v in zip(offs, vals):
            ax.text(x, v + 0.006, _ru(v, 3), ha="center", va="bottom",
                    fontsize=6.8, rotation=90, zorder=5)

    # Подпись фолда: отложенные идентичности + AUC «дубль против разных пациентов».
    labels = [
        _held_label(f["held"])
        + "\nAUC дубль/разные = "
        + _ru(f["duplicate_probe"]["auc_chb01chb21_vs_different"], 3)
        for f in folds
    ]

    n_dup_gt_within = sum(
        1 for f in folds
        if float(f["duplicate_probe"]["dist_chb01_chb21_mean"])
        > float(f["duplicate_probe"]["dist_within_subject_mean"])
    )
    n_dup_lt_diff = sum(
        1 for f in folds
        if float(f["duplicate_probe"]["dist_chb01_chb21_mean"])
        < float(f["duplicate_probe"]["dist_different_identity_mean"])
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Фолд: отложенные идентичности")
    # Расстояние — евклидова норма разности эмбеддингов (run_operator_cv.py, np.linalg.norm).
    ax.set_ylabel("Среднее евклидово расстояние\nмежду эмбеддингами окон")
    top = max(
        float(f["duplicate_probe"][k])
        for f in folds for k, _, _, _ in series
    )
    ax.set_ylim(0.0, top * 1.22)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
        frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=1.2,
    )

    ax.set_title(
        f"Вывод: пара-дубль дальше внутрисубъектных пар во всех {n_dup_gt_within} "
        f"из {n} фолдов;\nближе пар разных пациентов лишь в {n_dup_lt_diff} из {n} — "
        "оператор скрытый дубль не распознаёт",
        fontsize=8,
        pad=5,
    )
    fig.suptitle(
        f"[{SLICE_NOTE_OLD}]\n"
        "Рисунок 3.3 — Проверка скрытого дубля chb01/chb21 по расстояниям "
        "в пространстве оператора",
        fontsize=9,
        y=1.045,
    )
    fig.tight_layout()
    return _save(fig, "fig-3-3-dubl-rasstoyaniya")


# --------------------------------------------------------------------------- #
# Рисунок 3.6 «Ablation компонентов честного протокола»
# --------------------------------------------------------------------------- #

def make_fig_3_6() -> list:
    """Вклад каждого компонента честного протокола (ablation).

    Источник: outputs/track_b/ablation_components.json,
    component_contribution.f1[вариант].{mean, per_fold, n_folds}.

    Величина = насколько ЗАВЫШАЕТСЯ F1, если компонент отключить
    (в артефакте: F1 варианта минус F1 полного P2 на том же eval).
    Компонент «временное эмбарго» бара не имеет: в P2 обучающие и оценочные
    пациенты не пересекаются, вариант P2_с_эмбарго тождественен P2
    (removed_windows = 0), поэтому вклад не определён.
    """
    _apply_rc()
    # Читается ИСПРАВЛЕННЫЙ артефакт: из исходного снят вариант «без удаления дублей».
    # На общем оценочном множестве он неизмерим — оценка фолда содержит окна обеих
    # записей пары, поэтому партнёр по дублю присутствует в ней своими же файлами.
    # Вклад скрытого дубля измеряется отдельно (hidden_duplicate.json).
    abl = _load_json("ablation_components_corrected.json")  # версии на 23 идентичностях нет
    contrib = abl["component_contribution"]["f1"]
    folds = list(abl["per_fold"].keys())

    # Контроль целостности: per_fold из component_contribution должен совпадать
    # с разностью F1 (вариант минус полный_P2), пересчитанной из per_fold.
    for variant, rec in contrib.items():
        recomputed = []
        for fold in folds:
            variants = abl["per_fold"][fold]["variants"]
            if variant not in variants or "f1" not in variants[variant]:
                continue
            recomputed.append(
                float(variants[variant]["f1"]) - float(variants["полный_P2"]["f1"])
            )
        stored = [float(v) for v in rec["per_fold"]]
        if len(recomputed) != len(stored):
            raise ValueError(
                f"{variant}: пофолдовых значений {len(stored)}, пересчитано {len(recomputed)}"
            )
        for a, b in zip(stored, recomputed):
            if abs(a - b) > 1e-9:
                raise ValueError(f"{variant}: расхождение {a} и {b}")

    # (ключ в артефакте, подпись компонента, заливка, штриховка)
    rows = [
        ("без_субъектного_разбиения", "субъектное разбиение\n(разделение по пациентам)",
         "white", "////"),
        ("без_изоляции_предобработки", "изоляция предобработки\n(статистики только по train)",
         "0.85", "...."),
    ]

    fig, ax = plt.subplots(figsize=(6.9, 3.9))

    n_rows = len(rows) + 1  # плюс строка для неприменимого эмбарго
    ys = list(range(n_rows - 1, -1, -1))  # сверху вниз
    labels = []
    all_x = [0.0]

    for i, (key, label, fc, hatch) in enumerate(rows):
        rec = contrib[key]
        mean = float(rec["mean"])
        per_fold = [float(v) for v in rec["per_fold"]]
        n_folds = int(rec["n_folds"])
        y = ys[i]
        ax.barh(
            y, mean, height=0.52, facecolor=fc, edgecolor="black",
            linewidth=0.8, hatch=hatch, zorder=3,
        )
        # Пофолдовые значения точками поверх бара.
        ax.plot(
            per_fold, [y] * len(per_fold), linestyle="none", marker="o",
            markersize=4.0, markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.8, zorder=5,
        )
        all_x.extend(per_fold + [mean])
        # Числовая подпись среднего.
        off = 0.012 if mean >= 0 else -0.012
        ax.text(
            mean + off, y + 0.30, _ru(mean, 4, sign=True),
            ha="left" if mean >= 0 else "right", va="bottom",
            fontsize=8, fontweight="bold", zorder=6,
        )
        labels.append(
            label + ("\nфолдов: %d" % n_folds if n_folds > 1
                     else "\nфолдов: 1 (только пара-дубль)")
        )

    # Временное эмбарго — отдельная отметка, бара нет.
    y_emb = ys[-1]
    emb_notes = [
        v["variants"]["P2_с_эмбарго"]
        for v in abl["per_fold"].values()
        if "P2_с_эмбарго" in v["variants"]
    ]
    removed = sorted({int(e.get("removed_windows", -1)) for e in emb_notes})
    n_identical = sum(1 for e in emb_notes if e.get("identical_to_P2") is True)
    ax.plot(
        [0.0], [y_emb], linestyle="none", marker="x", markersize=8,
        markeredgecolor="black", markeredgewidth=1.4, zorder=6,
    )
    ax.text(
        0.018, y_emb, "неприменимо в P2 по построению\n"
        f"(удалено окон: {removed[0] if len(removed) == 1 else removed}; "
        f"вариант тождественен P2 во всех {n_identical} фолдах)",
        ha="left", va="center", fontsize=7.5, zorder=6,
    )
    labels.append("временное эмбарго\n(разрыв между train и eval)")

    ax.axvline(0.0, color="black", linewidth=0.9, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.6, n_rows - 0.4)
    lo, hi = min(all_x), max(all_x)
    span = hi - lo
    ax.set_xlim(lo - 0.10 * span, hi + 0.16 * span)
    ax.set_xlabel("Завышение F1 при отключении компонента\n"
                  "(F1 варианта минус F1 полного протокола P2 на том же наборе оценки)")
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor="black",
                      linewidth=0.8, hatch=hatch)
        for _, _, fc, hatch in rows
    ]
    dot = plt.Line2D([], [], linestyle="none", marker="o", markersize=4.0,
                     markerfacecolor="white", markeredgecolor="black",
                     markeredgewidth=0.8)
    cross = plt.Line2D([], [], linestyle="none", marker="x", markersize=8,
                       markeredgecolor="black", markeredgewidth=1.4)
    ax.legend(
        handles + [dot, cross],
        [r[1].split("\n")[0] for r in rows]
        + ["пофолдовые значения", "компонент неприменим"],
        loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
        frameon=False, fontsize=7.5, handlelength=1.8, columnspacing=1.2,
    )

    subj = float(contrib["без_субъектного_разбиения"]["mean"])
    prep = float(contrib["без_изоляции_предобработки"]["mean"])
    ax.set_title(
        f"Вывод: субъектное разбиение даёт {_ru(subj, 4, sign=True)} F1; "
        f"изоляция предобработки {_ru(prep, 4, sign=True)} —\n"
        "в пределах пофолдового шума. Вклад скрытого дубля на общем оценочном "
        "множестве неизмерим и вынесен отдельно",
        fontsize=8, pad=5,
    )
    fig.suptitle(
        f"[{SLICE_NOTE_OLD}]\n"
        "Рисунок 3.7 — Ablation компонентов честного протокола: "
        "вклад каждого компонента в F1",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    return _save(fig, "fig-3-7-ablation")


# --------------------------------------------------------------------------- #
# Рисунок 3.7 «Матрицы ошибок»
# --------------------------------------------------------------------------- #

def make_fig_3_7() -> list:
    """Матрицы ошибок 2x2 для P0 и P2 по объединённым данным.

    Источник: outputs/track_b/decomposition_v2.json,
    summary.pooled.metrics.{P0,P2}.confusion (TN, FP, FN, TP).
    Заливка — по логарифму значения клетки (общая шкала для обеих матриц).
    """
    import matplotlib.colors as mcolors

    _apply_rc()
    dec = _load_json("decomposition_v2.json")
    pooled = dec["summary"]["pooled"]["metrics"]

    panels = [
        ("P0", "P0 — наивный протокол\n(случайное разбиение окон)"),
        ("P2", "P2 — честный протокол\n(разбиение по пациентам, дубли удалены)"),
    ]

    mats, infos = [], []
    for key, _ in panels:
        c = pooled[key]["confusion"]
        # Строки — истина (нет приступа / приступ), столбцы — предсказание.
        mats.append([[int(c["TN"]), int(c["FP"])], [int(c["FN"]), int(c["TP"])]])
        infos.append(pooled[key])

    flat = [v for m in mats for row in m for v in row]
    if min(flat) <= 0:
        raise ValueError("Нулевая клетка: логарифмическая заливка неприменима")
    norm = mcolors.LogNorm(vmin=min(flat), vmax=max(flat))
    cmap = plt.get_cmap("Greys")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.5))
    cell_names = [["ИО (TN)", "ЛП (FP)"], ["ЛО (FN)", "ИП (TP)"]]

    for ax, (key, title), mat, info in zip(axes, panels, mats, infos):
        im = ax.imshow(mat, cmap=cmap, norm=norm)
        for i in range(2):
            for j in range(2):
                v = mat[i][j]
                dark = norm(v) > 0.55
                ax.text(j, i - 0.10, f"{v}", ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if dark else "black")
                ax.text(j, i + 0.26, cell_names[i][j], ha="center", va="center",
                        fontsize=7.5, color="white" if dark else "black")
                # Контур клетки — различимость в Ч/Б печати.
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="black", linewidth=0.8))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["нет приступа", "приступ"], fontsize=8)
        ax.set_yticklabels(["нет приступа", "приступ"], fontsize=8, rotation=90,
                           va="center")
        ax.set_xlabel("предсказание")
        ax.set_ylabel("истина")
        ax.tick_params(length=0)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            title + "\n"
            f"F1 = {_ru(info['f1'])}, AUPRC = {_ru(info['auprc'])}\n"
            f"ROC-AUC = {_ru(info['roc_auc'])}",
            fontsize=7.8, pad=6,
        )

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("число окон (логарифмическая шкала заливки)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    fp0 = mats[0][0][1]
    fp2 = mats[1][0][1]
    tp0 = mats[0][1][1]
    tp2 = mats[1][1][1]
    ns = {(int(i["n"]), int(i["n_pos"])) for i in infos}
    if len(ns) != 1:
        raise ValueError(f"Наборы оценки P0 и P2 различаются: {ns}")
    n_all, n_pos = ns.pop()

    fig.suptitle(
        "Рисунок 3.6 — Матрицы ошибок по объединённым данным: "
        "наивный протокол P0 и честный протокол P2\n"
        f"Один и тот же набор оценки: окон {n_all}, "
        f"из них приступных {n_pos}\n"
        f"Переход P0 → P2: ложных срабатываний {fp0} → {fp2} "
        f"(в {_ru(fp2 / fp0, 1)} раза больше), "
        f"верных обнаружений {tp0} → {tp2}\n"
        "ИО — истинно отрицательные, ЛП — ложноположительные, "
        "ЛО — ложноотрицательные, ИП — истинно положительные",
        fontsize=8.2, y=1.16,
    )
    return _save(fig, "fig-3-6-matricy-oshibok")


# --------------------------------------------------------------------------- #
# Рисунок 3.8 «Важность частотных полос тремя способами»
# --------------------------------------------------------------------------- #

def make_fig_3_8() -> list:
    """Вклад пяти частотных полос по трём метрикам.

    Источник: outputs/track_b/band_importance_v2.json,
    mean_band_contribution.{f1_at_0.5, auprc, roc_auc} и anomaly_check.

    Вклад полосы = метрика полной модели минус метрика без полосы
    (усреднение по фолдам). Отрицательный вклад = удаление полосы
    «улучшает» метрику.
    """
    _apply_rc()
    band = _load_json("band_importance_v2.json")  # версии на 23 идентичностях нет
    mbc = band["mean_band_contribution"]
    anomaly = band["anomaly_check"]

    bands = [
        ("delta", "дельта"),
        ("theta", "тета"),
        ("alpha", "альфа"),
        ("beta", "бета"),
        ("gamma", "гамма"),
    ]
    # Четыре способа измерения. F1 с КАЛИБРОВАННЫМ порогом — обязательный контроль:
    # сравнение только «F1@0,5 против AUPRC» меняет одновременно порог и семейство
    # метрики, поэтому не позволяет приписать эффект порогу.
    metrics = [
        ("f1_at_0.5", "F1 при пороге 0,5", "white", "////"),
        ("f1_calibrated", "F1 при калиброванном пороге", "0.35", "xxxx"),
        ("auprc", "AUPRC", "0.55", "\\\\\\\\"),
        ("roc_auc", "ROC-AUC", "0.85", "...."),
    ]

    xs = list(range(len(bands)))
    width = 0.20

    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    for j, (mkey, mlabel, fc, hatch) in enumerate(metrics):
        vals = [float(mbc[mkey][bkey]) for bkey, _ in bands]
        offs = [x + (j - (len(metrics) - 1) / 2) * width for x in xs]
        ax.bar(offs, vals, width=width, label=mlabel, facecolor=fc,
               edgecolor="black", linewidth=0.8, hatch=hatch, zorder=3)
        for x, v in zip(offs, vals):
            ax.text(x, v + (0.004 if v >= 0 else -0.004), _ru(v, 4, sign=True),
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.6, rotation=90, zorder=5)

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in bands])
    ax.set_xlabel("Удалённая частотная полоса")
    ax.set_ylabel("Вклад полосы: метрика полной модели\nминус метрика модели без полосы")

    allv = [float(mbc[m][b]) for m, _, _, _ in metrics for b, _ in bands]
    lo, hi = min(allv), max(allv)
    span = hi - lo
    ax.set_ylim(lo - 0.55 * span, hi + 0.30 * span)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Врезка: ROC-AUC в собственном масштабе (иначе вклады не видны рядом с F1).
    roc = [float(mbc["roc_auc"][b]) for b, _ in bands]
    axi = ax.inset_axes([0.50, 0.115, 0.48, 0.285])
    axi.bar(xs, roc, width=0.6, facecolor="0.85", edgecolor="black",
            linewidth=0.7, hatch="....", zorder=3)
    axi.axhline(0.0, color="black", linewidth=0.9, zorder=4)
    axi.set_xticks(xs)
    axi.set_xticklabels([lbl for _, lbl in bands], fontsize=6.5)
    axi.tick_params(axis="y", labelsize=6.5)
    axi.set_ylim(0.0, max(roc) * 1.30)
    axi.set_title("врезка: ROC-AUC в своём масштабе — все вклады > 0",
                  fontsize=6.8, pad=2)
    axi.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75", zorder=0)
    axi.set_axisbelow(True)
    for side in ("top", "right"):
        axi.spines[side].set_visible(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=False, fontsize=7.5, handlelength=1.8, columnspacing=1.4)

    # Доли случаев «удаление улучшило метрику» пересчитываются из пофолдовых данных
    # артефакта, а не берутся из готового блока anomaly_check (он не содержит
    # f1_calibrated и roc_auc). Литералов в коде нет.
    per_fold = band["per_fold"]
    n_cases = {}
    for mkey, _, _, _ in metrics:
        n_cases[mkey] = sum(
            1 for I in per_fold for bkey, _ in bands
            if float(per_fold[I]["bands"][bkey]["delta_vs_base"][mkey]) < 0
        )
    total_cases = len(per_fold) * len(bands)
    assert total_cases == int(anomaly["cases_total"]), "расхождение с anomaly_check"
    assert n_cases["f1_at_0.5"] == int(anomaly["cases_where_removal_improved_F1_at_0.5"])
    assert n_cases["auprc"] == int(anomaly["cases_where_removal_improved_AUPRC"])

    ax.set_title(
        "Вывод: аномалия «удаление полосы улучшает метрику» сохраняется и при "
        f"калиброванном пороге ({n_cases['f1_calibrated']} из {total_cases} против "
        f"{n_cases['f1_at_0.5']} из {total_cases} при пороге 0,5),\n"
        f"но исчезает при переходе к метрикам ранжирования "
        f"(AUPRC {n_cases['auprc']}, ROC-AUC {n_cases['roc_auc']} из {total_cases}). "
        "Это свойство F1 при дисбалансе 0,377 %, а не следствие выбора порога",
        fontsize=7.6, pad=5,
    )
    fig.suptitle(
        f"[{SLICE_NOTE_OLD}]\n"
        "Рисунок 3.8 — Важность частотных полос четырьмя способами "
        "(усреднение по фолдам P2)\n"
        "Случаев, где удаление полосы улучшило метрику: "
        + ", ".join(f"{lbl} — {n_cases[k]} из {total_cases}"
                    for k, lbl, _, _ in metrics),
        fontsize=8.0, y=1.06,
    )
    fig.tight_layout()
    return _save(fig, "fig-3-8-polosy")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    created = []
    created += make_fig_3_1()
    created += make_fig_3_2()
    created += make_fig_3_3()
    created += make_fig_3_4()
    created += make_fig_3_5()
    created += make_fig_3_6()
    created += make_fig_3_7()
    created += make_fig_3_8()
    for path in created:
        print(f"{os.path.getsize(path):>9d}  {path}")
