"""Рисунок: завышение как функция ёмкости модели.

Левая панель — расхождение протоколов: наивный растёт с ёмкостью безостановочно,
честный выходит на максимум и снижается. Правая — сам разрыв с интервалами по фолдам.

Ключ к прочтению рисунка: до примерно 78 тыс. параметров прирост ёмкости достаётся
обоим протоколам поровну, то есть модель действительно учится; далее весь прирост
забирает наивный протокол, то есть ёмкость расходуется на запоминание пациента.

Запуск: python stend/runs_v2/make_fig_capacity_curve.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9})
ru = lambda x, s=False: (("+" if s and x > 0 else "") + f"{x:.4f}").replace(".", ",")


def main():
    C = json.load(open("outputs/track_b/capacity_curve.json"))
    pts = C["points"]
    n = np.array([p["n_params"] for p in pts], dtype=float)
    P0 = np.array([p["P0_mean"] for p in pts])
    P2 = np.array([p["P2_mean"] for p in pts])
    D = np.array([p["delta_mean"] for p in pts])
    SD = np.array([p["delta_std"] for p in pts])
    nf = C["passport"]["n_folds"]
    se = SD / np.sqrt(nf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6))

    # --- левая панель: расхождение протоколов ---
    ax1.plot(n, P0, "o-", color="#b5482f", lw=1.8, ms=7, label="P₀ — протокол с утечкой")
    ax1.plot(n, P2, "s-", color="#2f6b4f", lw=1.8, ms=7, label="P₂ — честный протокол")
    ax1.fill_between(n, P2, P0, color="#d9b38c", alpha=0.35, zorder=0)
    ax1.set_xscale("log")
    for xi, a, b in zip(n, P0, P2):
        ax1.annotate("", xy=(xi, a), xytext=(xi, b),
                     arrowprops=dict(arrowstyle="-", lw=0.7, color="0.5"))
    i_best = int(np.argmax(P2))
    # подпись максимума ставится НАД точкой: снизу её затирают подписи оси
    # разделитель тысяч заменяется ТОЛЬКО в числе параметров: общая замена по строке
    # затирала бы и десятичную запятую в значении F1
    _np = f"{int(n[i_best]):,}".replace(",", " ")
    ax1.annotate(f"максимум честной F1: {ru(P2[i_best])}\nпри {_np} параметрах",
                 xy=(n[i_best], P2[i_best]), xytext=(n[i_best] * 0.24, P2[i_best] + 0.075),
                 fontsize=7.5, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
                 bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.3", lw=0.6))
    ax1.text(np.sqrt(n[1] * n[-1]), (P0[-1] + P2[-1]) / 2 + 0.02, "завышение Δ",
             fontsize=8.5, ha="center", color="#8a5a2b", style="italic")
    ax1.set_xlabel("число обучаемых параметров детектора (логарифмическая шкала)")
    ax1.set_ylabel("F1-мера, среднее по фолдам")
    ax1.set_title("Наивный протокол растёт, честный — нет", pad=6)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(ls=":", lw=0.5, color="0.85"); ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # --- правая панель: сам разрыв ---
    ax2.errorbar(n, D, yerr=se, fmt="o-", color="#3a3a8c", lw=1.8, ms=7,
                 capsize=4, elinewidth=1)
    ax2.set_xscale("log")
    for xi, di in zip(n, D):
        ax2.annotate(ru(di, True), (xi, di), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8)
    sl = C["slope_per_decade"]
    ax2.set_xlabel("число обучаемых параметров детектора (логарифмическая шкала)")
    ax2.set_ylabel("разрыв Δ = F1(P₀) − F1(P₂)")
    ax2.set_title(f"Разрыв растёт монотонно: {ru(sl, True)} на порядок ёмкости", pad=6)
    ax2.grid(ls=":", lw=0.5, color="0.85"); ax2.set_axisbelow(True)
    ax2.set_ylim(0.10, max(D + se) + 0.045)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    share = round(C["share_to_leakage"] * 100)
    fig.suptitle(
        "Рисунок 17 – Завышение метрики как функция ёмкости модели\n"
        f"Четыре архитектуры, различающиеся только шириной свёрточных блоков; {nf} фолдов; "
        f"из прироста ёмкости {share} % достаётся утечке.\n"
        f"Разрыв возрастает строго монотонно: точное перестановочное p = "
        f"{ru(C['permutation_p_exact'])} при пределе {ru(C['min_achievable_p'])} для четырёх точек. "
        f"Усы — стандартная ошибка среднего по фолдам.",
        fontsize=8.5, y=1.06)
    fig.tight_layout()
    os.makedirs("figures_final", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures_final/fig-3-10-emkost.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print("figures_final/fig-3-10-emkost.png")


if __name__ == "__main__":
    main()
