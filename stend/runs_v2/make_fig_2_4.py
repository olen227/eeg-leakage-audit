"""Рисунок 6 (fig-2-4): каскад протоколов и аддитивное разложение завышения.

Ранее схема рисовалась в draw.io вручную и содержала числа до поправки эмбарго
(Δ_эмбарго = -0,0503, Δ_пациент = +0,1338). Теперь числа читаются из артефактов:
  outputs/track_b/decomposition_24_embargofix.json   — F1 по протоколам (среднее по фолдам)
  outputs/track_b/bootstrap_contributions_24embargofix.json — вклады и 95 % ДИ
  outputs/track_b/significance_tests_24embargofix.json — перестановочный p
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

dec = json.load(open("outputs/track_b/decomposition_24_embargofix.json"))
boot = json.load(open("outputs/track_b/bootstrap_contributions_24embargofix.json"))["contributions"]["f1"]
sig = json.load(open("outputs/track_b/significance_tests_24embargofix.json"))

f1 = {p: float(np.mean(list(dec["summary"]["f1"][p]["per_fold"].values()))) for p in ["P0", "P1", "P1e", "P2"]}
# сверка: среднее по фолдам должно совпадать с полем mean
for p in f1:
    assert abs(f1[p] - dec["summary"]["f1"][p]["mean"]) < 1e-9, p

def r(x): return f"{x:+.4f}".replace(".", ",")
def r0(x): return f"{x:.4f}".replace(".", ",")
def ci(k): return f"[{r(boot[k]['lo'])}; {r(boot[k]['hi'])}]"

pv = None
s = json.dumps(sig)
for key in ("permutation", "perm"):
    pass
def find_p(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if "perm" in k.lower() and isinstance(v, dict) and "p" in v: return v["p"]
            if "perm" in k.lower() and isinstance(v, (int, float)): return v
            got = find_p(v)
            if got is not None: return got
    return None
pv = find_p(sig)
p_txt = f"p = {pv:.4f}".replace(".", ",") if pv is not None else ""

fig = plt.figure(figsize=(16.7, 5.0), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1710); ax.set_ylim(900, 230); ax.axis("off")

boxes = [
    (60,  "P₀ — оконное разбиение", f"F1 = {r0(f1['P0'])}", "почти-дубли смежных окон", "#f2f2f2"),
    (480, "P₁ — файловое разбиение", f"F1 = {r0(f1['P1'])}", "тот же пациент, окна разведены", "#f2f2f2"),
    (900, "P₁e — файловое + эмбарго 60 с", f"F1 = {r0(f1['P1e'])}", "добавлена временная изоляция", "#f2f2f2"),
    (1320, "P₂ — субъектное разбиение", f"F1 = {r0(f1['P2'])}", "честная оценка", "#d9d9d9"),
]
for x, t1, t2, t3, col in boxes:
    ax.add_patch(FancyBboxPatch((x-10, 430), 350, 140, boxstyle="round,pad=0,rounding_size=12",
                                fc=col, ec="black", lw=1.5))
    ax.text(x+165, 462, t1, ha="center", va="center", fontsize=12)
    ax.text(x+165, 505, t2, ha="center", va="center", fontsize=12)
    ax.text(x+165, 542, t3, ha="center", va="center", fontsize=12)
for x0 in (400, 820, 1240):
    ax.annotate("", xy=(x0+70, 500), xytext=(x0, 500),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=18))
labels = [
    (435, "Δ_окна = " + r(boot["delta_okna_pochti_dubli"]["point_estimate"]), ci("delta_okna_pochti_dubli")),
    (855, "Δ_эмбарго = " + r(boot["delta_embargo"]["point_estimate"]), ci("delta_embargo")),
    (1275, "Δ_пациент = " + r(boot["delta_subject"]["point_estimate"]), ci("delta_subject")),
]
for x, a, b in labels:
    ax.text(x, 330, a, ha="center", va="center", fontsize=12.5)
    ax.text(x, 370, b, ha="center", va="center", fontsize=12.5)
tot = boot["delta_total"]
ax.add_patch(Rectangle((60, 640), 1590, 80, fc="none", ec="black", lw=1.5))
ax.text(855, 680, f"Полное завышение Δ = Δ_окна + Δ_эмбарго + Δ_пациент = {r(tot['point_estimate'])}   "
        f"[{r(tot['lo'])}; {r(tot['hi'])}],   {p_txt}", ha="center", va="center", fontsize=15, fontweight="bold")
ax.add_patch(Rectangle((60, 760), 1590, 110, fc="none", ec="black", lw=1.5))
ax.text(80, 800, "Тождество точно по построению: слагаемые образуют телескопическую сумму разностей соседних протоколов каскада.",
        ha="left", va="center", fontsize=13)
ax.text(80, 835, "Внутри фолда оценочное множество ОДНО для всех протоколов — различается только состав обучающей части.",
        ha="left", va="center", fontsize=13)
for ext in ("png", "pdf"):
    fig.savefig(f"figures_final/fig-2-4.{ext}", dpi=200, facecolor="white")
print({k: round(v, 4) for k, v in f1.items()}, {k: round(boot[k]["point_estimate"], 4) for k in boot if not k.startswith("_")}, "p:", pv)
