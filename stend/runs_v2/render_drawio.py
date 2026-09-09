"""Отрисовка схем формата .drawio (mxGraph) в PNG и PDF без установки редактора.

ЗАЧЕМ. Рисунки глав 1 и 2 существуют только как исходники .drawio, а в тексте на их
местах стоят пометки «заглушка make-figures». Приложение diagrams.net на машине не
установлено, поэтому разбор и отрисовка выполняются напрямую: читается XML mxGraph,
из него берутся прямоугольники, подписи и стрелки, и всё это выводится средствами
matplotlib в том же оформлении, что и рисунки главы 3.

ПОДДЕРЖИВАЕТСЯ то подмножество mxGraph, которым сделаны схемы работы: прямоугольники
(в том числе скруглённые и с пунктиром), текстовые блоки без рамки, эллипсы, прямые и
ортогональные стрелки между узлами, заливка и цвет обводки, кегль и выравнивание.

ЗАПУСК:
    python stend/runs_v2/render_drawio.py fig-2-1.drawio [ещё файлы ...] --out figures_out
"""
import os, re, html, textwrap, argparse
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch

DPI = 200
SCALE = 100.0          # единиц mxGraph на дюйм


def parse_style(s):
    d = {}
    for part in (s or "").split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
        else:
            d[part.strip()] = "1"
    return d


def clean(v):
    """Значение подписи -> текст: снять html-разметку, развернуть переводы строк."""
    v = html.unescape(v or "")
    v = re.sub(r"<\s*br\s*/?\s*>", "\n", v, flags=re.I)
    v = re.sub(r"<[^>]+>", "", v)
    return v.strip()


def color(v, default=None):
    if not v or v.lower() == "none":
        return default
    return v


def load(path):
    root = ET.parse(path).getroot()
    model = root.find(".//mxGraphModel")
    cells = model.find("root").findall("mxCell")
    nodes, edges = {}, []
    for c in cells:
        st = parse_style(c.get("style"))
        g = c.find("mxGeometry")
        if c.get("vertex") == "1" and g is not None:
            nodes[c.get("id")] = {
                "x": float(g.get("x", 0)), "y": float(g.get("y", 0)),
                "w": float(g.get("width", 0)), "h": float(g.get("height", 0)),
                "label": clean(c.get("value")), "style": st,
            }
        elif c.get("edge") == "1":
            edges.append({"src": c.get("source"), "dst": c.get("target"),
                          "label": clean(c.get("value")), "style": st})
    return nodes, edges


def anchor(a, b):
    """Точки выхода и входа стрелки: по ближайшим сторонам прямоугольников."""
    ax, ay = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bx, by = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    if abs(bx - ax) >= abs(by - ay):
        p = (a["x"] + a["w"], ay) if bx > ax else (a["x"], ay)
        q = (b["x"], by) if bx > ax else (b["x"] + b["w"], by)
    else:
        p = (ax, a["y"] + a["h"]) if by > ay else (ax, a["y"])
        q = (bx, b["y"]) if by > ay else (bx, b["y"] + b["h"])
    return p, q



def fit_text(fig, ax, t, w, h, min_pt=5.0):
    """Уменьшить кегль надписи, пока она не уместится в свой блок.

    Оценка ширины по числу знаков неточна для полужирного начертания и кириллицы,
    поэтому размер измеряется фактически, по отрисованной надписи.
    """
    fig.canvas.draw()
    for _ in range(24):
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        bb = bb.transformed(ax.transData.inverted())
        if (bb.width <= w - 8 and abs(bb.height) <= h - 6) or t.get_fontsize() <= min_pt:
            return
        t.set_fontsize(t.get_fontsize() * 0.92)



def trim_zones(nodes, margin=40):
    """Подрезать снизу блоки-зоны, у которых нижняя часть пуста.

    Зоны в исходных схемах заданы с запасом по высоте, и при отрисовке до 40 %
    площади рисунка оказывается пустым, что для печати неприемлемо. Содержание
    при подрезке не меняется: перемещаются только нижние границы рамок, внутри
    которых ничего не расположено.
    """
    for zid, z in nodes.items():
        inner = [m for k, m in nodes.items()
                 if k != zid and m["x"] >= z["x"] and m["y"] >= z["y"]
                 and m["x"] + m["w"] <= z["x"] + z["w"]
                 and m["y"] + m["h"] <= z["y"] + z["h"]]
        if len(inner) < 2:
            continue
        bottom = max(m["y"] + m["h"] for m in inner) + margin
        if bottom < z["y"] + z["h"]:
            z["h"] = bottom - z["y"]
    return nodes



def collapse_gaps(nodes, keep=60):
    """Схлопнуть пустые горизонтальные полосы, оставив зазор keep.

    После подрезки зон между ними и нижними подписями остаётся провал в несколько
    сотен единиц. Полосы, где нет ни одного блока, сжимаются, а всё, что лежит ниже
    полосы, поднимается на ту же величину — взаимное расположение блоков сохраняется.
    """
    if not nodes:
        return nodes
    spans = sorted((n["y"], n["y"] + n["h"]) for n in nodes.values())
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    shift = 0.0
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        gap = next_start - prev_end
        if gap > keep:
            cut = gap - keep
            for n in nodes.values():
                if n["y"] >= next_start - shift:
                    n["y"] -= cut
            shift += cut
    return nodes


def render(path, outdir):
    nodes, edges = load(path)
    trim_zones(nodes)
    collapse_gaps(nodes)
    if not nodes:
        raise SystemExit(f"{path}: узлов не найдено")
    xs = [n["x"] for n in nodes.values()] + [n["x"] + n["w"] for n in nodes.values()]
    ys = [n["y"] for n in nodes.values()] + [n["y"] + n["h"] for n in nodes.values()]
    pad = 30
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    fig, ax = plt.subplots(figsize=((x1 - x0) / SCALE, (y1 - y0) / SCALE))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)                       # ось Y вниз, как в mxGraph
    ax.set_aspect("equal")                    # без этого схема растягивается по одной оси
    ax.axis("off")

    for n in nodes.values():
        st = n["style"]
        fc = color(st.get("fillColor"), "none")
        ec = color(st.get("strokeColor"), "#000000")
        is_text = "text" in st or (fc == "none" and st.get("strokeColor", "").lower() == "none")
        if not is_text:
            dash = (0, (4, 3)) if st.get("dashed") == "1" else "solid"
            if "ellipse" in st:
                ax.add_patch(Ellipse((n["x"] + n["w"] / 2, n["y"] + n["h"] / 2),
                                     n["w"], n["h"], facecolor=fc, edgecolor=ec,
                                     linewidth=1.2, linestyle=dash, zorder=2))
            elif st.get("rounded") == "1":
                ax.add_patch(FancyBboxPatch((n["x"] + 8, n["y"] + 8),
                                            n["w"] - 16, n["h"] - 16,
                                            boxstyle="round,pad=8,rounding_size=12",
                                            facecolor=fc, edgecolor=ec,
                                            linewidth=1.2, linestyle=dash, zorder=2))
            else:
                ax.add_patch(Rectangle((n["x"], n["y"]), n["w"], n["h"],
                                       facecolor=fc, edgecolor=ec,
                                       linewidth=1.2, linestyle=dash, zorder=2))
        if n["label"]:
            # mxGraph задаёт кегль в тех же единицах, что и геометрию: 100 ед. на дюйм,
            # значит в пунктах это fontSize * 72/100.
            fs = float(st.get("fontSize", 22)) * 0.72
            ha = {"left": "left", "right": "right"}.get(st.get("align"), "center")
            va = {"top": "top", "bottom": "bottom"}.get(st.get("verticalAlign"), "center")
            tx = {"left": n["x"] + 10, "right": n["x"] + n["w"] - 10}.get(
                ha, n["x"] + n["w"] / 2)
            ty = {"top": n["y"] + 8, "bottom": n["y"] + n["h"] - 8}.get(
                va, n["y"] + n["h"] / 2)
            # перенос по ширине блока: ширина в пунктах / средняя ширина знака
            ncols = max(8, int((n["w"] * 0.72 - 12) / (fs * 0.5)))
            label = "\n".join("\n".join(textwrap.wrap(ln, ncols)) if ln.strip() else ln
                             for ln in n["label"].split("\n"))
            t = ax.text(tx, ty, label, ha=ha, va=va, fontsize=fs, zorder=4,
                        color=color(st.get("fontColor"), "#000000"),
                        fontweight="bold" if st.get("fontStyle") in ("1", "3") else "normal",
                        linespacing=1.35)
            fit_text(fig, ax, t, n["w"], n["h"])

    for e in edges:
        a, b = nodes.get(e["src"]), nodes.get(e["dst"])
        if not a or not b:
            continue
        p, q = anchor(a, b)
        st = e["style"]
        ax.add_patch(FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=13, zorder=3,
            linewidth=1.1, color=color(st.get("strokeColor"), "#000000"),
            linestyle=(0, (4, 3)) if st.get("dashed") == "1" else "solid",
            connectionstyle="arc3,rad=0.0", shrinkA=2, shrinkB=4))
        if e["label"]:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 12, e["label"],
                    ha="center", va="bottom", fontsize=8, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return f"{base}.png", len(nodes), len(edges)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="figures_out")
    a = ap.parse_args()
    for f in a.files:
        p, nn, ne = render(f, a.out)
        print(f"{p}: узлов {nn}, стрелок {ne}")
