"""Подстановка иллюстраций в .docx на места пометок «[РИСУНОК N.M — …]».

ЗАЧЕМ. В тексте глав рисунки обозначены пометками в квадратных скобках, а сами
изображения лежат отдельно в figures_out/. Скрипт заменяет каждую пометку на само
изображение с подписью под ним, как требует оформление: рисунок по центру, подпись
по центру, кегль 12, вида «Рисунок N.M — Название».

СООТВЕТСТВИЕ ФАЙЛОВ. Номер берётся из пометки: «[РИСУНОК 3.2 — …]» → fig-3-2-*.png.
Если файла нет, пометка остаётся нетронутой и попадает в отчёт как пропуск — молча
исчезнуть рисунок не может.

ЗАПУСК:
    python stend/runs_v2/insert_figures.py ДОКУМЕНТ.docx [--figdir figures_out]
"""
import os, re, glob, argparse

from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

MARK = re.compile(r"^\s*\[РИСУНОК\s+(\d+\.\d+)\s*[—–-]\s*(.+?)\]\s*$", re.S)
MAX_W = Cm(16.0)          # ширина полосы набора при полях 3/1,5/2/2 см


def find_image(figdir, num):
    stem = "fig-" + num.replace(".", "-")
    hits = sorted(glob.glob(os.path.join(figdir, stem + "*.png")))
    return hits[0] if hits else None


def insert(docx_path, figdir, out_path):
    doc = Document(docx_path)
    done, missing = [], []

    for p in list(doc.paragraphs):
        m = MARK.match(p.text)
        if not m:
            continue
        num, caption = m.group(1), " ".join(m.group(2).split())
        img = find_image(figdir, num)
        if not img:
            missing.append(num)
            continue

        # абзац пометки превращается в абзац с изображением
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run()
        run.add_picture(img, width=MAX_W)

        # подпись отдельным абзацем сразу после
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.first_line_indent = Cm(0)
        cap.paragraph_format.space_after = Pt(10)
        cr = cap.add_run(f"Рисунок {num} — {caption[0].upper()}{caption[1:]}")
        cr.font.size, cr.font.name = Pt(12), "Times New Roman"
        p._element.addnext(cap._element)
        done.append(num)

    doc.save(out_path)
    return done, missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--figdir", default="figures_out")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or a.docx
    d, miss = insert(a.docx, a.figdir, out)
    print(f"{out}: вставлено {len(d)} {d}")
    if miss:
        print(f"  НЕ НАЙДЕНЫ изображения для: {miss} — пометки оставлены в тексте")
