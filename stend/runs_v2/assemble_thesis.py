"""Сборка глав и списка литературы в единый файл диссертации.

ЗАЧЕМ. Главы существуют отдельными файлами, а научному руководителю нужен один
документ, который открывается и листается подряд. Сторонних библиотек для склейки
.docx в окружении нет, поэтому содержимое переносится напрямую: элементы тела
документа копируются в приёмник вместе с изображениями, каждая глава начинается
с новой страницы.

ИЗОБРАЖЕНИЯ. При копировании абзаца с картинкой ссылка на вложение (r:embed)
указывает на идентификатор в исходном документе и в приёмнике недействительна,
поэтому вложение переносится и ссылка переписывается на новый идентификатор.

ЗАПУСК:
    python stend/runs_v2/assemble_thesis.py ВЫХОД.docx ГЛАВА1.docx ГЛАВА2.docx ...
"""
import copy, io, sys

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml.ns import qn

NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def transplant_images(el, src_doc, dst_doc):
    """Перенести вложения, на которые ссылается элемент, и обновить идентификаторы.

    Переносится не сама часть-вложение, а её содержимое: у разных глав вложения
    называются одинаково (image1.png и далее), и при переносе как есть они
    сталкиваются по имени внутри контейнера, отчего часть изображений теряется.
    Приёмник заводит вложение заново, сам присваивает незанятое имя и, если такое
    изображение уже добавлено, переиспользует его по хэшу содержимого.
    """
    for blip in el.iter(qn("a:blip")):
        rid = blip.get(f"{{{NS_R}}}embed")
        if not rid:
            continue
        part = src_doc.part.related_parts.get(rid)
        if part is None:
            continue
        new_rid, _ = dst_doc.part.get_or_add_image(io.BytesIO(part.blob))
        blip.set(f"{{{NS_R}}}embed", new_rid)



def style_map(src_doc, dst_doc):
    """Сопоставление идентификаторов стилей источника и приёмника ПО НАЗВАНИЮ.

    Word присваивает стилям произвольные идентификаторы: в одной главе заголовок
    первого уровня имеет id «1», в другой — «Heading1». При переносе абзаца ссылка
    на несуществующий в приёмнике идентификатор молча игнорируется, и заголовок
    превращается в обычный текст, пропадая из оглавления. Поэтому ссылка
    переписывается через название стиля, одинаковое в обоих документах.
    """
    dst_by_name = {s.name: s.style_id for s in dst_doc.styles if s.style_id}
    return {s.style_id: dst_by_name[s.name]
            for s in src_doc.styles
            if s.style_id and s.name in dst_by_name and dst_by_name[s.name] != s.style_id}


def remap_styles(el, smap):
    if not smap:
        return
    for ps in el.iter(qn("w:pStyle")):
        v = ps.get(qn("w:val"))
        if v in smap:
            ps.set(qn("w:val"), smap[v])


def append_document(dst, src_path, page_break=True):
    src = Document(src_path)
    smap = style_map(src, dst)
    if page_break:
        dst.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    n = 0
    for child in src.element.body:
        if child.tag == qn("w:sectPr"):          # параметры раздела не переносим
            continue
        el = copy.deepcopy(child)
        remap_styles(el, smap)
        transplant_images(el, src, dst)
        dst.element.body.append(el)
        n += 1
    return n


if __name__ == "__main__":
    out, parts = sys.argv[1], sys.argv[2:]
    # Приёмником служит ПЕРВЫЙ документ, а не пустой шаблон: стили заголовков
    # и основного текста описаны в самих главах, и при сборке в пустой документ
    # ссылки на них не разрешаются — заголовки теряют оформление и не попадают
    # в оглавление.
    dst = Document(parts[0])
    total = len(dst.element.body)
    print(f"  основа: {parts[0]}, элементов {total}")
    for p in parts[1:]:
        n = append_document(dst, p, page_break=True)
        print(f"  + {p}: элементов {n}")
        total += n
    dst.save(out)
    import os
    print(f"{out}: элементов {total}, {os.path.getsize(out)/1024/1024:.1f} МБ")
