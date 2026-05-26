from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}
W_VAL = f"{{{WORD_NS}}}val"


@dataclass
class DocxContext:
    path: Path
    text: str
    xml: str


def read_docx_context(path: Path) -> DocxContext:
    document_root = etree.fromstring(read_document_xml(path))
    root = etree.Element("docx_context", path=str(path))
    paragraphs_root = etree.SubElement(root, "paragraphs")
    tables_root = etree.SubElement(root, "tables")
    candidates_root = etree.SubElement(root, "editable_candidates")
    lines = ["[paragraphs]"]

    paragraph_index = 0
    for paragraph in document_root.xpath("./w:body/w:p", namespaces=NS):
        text = normalize_text(text_of(paragraph))
        if not text:
            continue
        style = paragraph_style(paragraph)
        etree.SubElement(paragraphs_root, "paragraph", index=str(paragraph_index), style=style).text = text
        lines.append(f"paragraph[{paragraph_index}] style={style}: {text}")
        paragraph_index += 1

    lines.append("[tables]")
    for table_index, table in enumerate(document_root.xpath("./w:body/w:tbl", namespaces=NS)):
        table_node = etree.SubElement(tables_root, "table", index=str(table_index))
        lines.append(f"table[{table_index}]")
        for row_index, row in enumerate(table.findall("./w:tr", NS)):
            row_node = etree.SubElement(table_node, "row", index=str(row_index))
            row_cells = []
            for cell_index, cell in enumerate(row.findall("./w:tc", NS)):
                value = normalize_text(text_of(cell))
                attrs = {
                    "index": str(cell_index),
                    "target": f"table[{table_index}] row[{row_index}] cell[{cell_index}]",
                    "chars": str(len(value)),
                }
                grid_span = grid_span_of(cell)
                vmerge = vertical_merge_of(cell)
                if grid_span:
                    attrs["gridSpan"] = grid_span
                if vmerge:
                    attrs["vMerge"] = vmerge
                etree.SubElement(row_node, "cell", **attrs).text = value
                row_cells.append((cell_index, value, grid_span, vmerge))
            if row_cells:
                lines.append(
                    f"  row[{row_index}]: "
                    + " | ".join(format_cell(cell_index, value, grid_span, vmerge) for cell_index, value, grid_span, vmerge in row_cells)
                )
            add_editable_candidates(candidates_root, table_index, row_index, row_cells)

    text = "\n".join(lines)
    if text.strip() == "[paragraphs]\n[tables]":
        raise ValueError("DOCX에서 텍스트를 찾지 못했습니다.")
    xml = etree.tostring(root, pretty_print=True, encoding="unicode")
    return DocxContext(path=path, text=text, xml=xml)


def read_document_xml(path: Path) -> bytes:
    with ZipFile(path) as archive:
        return archive.read("word/document.xml")


def text_of(element: etree._Element) -> str:
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


def paragraph_style(paragraph: etree._Element) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", NS)
    return style.get(W_VAL, "") if style is not None else ""


def grid_span_of(cell: etree._Element) -> str:
    grid_span = cell.find("./w:tcPr/w:gridSpan", NS)
    return grid_span.get(W_VAL, "") if grid_span is not None else ""


def vertical_merge_of(cell: etree._Element) -> str:
    vmerge = cell.find("./w:tcPr/w:vMerge", NS)
    return vmerge.get(W_VAL, "continue") if vmerge is not None else ""


def add_editable_candidates(root: etree._Element, table_index: int, row_index: int, cells: list[tuple[int, str, str, str]]) -> None:
    if len(cells) < 2:
        return
    left_label = cells[0][1]
    for cell_index, value, grid_span, _vmerge in cells[1:]:
        if not value:
            continue
        attrs = {
            "target": f"table[{table_index}] row[{row_index}] cell[{cell_index}]",
            "left_label": left_label,
            "chars": str(len(value)),
        }
        if grid_span:
            attrs["gridSpan"] = grid_span
        etree.SubElement(root, "candidate", **attrs).text = value


def format_cell(cell_index: int, value: str, grid_span: str, vmerge: str) -> str:
    meta = []
    if grid_span:
        meta.append(f"gridSpan={grid_span}")
    if vmerge:
        meta.append(f"vMerge={vmerge}")
    meta_text = f" ({', '.join(meta)})" if meta else ""
    return f"cell[{cell_index}]{meta_text}={value}"


def compact_context(text: str, limit: int = 5000) -> str:
    lines = []
    total = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        clipped = line[:700]
        lines.append(clipped)
        total += len(clipped)
        if total >= limit:
            break
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    return " ".join(text.split())
