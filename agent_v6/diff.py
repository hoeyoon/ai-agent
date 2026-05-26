from lxml import etree


def build_diff_context(before_xml: str, after_xml: str) -> str:
    before = etree.fromstring(before_xml.encode("utf-8"))
    after = etree.fromstring(after_xml.encode("utf-8"))
    root = etree.Element("diff_context")

    add_diff_group(root, "paragraph", "paragraphs", paragraph_map(before), paragraph_map(after))
    add_diff_group(root, "cell", "cells", cell_map(before), cell_map(after))
    add_diff_group(root, "candidate", "candidates", candidate_map(before), candidate_map(after))

    return etree.tostring(root, pretty_print=True, encoding="unicode")


def add_diff_group(
    root: etree._Element,
    item_tag: str,
    group_name: str,
    before_map: dict[str, dict[str, str]],
    after_map: dict[str, dict[str, str]],
) -> None:
    changed = etree.SubElement(root, f"changed_{group_name}")
    unchanged = etree.SubElement(root, f"unchanged_{group_name}")
    missing = etree.SubElement(root, f"missing_{group_name}")

    for target in after_map:
        if target not in before_map:
            item = etree.SubElement(changed, item_tag, attrs_for(after_map[target]))
            etree.SubElement(item, "before").text = ""
            etree.SubElement(item, "after").text = after_map[target]["value"]

    for target, before_item in before_map.items():
        after_item = after_map.get(target)
        if after_item is None:
            item = etree.SubElement(missing, item_tag, attrs_for(before_item))
            etree.SubElement(item, "before").text = before_item["value"]
            continue
        if normalize(before_item["value"]) == normalize(after_item["value"]):
            item = etree.SubElement(unchanged, item_tag, attrs_for(before_item))
            item.text = before_item["value"]
        else:
            item = etree.SubElement(changed, item_tag, attrs_for(before_item))
            etree.SubElement(item, "before").text = before_item["value"]
            etree.SubElement(item, "after").text = after_item["value"]


def paragraph_map(root: etree._Element) -> dict[str, dict[str, str]]:
    result = {}
    for paragraph in root.findall("./paragraphs/paragraph"):
        index = paragraph.get("index", "")
        if not index:
            continue
        target = f"paragraph[{index}]"
        result[target] = {
            "target": target,
            "index": index,
            "value": text_of(paragraph),
        }
    return result


def cell_map(root: etree._Element) -> dict[str, dict[str, str]]:
    result = {}
    for table in root.findall("./tables/table"):
        table_index = table.get("index", "")
        for row in table.findall("./row"):
            row_index = row.get("index", "")
            for cell in row.findall("./cell"):
                target = cell.get("target", "")
                if not target:
                    target = f"table[{table_index}] row[{row_index}] cell[{cell.get('index', '')}]"
                result[target] = {
                    "target": target,
                    "table": table_index,
                    "row": row_index,
                    "cell": cell.get("index", ""),
                    "gridSpan": cell.get("gridSpan", ""),
                    "value": text_of(cell),
                }
    return result


def candidate_map(root: etree._Element) -> dict[str, dict[str, str]]:
    result = {}
    for candidate in root.findall("./editable_candidates/candidate"):
        target = candidate.get("target", "")
        if not target:
            continue
        result[target] = {
            "target": target,
            "left_label": candidate.get("left_label", ""),
            "value": text_of(candidate),
        }
    return result


def attrs_for(item: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in item.items() if key != "value" and value}


def text_of(element: etree._Element) -> str:
    return " ".join("".join(element.itertext()).split())


def normalize(value: str) -> str:
    return "".join(value.split()).lower()

