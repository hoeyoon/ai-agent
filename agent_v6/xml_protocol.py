import re
from dataclasses import dataclass

from lxml import etree

ALLOWED_ACTIONS = {"fill_template", "replace_table_value", "replace_paragraph", "replace_text"}
PLACEHOLDERS = {"...", "…", "새내용", "수정된내용", "확장된새내용", "보강된내용", "내용", "value", "newvalue"}


class ProtocolError(RuntimeError):
    pass


@dataclass
class Action:
    tool: str
    label: str = ""
    value: str = ""
    index: int | None = None
    old: str = ""
    new: str = ""
    fields: dict[str, str] | None = None


@dataclass
class Evaluation:
    passed: bool
    reason: str
    feedback: str


@dataclass
class AnalysisResult:
    mode: str
    document_type: str
    topic: str
    summary: str
    preserve: str
    change: str
    reason: str


def extract_required_xml(text: str, root_name: str) -> str:
    text = strip_noise(text)
    pattern = rf"(?is)<{root_name}\b.*?</{root_name}>"
    match = re.search(pattern, text)
    if not match:
        raise ProtocolError(f"<{root_name}> XML을 찾지 못했습니다.")
    return match.group(0).strip()


def strip_noise(text: str) -> str:
    text = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", text)
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    return text.replace("```xml", "").replace("```XML", "").replace("```", "").strip()


def parse_strict_xml(xml_text: str) -> etree._Element:
    parser = etree.XMLParser(recover=False, remove_blank_text=False, resolve_entities=False)
    try:
        return etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ProtocolError(f"XML 문법 오류: {exc}") from exc


def parse_request_analysis(text: str) -> tuple[str, etree._Element]:
    xml_text = extract_required_xml(text, "request_analysis")
    root = parse_strict_xml(xml_text)
    if root.tag != "request_analysis":
        raise ProtocolError("루트가 <request_analysis>가 아닙니다.")
    return xml_text, root


def parse_analysis_result(text: str) -> tuple[str, AnalysisResult]:
    xml_text = extract_required_xml(text, "analysis_result")
    root = parse_strict_xml(xml_text)
    if root.tag != "analysis_result":
        raise ProtocolError("루트가 <analysis_result>가 아닙니다.")
    mode = child_text(root, "recommended_mode")
    if mode not in {"edit_existing", "create_from_source", "create_new"}:
        raise ProtocolError(f"지원하지 않는 recommended_mode입니다: {mode}")
    result = AnalysisResult(
        mode=mode,
        document_type=child_text(root, "document_type"),
        topic=child_text(root, "topic"),
        summary=child_text(root, "summary"),
        preserve=child_text(root, "preserve"),
        change=child_text(root, "change"),
        reason=child_text(root, "reason"),
    )
    require_real_text(result.reason, "reason")
    return xml_text, result


def parse_actions_output(text: str) -> tuple[str, list[Action]]:
    xml_text = extract_required_xml(text, "actions")
    root = parse_strict_xml(xml_text)
    if root.tag != "actions":
        raise ProtocolError("루트가 <actions>가 아닙니다.")

    actions = []
    for child in root:
        if child.tag not in ALLOWED_ACTIONS:
            raise ProtocolError(f"지원하지 않는 액션입니다: {child.tag}")
        if child.tag == "fill_template":
            fields = parse_template_fields(child)
            actions.append(Action(tool=child.tag, fields=fields))
        elif child.tag == "replace_table_value":
            label = child.get("target") or child.get("label") or child_text(child, "target") or child_text(child, "label")
            value = child.get("value") or child_text(child, "value") or element_text(child)
            require_real_text(label, "label")
            require_real_text(value, "value")
            actions.append(Action(tool=child.tag, label=label, value=value))
        elif child.tag == "replace_paragraph":
            index_text = child.get("index") or child_text(child, "index")
            target = child.get("target") or child_text(child, "target")
            if not index_text and target:
                match = re.fullmatch(r"paragraph\[(\d+)\]", target.strip(), flags=re.IGNORECASE)
                if match:
                    index_text = match.group(1)
            value = child.get("value") or child_text(child, "value") or element_text(child)
            if not index_text or not index_text.isdigit():
                raise ProtocolError("replace_paragraph index가 숫자가 아닙니다.")
            require_real_text(value, "value")
            actions.append(Action(tool=child.tag, index=int(index_text), value=value))
        elif child.tag == "replace_text":
            old = child.get("old") or child_text(child, "old")
            new = child.get("new") or child_text(child, "new")
            require_real_text(old, "old")
            require_real_text(new, "new")
            actions.append(Action(tool=child.tag, old=old, new=new))

    if not actions:
        raise ProtocolError("<actions> 안에 액션이 없습니다.")
    return xml_text, actions


def parse_agent_result_output(text: str) -> tuple[str, str, str, list[Action]]:
    xml_text = extract_required_xml(text, "agent_result")
    root = parse_strict_xml(xml_text)
    if root.tag != "agent_result":
        raise ProtocolError("루트가 <agent_result>가 아닙니다.")

    analysis = root.find("request_analysis")
    actions_root = root.find("actions")
    if analysis is None:
        raise ProtocolError("<request_analysis>가 없습니다.")
    if actions_root is None:
        raise ProtocolError("<actions>가 없습니다.")

    analysis_xml = etree.tostring(analysis, pretty_print=True, encoding="unicode").strip()
    actions_xml = etree.tostring(actions_root, pretty_print=True, encoding="unicode").strip()
    _actions_xml, actions = parse_actions_output(actions_xml)
    return xml_text, analysis_xml, actions_xml, actions


def parse_template_fields(root: etree._Element) -> dict[str, str]:
    fields = {}
    for field in root.findall("field"):
        target = child_text(field, "target")
        value = child_text(field, "value")
        require_real_text(target, "target")
        require_real_text(value, "value")
        fields[target] = value
    if not fields:
        raise ProtocolError("fill_template field가 없습니다.")
    return fields


def parse_document_output(text: str) -> tuple[str, etree._Element]:
    xml_text = extract_required_xml(text, "document")
    root = parse_strict_xml(xml_text)
    if root.tag != "document":
        raise ProtocolError("루트가 <document>가 아닙니다.")
    require_real_text(child_text(root, "title"), "title")
    if len(root.findall("section")) < 1:
        raise ProtocolError("section이 필요합니다.")
    return xml_text, root


def parse_evaluation_output(text: str) -> tuple[str, Evaluation]:
    xml_text = extract_required_xml(text, "evaluation")
    root = parse_strict_xml(xml_text)
    if root.tag != "evaluation":
        raise ProtocolError("루트가 <evaluation>이 아닙니다.")
    passed = (root.get("passed") or "false").strip().lower() == "true"
    reason = child_text(root, "reason")
    feedback = child_text(root, "feedback")
    require_real_text(reason, "reason")
    if not passed:
        require_real_text(feedback, "feedback")
    return xml_text, Evaluation(passed=passed, reason=reason, feedback=feedback)


def child_text(root: etree._Element, tag: str) -> str:
    child = root.find(tag)
    if child is None:
        return ""
    return " ".join("".join(child.itertext()).split())


def element_text(root: etree._Element) -> str:
    return " ".join("".join(root.itertext()).split())


def require_real_text(value: str, name: str) -> None:
    normalized = "".join(value.split()).lower()
    if not normalized:
        raise ProtocolError(f"{name} 값이 비어 있습니다.")
    if normalized in PLACEHOLDERS or set(normalized) <= {"."}:
        raise ProtocolError(f"{name} 값이 placeholder입니다: {value}")

