import re
from dataclasses import dataclass

from lxml import etree

from agent_v5.xml_protocol import ProtocolError, child_text, extract_required_xml, parse_strict_xml, strip_noise


ALLOWED_TOOLS = {
    "read_docx_context",
    "analyze_docx",
    "edit_docx",
    "create_docx",
    "evaluate_result",
    "finish",
}


@dataclass
class ToolCall:
    name: str
    reason: str = ""
    args: dict[str, str] | None = None


def parse_tool_call_output(text: str) -> tuple[str, ToolCall]:
    xml_text = extract_tool_call_xml(text)
    root = parse_strict_xml(xml_text)
    if root.tag != "tool_call":
        raise ProtocolError("루트가 <tool_call>이 아닙니다.")
    name = root.get("name") or child_text(root, "name")
    if name not in ALLOWED_TOOLS:
        raise ProtocolError(f"지원하지 않는 tool_call입니다: {name}")
    args = {}
    args_root = root.find("args")
    if args_root is not None:
        for child in args_root:
            args[child.tag] = " ".join("".join(child.itertext()).split())
    return xml_text, ToolCall(name=name, reason=child_text(root, "reason"), args=args)


def extract_tool_call_xml(text: str) -> str:
    cleaned = strip_noise(text)
    start_match = re.search(r'(?is)<tool_call\s+name\s*=', cleaned)
    if start_match:
        close_index = cleaned.find("</tool_call>", start_match.start())
        if close_index >= 0:
            close_index += len("</tool_call>")
            return cleaned[start_match.start():close_index].strip()
    return extract_required_xml(cleaned, "tool_call")


def observation_xml(tool: str, status: str, message: str = "", **fields: str) -> str:
    root = etree.Element("observation", tool=tool, status=status)
    if message:
        etree.SubElement(root, "message").text = message
    for key, value in fields.items():
        if value is None:
            continue
        safe_key = safe_tag(key)
        etree.SubElement(root, safe_key).text = str(value)
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def safe_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if not cleaned or not re.match(r"[A-Za-z_]", cleaned):
        return "field"
    return cleaned
