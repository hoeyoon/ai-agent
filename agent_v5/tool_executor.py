from pathlib import Path

from lxml import etree

from agent_v5.context import DocxContext, compact_context, read_docx_context
from agent_v5.diff import build_diff_context
from agent_v5.docx_tools import apply_actions, write_document_xml
from agent_v5.evaluator import evaluate_result
from agent_v5.generator import generate_document_xml
from agent_v5.tool_protocol import ToolCall, observation_xml
from agent_v5.xml_protocol import Action, ProtocolError, parse_actions_output


class ToolState:
    def __init__(self, run_dir: Path, model: str, input_path: Path | None = None):
        self.run_dir = run_dir
        self.model = model
        self.input_path = input_path
        self.current_docx = input_path
        self.source_context: DocxContext | None = None
        self.last_context: DocxContext | None = None
        self.last_result_path: Path | None = None
        self.last_actions_xml = ""
        self.last_document_xml = ""
        self.last_evaluation_xml = ""
        self.last_attempt = 0


def execute_tool(call: ToolCall, state: ToolState, request: str) -> str:
    try:
        if call.name == "read_docx_context":
            return read_context_tool(call, state)
        if call.name == "analyze_docx":
            return analyze_docx_tool(state)
        if call.name == "edit_docx":
            return edit_docx_tool(call, state)
        if call.name == "create_docx":
            return create_docx_tool(call, state, request)
        if call.name == "evaluate_result":
            return evaluate_result_tool(state, request)
        if call.name == "finish":
            return finish_tool(call, state)
        return observation_xml(call.name, "error", f"지원하지 않는 도구입니다: {call.name}")
    except Exception as exc:
        return observation_xml(call.name, "error", str(exc))


def read_context_tool(call: ToolCall, state: ToolState) -> str:
    path_text = (call.args or {}).get("path", "").strip()
    path = Path(path_text) if path_text else state.current_docx
    if path is None:
        return observation_xml("read_docx_context", "error", "읽을 DOCX 파일이 없습니다.")
    context = read_docx_context(path)
    state.current_docx = path
    state.source_context = context
    state.last_context = context
    state.run_dir.joinpath("source_context.txt").write_text(context.text, encoding="utf-8")
    state.run_dir.joinpath("source_context.xml").write_text(context.xml, encoding="utf-8")
    return observation_xml(
        "read_docx_context",
        "ok",
        "DOCX context를 읽었습니다.",
        path=str(path),
        context_preview=compact_context(context.xml, limit=3500),
    )


def analyze_docx_tool(state: ToolState) -> str:
    context = state.last_context or state.source_context
    if context is None:
        return observation_xml("analyze_docx", "error", "먼저 read_docx_context를 실행해야 합니다.")
    return observation_xml(
        "analyze_docx",
        "ok",
        "문서 구조와 내용을 요약했습니다.",
        analysis=compact_context(context.xml, limit=6000),
    )


def edit_docx_tool(call: ToolCall, state: ToolState) -> str:
    if state.current_docx is None:
        return observation_xml("edit_docx", "error", "수정할 DOCX 파일이 없습니다.")
    actions_xml = (call.args or {}).get("actions_xml", "").strip()
    if not actions_xml:
        return observation_xml("edit_docx", "error", "actions_xml 인자가 비어 있습니다.")
    if not actions_xml.startswith("<actions"):
        actions_xml = f"<actions>{actions_xml}</actions>"
    _xml_text, actions = parse_actions_output(actions_xml)
    state.last_attempt += 1
    result_path = state.run_dir / f"result_step{state.last_attempt}.docx"
    changed = apply_actions(state.current_docx, actions, result_path)
    after = read_docx_context(result_path)
    before = state.last_context or state.source_context
    diff_context = build_diff_context(before.xml, after.xml) if before else ""
    state.current_docx = result_path
    state.last_result_path = result_path
    state.last_context = after
    state.last_actions_xml = actions_xml
    state.run_dir.joinpath(f"actions_step{state.last_attempt}.xml").write_text(actions_xml, encoding="utf-8")
    state.run_dir.joinpath(f"after_context_step{state.last_attempt}.xml").write_text(after.xml, encoding="utf-8")
    if diff_context:
        state.run_dir.joinpath(f"diff_context_step{state.last_attempt}.xml").write_text(diff_context, encoding="utf-8")
    return observation_xml(
        "edit_docx",
        "ok",
        "DOCX 수정 도구를 실행했습니다.",
        result_path=str(result_path),
        changed=str(changed),
        after_preview=compact_context(after.xml, limit=3000),
        diff_preview=compact_context(diff_context, limit=3000),
    )


def create_docx_tool(call: ToolCall, state: ToolState, request: str) -> str:
    document_xml = (call.args or {}).get("document_xml", "").strip()
    if document_xml:
        root = etree.fromstring(document_xml.encode("utf-8"))
    else:
        analysis_xml = (call.args or {}).get("analysis_xml", "<analysis_result><summary>새 문서 생성</summary></analysis_result>")
        document_xml, root = generate_document_xml(
            request,
            analysis_xml,
            state.model,
            state.run_dir,
            source_context=state.source_context,
        )
    state.last_attempt += 1
    result_path = state.run_dir / f"document_step{state.last_attempt}.docx"
    write_document_xml(root, result_path)
    after = read_docx_context(result_path)
    state.current_docx = result_path
    state.last_result_path = result_path
    state.last_context = after
    state.last_document_xml = document_xml
    state.run_dir.joinpath(f"document_step{state.last_attempt}.xml").write_text(document_xml, encoding="utf-8")
    state.run_dir.joinpath(f"after_context_step{state.last_attempt}.xml").write_text(after.xml, encoding="utf-8")
    return observation_xml(
        "create_docx",
        "ok",
        "새 DOCX 문서를 생성했습니다.",
        result_path=str(result_path),
        after_preview=compact_context(after.xml, limit=3500),
    )


def evaluate_result_tool(state: ToolState, request: str) -> str:
    if state.last_result_path is None or state.last_context is None:
        return observation_xml("evaluate_result", "error", "평가할 결과 문서가 없습니다.")
    before_xml = state.source_context.xml if state.source_context is not None else "<docx_context><paragraphs/><tables/><editable_candidates/></docx_context>"
    diff_context = build_diff_context(before_xml, state.last_context.xml)
    payload_xml = state.last_actions_xml or state.last_document_xml or "<result/>"
    evaluation_xml, evaluation = evaluate_result(
        request,
        "<request_analysis><summary>tool loop evaluation</summary></request_analysis>",
        payload_xml,
        before_xml,
        state.last_context.xml,
        diff_context,
        state.model,
        state.run_dir,
        state.last_attempt,
    )
    state.last_evaluation_xml = evaluation_xml
    return observation_xml(
        "evaluate_result",
        "ok",
        "평가를 완료했습니다.",
        passed=str(evaluation.passed).lower(),
        reason=evaluation.reason,
        feedback=evaluation.feedback,
        evaluation_xml=evaluation_xml,
    )


def finish_tool(call: ToolCall, state: ToolState) -> str:
    status = (call.args or {}).get("status", "done")
    summary = (call.args or {}).get("summary", call.reason)
    return observation_xml(
        "finish",
        "ok",
        "작업을 종료합니다.",
        status=status,
        summary=summary,
        result_path=str(state.last_result_path or ""),
    )
