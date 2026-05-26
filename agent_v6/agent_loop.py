from pathlib import Path

from agent_v6.context import compact_context
from agent_v6.llm import run_llm
from agent_v6.logger import append_jsonl, make_run_dir
from agent_v6.tool_executor import ToolState, execute_tool
from agent_v6.tool_protocol import ToolCall, observation_xml, parse_tool_call_output
from agent_v6.xml_protocol import ProtocolError


AGENT_SYSTEM = (
    "당신은 DOCX 작업을 수행하는 Codex/Claude Code 스타일 AI 에이전트입니다. "
    "매 턴 하나의 도구만 선택합니다. 내부적으로 판단하되 분석 과정은 출력하지 마세요. "
    "반드시 완전한 <tool_call> XML만 출력하세요."
)


class AgentLoopError(RuntimeError):
    pass


def run_tool_loop(
    request: str,
    input_path: Path | None,
    out_dir: Path,
    prefix: str,
    model: str,
    max_steps: int = 8,
) -> tuple[Path, Path | None, str]:
    run_dir = make_run_dir(out_dir, prefix)
    run_dir.joinpath("request.txt").write_text(request, encoding="utf-8")
    if input_path:
        run_dir.joinpath("input_path.txt").write_text(str(input_path), encoding="utf-8")

    state = ToolState(run_dir=run_dir, model=model, input_path=input_path)
    observations: list[str] = []
    last_tool_xml = ""

    for step in range(1, max_steps + 1):
        prompt = build_agent_prompt(request, input_path, observations)
        raw = run_llm(model, AGENT_SYSTEM, prompt, response_prefix="", num_predict=1536)
        (run_dir / f"tool_call_step{step}_raw.txt").write_text(raw, encoding="utf-8")
        try:
            tool_xml, call = parse_tool_call_output(raw)
        except Exception as exc:
            retry_raw = run_llm(
                model,
                AGENT_SYSTEM,
                build_retry_prompt(request, input_path, observations, str(exc)),
                response_prefix="",
                num_predict=1536,
            )
            (run_dir / f"tool_call_step{step}_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
            tool_xml, call = parse_tool_call_output(retry_raw)

        last_tool_xml = tool_xml
        (run_dir / f"tool_call_step{step}.xml").write_text(tool_xml, encoding="utf-8")
        obs = execute_tool(call, state, request)
        observations.append(obs)
        (run_dir / f"observation_step{step}.xml").write_text(obs, encoding="utf-8")

        append_jsonl(
            {
                "mode": "tool_loop",
                "request": request,
                "model": model,
                "step": step,
                "tool": call.name,
                "run_dir": str(run_dir),
                "tool_call_xml": tool_xml,
                "observation_xml": obs,
                "passed": call.name == "finish",
            }
        )

        if call.name == "finish":
            return run_dir, state.last_result_path, obs

    raise AgentLoopError(f"최대 step {max_steps} 안에 finish 도구가 호출되지 않았습니다. 마지막 tool_call: {last_tool_xml[:300]}")


def build_agent_prompt(request: str, input_path: Path | None, observations: list[str]) -> str:
    observation_block = "\n".join(
        compact_context(observation, limit=2500) for observation in observations[-5:]
    )
    input_block = f"열린 DOCX 경로: {input_path}\n" if input_path else "열린 DOCX 경로: 없음\n"
    return (
        f"사용자 요청:\n{request}\n\n"
        f"{input_block}\n"
        f"최근 observation:\n{observation_block if observation_block else '(아직 없음)'}\n\n"
        "너는 다음에 실행할 도구를 하나만 선택해야 합니다.\n"
        "도구 목록:\n"
        "1. read_docx_context: DOCX 파일을 읽고 문단/표/셀 context를 얻습니다.\n"
        "2. analyze_docx: 읽은 context를 더 짧은 observation으로 정리합니다.\n"
        "3. edit_docx: 기존 DOCX를 수정합니다. args/actions_xml 안에 actions XML을 넣습니다.\n"
        "4. create_docx: 새 DOCX를 생성합니다. args/document_xml 안에 document XML을 직접 넣거나 비워 둘 수 있습니다.\n"
        "5. evaluate_result: 마지막 결과 문서를 평가합니다.\n"
        "6. finish: 결과가 충분하면 종료합니다.\n\n"
        "일반 전략:\n"
        "- 열린 DOCX가 있고 아직 읽지 않았다면 먼저 read_docx_context를 호출하세요.\n"
        "- 기존 문서를 고치는 요청이면 edit_docx를 호출하세요.\n"
        "- 열린 문서를 분석해 새 문서를 만드는 요청이면 read_docx_context 후 create_docx를 호출하세요.\n"
        "- 결과 생성이나 수정 뒤에는 evaluate_result를 호출하세요.\n"
        "- 평가가 통과했거나 결과가 충분하면 finish를 호출하세요.\n"
        "- XML 본문 값에는 태그 예시를 꺾쇠괄호로 쓰지 마세요.\n"
        "- 반드시 아래 형식 중 하나만 출력하세요.\n\n"
        "<tool_call name=\"read_docx_context\">\n"
        "  <reason>문서를 읽기 위해 실행</reason>\n"
        "  <args><path>파일 경로</path></args>\n"
        "</tool_call>\n\n"
        "<tool_call name=\"edit_docx\">\n"
        "  <reason>기존 문서 수정</reason>\n"
        "  <args><actions_xml><![CDATA[<actions><replace_table_value target=\"table[0] row[0] cell[1]\" value=\"새 값\"/></actions>]]></actions_xml></args>\n"
        "</tool_call>\n\n"
        "<tool_call name=\"create_docx\">\n"
        "  <reason>새 문서 생성</reason>\n"
        "  <args><document_xml><![CDATA[<document><title>제목</title><description>설명</description><section><heading>섹션</heading><paragraph>본문</paragraph></section></document>]]></document_xml></args>\n"
        "</tool_call>\n\n"
        "<tool_call name=\"evaluate_result\"><reason>결과 검증</reason><args/></tool_call>\n"
        "<tool_call name=\"finish\"><reason>완료</reason><args><status>done</status><summary>완료 요약</summary></args></tool_call>\n"
        "마지막 닫는 </tool_call> 뒤에는 아무것도 출력하지 마세요."
    )


def build_retry_prompt(request: str, input_path: Path | None, observations: list[str], error: str) -> str:
    return (
        build_agent_prompt(request, input_path, observations)
        + f"\n\n이전 출력 오류: {error}\n완전한 <tool_call> XML 하나만 다시 출력하세요."
    )

