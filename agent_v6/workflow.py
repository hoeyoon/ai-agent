from pathlib import Path

from agent_v6.context import compact_context
from agent_v6.llm import run_llm
from agent_v6.logger import append_jsonl, make_run_dir
from agent_v6.tool_executor import ToolState, evaluate_result_tool, execute_tool
from agent_v6.tool_protocol import ToolCall, observation_xml, parse_tool_call_output
from agent_v6.xml_protocol import extract_required_xml, parse_strict_xml


PLANNER_SYSTEM = (
    "당신은 DOCX AI 에이전트의 Workflow Planner입니다. "
    "사용자 목표를 분석해 실행 전략을 세우세요. 내부 추론은 출력하지 말고 "
    "반드시 <workflow_plan> XML만 출력하세요."
)

WORKER_SYSTEM = (
    "당신은 DOCX AI 에이전트의 Tool Worker입니다. "
    "workflow_plan과 observation을 보고 다음에 실행할 도구 하나만 선택하세요. "
    "내부 추론은 출력하지 말고 반드시 완전한 <tool_call> XML만 출력하세요."
)


class WorkflowError(RuntimeError):
    pass


def run_agentic_workflow(
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

    plan_xml = create_workflow_plan(request, input_path, model, run_dir)
    observations.append(observation_xml("workflow_plan", "ok", "워크플로우 계획을 생성했습니다.", plan_preview=compact_context(plan_xml, limit=2500)))

    last_tool_xml = ""
    last_observation = observations[-1]

    for step in range(1, max_steps + 1):
        tool_xml, call = choose_next_tool(request, input_path, plan_xml, observations, model, run_dir, step)
        last_tool_xml = tool_xml
        run_dir.joinpath(f"tool_call_step{step}.xml").write_text(tool_xml, encoding="utf-8")

        obs = execute_tool(call, state, request)
        observations.append(obs)
        last_observation = obs
        run_dir.joinpath(f"observation_step{step}.xml").write_text(obs, encoding="utf-8")

        append_jsonl(
            {
                "mode": "agentic_workflow_v6",
                "request": request,
                "model": model,
                "step": step,
                "phase": "tool_use",
                "tool": call.name,
                "run_dir": str(run_dir),
                "workflow_plan_xml": plan_xml,
                "tool_call_xml": tool_xml,
                "observation_xml": obs,
                "passed": call.name == "finish",
            }
        )

        if call.name == "finish":
            run_dir.joinpath("final_observation.xml").write_text(obs, encoding="utf-8")
            return run_dir, state.last_result_path, obs

        if call.name in {"create_docx", "edit_docx"} and observation_ok(obs):
            eval_obs = evaluate_result_tool(state, request)
            observations.append(eval_obs)
            last_observation = eval_obs
            run_dir.joinpath(f"evaluation_observation_step{step}.xml").write_text(eval_obs, encoding="utf-8")
            append_jsonl(
                {
                    "mode": "agentic_workflow_v6",
                    "request": request,
                    "model": model,
                    "step": step,
                    "phase": "reflection",
                    "tool": "evaluate_result",
                    "run_dir": str(run_dir),
                    "workflow_plan_xml": plan_xml,
                    "observation_xml": eval_obs,
                    "passed": evaluation_passed(eval_obs),
                }
            )
            if evaluation_passed(eval_obs):
                final_obs = observation_xml(
                    "finish",
                    "ok",
                    "LLM 평가를 통과하여 작업을 종료합니다.",
                    status="done",
                    summary=evaluation_reason(eval_obs),
                    result_path=str(state.last_result_path or ""),
                )
                run_dir.joinpath("final_observation.xml").write_text(final_obs, encoding="utf-8")
                return run_dir, state.last_result_path, final_obs

    raise WorkflowError(f"최대 step {max_steps} 안에 작업을 완료하지 못했습니다. 마지막 tool_call: {last_tool_xml[:300]} 마지막 observation: {last_observation[:300]}")


def create_workflow_plan(request: str, input_path: Path | None, model: str, run_dir: Path) -> str:
    raw = run_llm(model, PLANNER_SYSTEM, build_planner_prompt(request, input_path), response_prefix="", num_predict=1024)
    run_dir.joinpath("workflow_plan_raw.txt").write_text(raw, encoding="utf-8")
    try:
        plan_xml = extract_required_xml(raw, "workflow_plan")
        parse_strict_xml(plan_xml)
    except Exception as exc:
        retry_raw = run_llm(
            model,
            PLANNER_SYSTEM,
            build_planner_retry_prompt(request, input_path, str(exc)),
            response_prefix="",
            num_predict=1024,
        )
        run_dir.joinpath("workflow_plan_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
        plan_xml = extract_required_xml(retry_raw, "workflow_plan")
        parse_strict_xml(plan_xml)
    run_dir.joinpath("workflow_plan.xml").write_text(plan_xml, encoding="utf-8")
    return plan_xml


def choose_next_tool(
    request: str,
    input_path: Path | None,
    plan_xml: str,
    observations: list[str],
    model: str,
    run_dir: Path,
    step: int,
) -> tuple[str, ToolCall]:
    raw = run_llm(
        model,
        WORKER_SYSTEM,
        build_worker_prompt(request, input_path, plan_xml, observations),
        response_prefix="",
        num_predict=1536,
    )
    run_dir.joinpath(f"tool_call_step{step}_raw.txt").write_text(raw, encoding="utf-8")
    try:
        return parse_tool_call_output(raw)
    except Exception as exc:
        retry_raw = run_llm(
            model,
            WORKER_SYSTEM,
            build_worker_retry_prompt(request, input_path, plan_xml, observations, str(exc)),
            response_prefix="",
            num_predict=1536,
        )
        run_dir.joinpath(f"tool_call_step{step}_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
        return parse_tool_call_output(retry_raw)


def build_planner_prompt(request: str, input_path: Path | None) -> str:
    input_text = str(input_path) if input_path else "없음"
    return (
        f"사용자 요청:\n{request}\n\n"
        f"열린 DOCX 파일:\n{input_text}\n\n"
        "에이전틱 워크플로우 원리:\n"
        "1. Planning: 목표를 하위 작업으로 나눕니다.\n"
        "2. Routing: 새 문서 생성, 기존 문서 수정, 문서 분석 중 알맞은 경로를 고릅니다.\n"
        "3. Tool Use: 필요한 도구를 순서대로 사용합니다.\n"
        "4. Reflection: 결과를 LLM 평가로 검증하고 부족하면 수정합니다.\n"
        "5. Collaboration: planner, worker, reviewer 역할이 observation을 주고받습니다.\n\n"
        "출력 형식:\n"
        "<workflow_plan>\n"
        "  <goal>사용자 목표 요약</goal>\n"
        "  <route>create|edit|analyze_then_create|analyze_then_edit</route>\n"
        "  <needs_input_docx>true 또는 false</needs_input_docx>\n"
        "  <steps>\n"
        "    <step order=\"1\" actor=\"worker\" tool=\"read_docx_context\">실행 이유</step>\n"
        "    <step order=\"2\" actor=\"worker\" tool=\"create_docx\">실행 이유</step>\n"
        "    <step order=\"3\" actor=\"reviewer\" tool=\"evaluate_result\">검증 이유</step>\n"
        "  </steps>\n"
        "  <success_criteria>완료 기준</success_criteria>\n"
        "</workflow_plan>\n"
        "마지막 </workflow_plan> 뒤에는 아무것도 출력하지 마세요."
    )


def build_planner_retry_prompt(request: str, input_path: Path | None, error: str) -> str:
    return build_planner_prompt(request, input_path) + f"\n\n이전 계획 XML 오류: {error}\n완전한 <workflow_plan> XML만 다시 출력하세요."


def build_worker_prompt(request: str, input_path: Path | None, plan_xml: str, observations: list[str]) -> str:
    input_block = f"열린 DOCX 경로: {input_path}\n" if input_path else "열린 DOCX 경로: 없음\n"
    observation_block = "\n".join(compact_context(item, limit=2500) for item in observations[-6:])
    return (
        f"사용자 요청:\n{request}\n\n"
        f"{input_block}\n"
        f"workflow_plan:\n{compact_context(plan_xml, limit=3500)}\n\n"
        f"최근 observation:\n{observation_block if observation_block else '(아직 없음)'}\n\n"
        "사용 가능한 도구:\n"
        "- read_docx_context: DOCX 구조와 내용을 읽습니다.\n"
        "- analyze_docx: 읽은 context를 요약합니다.\n"
        "- create_docx: 새 DOCX를 생성합니다.\n"
        "- edit_docx: 기존 DOCX를 수정합니다.\n"
        "- evaluate_result: 마지막 결과 문서를 LLM으로 평가합니다.\n"
        "- finish: 평가가 통과했거나 결과가 충분하면 종료합니다.\n\n"
        "선택 규칙:\n"
        "- 열린 DOCX가 있고 아직 read_docx_context observation이 없다면 먼저 read_docx_context를 선택하세요.\n"
        "- create_docx 또는 edit_docx 성공 뒤에는 시스템이 자동으로 evaluate_result observation을 추가합니다.\n"
        "- evaluate_result observation의 passed가 true이면 finish를 선택하세요.\n"
        "- evaluate_result observation의 passed가 false이면 feedback을 반영해 create_docx 또는 edit_docx를 다시 선택하세요.\n"
        "- 도구는 한 번에 하나만 선택하세요.\n"
        "- 내부 분석, 마크다운, 설명문을 출력하지 마세요.\n\n"
        "출력 예시 중 하나의 형식만 사용하세요:\n"
        "<tool_call name=\"read_docx_context\"><reason>문서 분석</reason><args><path>파일 경로</path></args></tool_call>\n"
        "<tool_call name=\"create_docx\"><reason>새 문서 생성</reason><args><document_xml><![CDATA[<document><title>제목</title><description>설명</description><section><heading>섹션</heading><paragraph>본문</paragraph></section></document>]]></document_xml></args></tool_call>\n"
        "<tool_call name=\"edit_docx\"><reason>문서 수정</reason><args><actions_xml><![CDATA[<actions><replace_paragraph index=\"0\" value=\"새 문장\"/></actions>]]></actions_xml></args></tool_call>\n"
        "<tool_call name=\"evaluate_result\"><reason>결과 검증</reason><args/></tool_call>\n"
        "<tool_call name=\"finish\"><reason>완료</reason><args><status>done</status><summary>완료 요약</summary></args></tool_call>"
    )


def build_worker_retry_prompt(
    request: str,
    input_path: Path | None,
    plan_xml: str,
    observations: list[str],
    error: str,
) -> str:
    return (
        build_worker_prompt(request, input_path, plan_xml, observations)
        + f"\n\n이전 tool_call XML 오류: {error}\n완전한 <tool_call> XML 하나만 다시 출력하세요."
    )


def observation_ok(observation: str) -> bool:
    return 'status="ok"' in observation


def evaluation_passed(observation: str) -> bool:
    lowered = observation.lower()
    return "<passed>true</passed>" in lowered or "passed=\"true\"" in lowered


def evaluation_reason(observation: str) -> str:
    root = parse_strict_xml(observation)
    reason = root.findtext("reason") or root.findtext("message") or "평가 통과"
    return " ".join(reason.split())
