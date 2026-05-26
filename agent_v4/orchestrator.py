from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from agent_v4.context import DocxContext, compact_context
from agent_v4.llm import run_llm
from agent_v4.xml_protocol import ProtocolError, child_text, extract_required_xml, parse_strict_xml


ORCHESTRATOR_SYSTEM = (
    "당신은 DOCX 에이전트의 Orchestrator입니다. "
    "사용자 요청을 하위 분석 작업으로 나누고 Worker 작업 목록을 만듭니다. "
    "내부적으로 판단하되 분석 과정은 출력하지 마세요. "
    "반드시 완전한 <work_plan> XML만 출력하세요."
)

WORKER_SYSTEM = (
    "당신은 DOCX 에이전트의 Worker입니다. "
    "할당된 하위 작업만 수행하고 결과를 요약합니다. "
    "내부적으로 판단하되 분석 과정은 출력하지 마세요. "
    "반드시 완전한 <worker_result> XML만 출력하세요."
)


@dataclass
class WorkerTask:
    task_id: str
    role: str
    instruction: str


def build_worker_context(
    request: str,
    analysis_xml: str,
    source_context: DocxContext | None,
    model: str,
    run_dir: Path,
    max_tasks: int = 3,
    parallel_workers: int = 1,
) -> str:
    if source_context is None:
        return "<worker_context><note>열린 원본 DOCX가 없어 Worker 분석을 생략했습니다.</note></worker_context>"

    try:
        plan_xml, tasks = plan_worker_tasks(request, analysis_xml, source_context, model, run_dir, max_tasks)
    except Exception as exc:
        run_dir.joinpath("orchestrator_error.txt").write_text(str(exc), encoding="utf-8")
        return fallback_worker_context(f"Orchestrator 작업 계획 생성 실패: {exc}")
    run_dir.joinpath("work_plan.xml").write_text(plan_xml, encoding="utf-8")
    if not tasks:
        return "<worker_context><note>생성된 Worker 작업이 없습니다.</note></worker_context>"

    workers = max(1, min(parallel_workers, len(tasks)))
    if workers == 1:
        results = [safe_run_worker_task(task, request, analysis_xml, source_context, model, run_dir) for task in tasks]
    else:
        results = run_worker_tasks_parallel(tasks, request, analysis_xml, source_context, model, run_dir, workers)

    return build_worker_context_xml(plan_xml, results, workers)


def plan_worker_tasks(
    request: str,
    analysis_xml: str,
    source_context: DocxContext,
    model: str,
    run_dir: Path,
    max_tasks: int,
) -> tuple[str, list[WorkerTask]]:
    prompt = (
        f"사용자 요청:\n{request}\n\n"
        f"Analyzer 결과 XML:\n{analysis_xml}\n\n"
        f"원본 DOCX 구조와 내용:\n{compact_context(source_context.xml, limit=7000)}\n\n"
        f"최대 {max_tasks}개의 하위 작업을 만드세요.\n"
        "각 작업은 Writer가 새 문서를 작성하는 데 필요한 독립 분석이어야 합니다.\n"
        "권장 역할 예: 핵심요약, 주요근거, 문서구조, 결론정리.\n"
        "작업 지시에는 XML 태그 예시나 꺾쇠괄호 문자를 쓰지 마세요.\n"
        "반드시 아래 XML만 출력하세요.\n"
        "<work_plan>\n"
        "  <task id=\"1\" role=\"핵심요약\">작업 지시</task>\n"
        "</work_plan>\n"
        "마지막 </work_plan> 뒤에는 아무것도 출력하지 마세요."
    )
    raw = run_llm(model, ORCHESTRATOR_SYSTEM, prompt, response_prefix="<work_plan>\n", num_predict=768)
    (run_dir / "orchestrator_raw.txt").write_text(raw, encoding="utf-8")
    try:
        xml_text, tasks = parse_work_plan(raw, max_tasks)
    except Exception:
        retry_raw = run_llm(
            model,
            ORCHESTRATOR_SYSTEM,
            prompt + "\nXML 문법을 엄격히 지켜 다시 출력하세요. 설명문 안에 < 또는 > 문자를 쓰지 마세요.",
            response_prefix="<work_plan>\n",
            num_predict=768,
        )
        (run_dir / "orchestrator_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
        xml_text, tasks = parse_work_plan(retry_raw, max_tasks)
    return xml_text, tasks


def parse_work_plan(text: str, max_tasks: int) -> tuple[str, list[WorkerTask]]:
    xml_text = extract_required_xml(text, "work_plan")
    root = parse_strict_xml(xml_text)
    if root.tag != "work_plan":
        raise ProtocolError("루트가 <work_plan>이 아닙니다.")
    tasks = []
    for index, task in enumerate(root.findall("task"), start=1):
        if len(tasks) >= max_tasks:
            break
        instruction = " ".join("".join(task.itertext()).split())
        if not instruction:
            continue
        task_id = safe_identifier(task.get("id") or str(index), str(index))
        tasks.append(
            WorkerTask(
                task_id=task_id,
                role=task.get("role") or "분석",
                instruction=instruction,
            )
        )
    if not tasks:
        raise ProtocolError("work_plan에 task가 없습니다.")
    return xml_text, tasks


def run_worker_tasks_parallel(
    tasks: list[WorkerTask],
    request: str,
    analysis_xml: str,
    source_context: DocxContext,
    model: str,
    run_dir: Path,
    workers: int,
) -> list[str]:
    results_by_id = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_worker_task, task, request, analysis_xml, source_context, model, run_dir): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                results_by_id[task.task_id] = future.result()
            except Exception as exc:
                results_by_id[task.task_id] = fallback_worker_result(task, str(exc))
    return [results_by_id[task.task_id] for task in tasks]


def safe_run_worker_task(
    task: WorkerTask,
    request: str,
    analysis_xml: str,
    source_context: DocxContext,
    model: str,
    run_dir: Path,
) -> str:
    try:
        return run_worker_task(task, request, analysis_xml, source_context, model, run_dir)
    except Exception as exc:
        safe_name = safe_identifier(task.task_id, "task")
        (run_dir / f"worker_{safe_name}_error.txt").write_text(str(exc), encoding="utf-8")
        return fallback_worker_result(task, str(exc))


def run_worker_task(
    task: WorkerTask,
    request: str,
    analysis_xml: str,
    source_context: DocxContext,
    model: str,
    run_dir: Path,
) -> str:
    prompt = (
        f"사용자 요청:\n{request}\n\n"
        f"Analyzer 결과 XML:\n{analysis_xml}\n\n"
        f"Worker 역할: {task.role}\n"
        f"Worker 작업: {task.instruction}\n\n"
        f"원본 DOCX 구조와 내용:\n{compact_context(source_context.xml, limit=9000)}\n\n"
        "할당된 작업만 수행하세요. Writer가 새 문서를 작성할 수 있도록 핵심 결과를 간결하게 정리하세요.\n"
        "findings와 recommendation에는 XML 태그 예시나 꺾쇠괄호 문자를 쓰지 마세요.\n"
        "점 세 개 같은 placeholder를 쓰지 말고 실제 분석 결과를 작성하세요.\n"
        "반드시 아래 XML만 출력하세요.\n"
        f"<worker_result id=\"{escape_attr(task.task_id)}\" role=\"{escape_attr(task.role)}\">\n"
        "  <findings>핵심 분석 결과</findings>\n"
        "  <recommendation>Writer에게 줄 작성 권고</recommendation>\n"
        "</worker_result>\n"
        "마지막 </worker_result> 뒤에는 아무것도 출력하지 마세요."
    )
    raw = run_llm(
        model,
        WORKER_SYSTEM,
        prompt,
        response_prefix=f"<worker_result id=\"{escape_attr(task.task_id)}\"",
        num_predict=768,
    )
    safe_name = safe_identifier(task.task_id, "task")
    (run_dir / f"worker_{safe_name}_raw.txt").write_text(raw, encoding="utf-8")
    xml_text = extract_required_xml(raw, "worker_result")
    root = parse_strict_xml(xml_text)
    if root.tag != "worker_result":
        raise ProtocolError("루트가 <worker_result>가 아닙니다.")
    if not child_text(root, "findings"):
        raise ProtocolError("worker_result findings가 비어 있습니다.")
    (run_dir / f"worker_{safe_name}.xml").write_text(xml_text, encoding="utf-8")
    return xml_text


def build_worker_context_xml(plan_xml: str, results: list[str], workers: int) -> str:
    root = etree.Element("worker_context", parallel_workers=str(workers))
    plan = etree.SubElement(root, "plan")
    plan.append(parse_strict_xml(plan_xml))
    results_root = etree.SubElement(root, "results")
    for result_xml in results:
        results_root.append(parse_strict_xml(result_xml))
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def fallback_worker_context(message: str) -> str:
    root = etree.Element("worker_context", parallel_workers="0")
    etree.SubElement(root, "note").text = message
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def fallback_worker_result(task: WorkerTask, message: str) -> str:
    root = etree.Element("worker_result", id=task.task_id, role=task.role)
    etree.SubElement(root, "findings").text = f"Worker 분석 실패: {message}"
    etree.SubElement(root, "recommendation").text = "Writer는 Analyzer 결과와 원본 DOCX context를 근거로 문서를 작성하세요."
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def safe_identifier(value: str, fallback: str) -> str:
    safe = "".join(ch for ch in str(value) if ch.isalnum() or ch in ("-", "_"))
    return safe or fallback


def escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace("\"", "&quot;").replace("<", "").replace(">", "")
