from pathlib import Path

from agent_v4.analyzer import analyze_request
from agent_v4.combined_planner import plan_agent_result
from agent_v4.context import DocxContext, read_docx_context
from agent_v4.diff import build_diff_context
from agent_v4.docx_tools import ToolError, apply_actions, write_document_xml
from agent_v4.evaluator import evaluate_result
from agent_v4.generator import generate_document_xml
from agent_v4.logger import append_jsonl, make_run_dir, update_few_shot_examples
from agent_v4.orchestrator import build_worker_context
from agent_v4.xml_protocol import ProtocolError


class AgentError(RuntimeError):
    pass


EMPTY_CONTEXT_XML = "<docx_context><paragraphs/><tables/><editable_candidates/></docx_context>"


def run_agent(
    request: str,
    input_path: Path | None,
    out_dir: Path,
    prefix: str,
    model: str,
    max_attempts: int = 2,
    parallel_workers: int = 1,
    worker_tasks: int = 3,
) -> tuple[Path, Path, str, int | None]:
    run_dir = make_run_dir(out_dir, prefix)
    run_dir.joinpath("request.txt").write_text(request, encoding="utf-8")
    if input_path:
        run_dir.joinpath("input_path.txt").write_text(str(input_path), encoding="utf-8")

    source_context = read_source_context(input_path, run_dir)
    analysis_xml, analysis = analyze_request(request, source_context, model, run_dir, attempt=1)
    run_dir.joinpath("selected_mode.txt").write_text(analysis.mode, encoding="utf-8")

    if analysis.mode == "edit_existing":
        if input_path is None or source_context is None:
            raise AgentError("edit_existing 모드에는 열린 DOCX 파일이 필요합니다.")
        result_run_dir, result_path, changed = edit_existing(
            input_path,
            request,
            analysis_xml,
            source_context,
            run_dir,
            model,
            max_attempts,
        )
        return result_run_dir, result_path, analysis.mode, changed

    result_run_dir, result_path = create_from_analysis(
        request,
        analysis_xml,
        source_context,
        run_dir,
        model,
        max_attempts,
        analysis.mode,
        parallel_workers,
        worker_tasks,
    )
    return result_run_dir, result_path, analysis.mode, None


def read_source_context(input_path: Path | None, run_dir: Path) -> DocxContext | None:
    if input_path is None:
        return None
    context = read_docx_context(input_path)
    run_dir.joinpath("source_context.txt").write_text(context.text, encoding="utf-8")
    run_dir.joinpath("source_context.xml").write_text(context.xml, encoding="utf-8")
    return context


def edit_existing(
    input_path: Path,
    request: str,
    analysis_xml: str,
    before: DocxContext,
    run_dir: Path,
    model: str,
    max_attempts: int,
) -> tuple[Path, Path, int]:
    feedback = ""
    last_reason = ""
    for attempt in range(1, max_attempts + 1):
        try:
            agent_result_xml, request_analysis_xml, actions_xml, actions = plan_agent_result(
                request,
                before,
                model,
                run_dir,
                analysis_xml=analysis_xml,
                feedback=feedback,
                attempt=attempt,
            )
            result_path = run_dir / f"result_attempt{attempt}.docx"
            changed = apply_actions(input_path, actions, result_path)
            after = read_docx_context(result_path)
            run_dir.joinpath(f"after_context_attempt{attempt}.txt").write_text(after.text, encoding="utf-8")
            run_dir.joinpath(f"after_context_attempt{attempt}.xml").write_text(after.xml, encoding="utf-8")
            diff_context = build_diff_context(before.xml, after.xml)
            run_dir.joinpath(f"diff_context_attempt{attempt}.xml").write_text(diff_context, encoding="utf-8")

            evaluation_xml, evaluation = evaluate_result(
                request,
                request_analysis_xml,
                actions_xml,
                before.xml,
                after.xml,
                diff_context,
                model,
                run_dir,
                attempt,
            )
            record = {
                "mode": "edit_existing",
                "request": request,
                "model": model,
                "attempt": attempt,
                "passed": evaluation.passed,
                "reason": evaluation.reason,
                "feedback": evaluation.feedback,
                "changed": changed,
                "run_dir": str(run_dir),
                "analysis_xml": analysis_xml,
                "agent_result_xml": agent_result_xml,
                "request_analysis_xml": request_analysis_xml,
                "actions_xml": actions_xml,
                "evaluation_xml": evaluation_xml,
            }
            append_jsonl(record)
            update_few_shot_examples(record)
            if evaluation.passed:
                return run_dir, result_path, changed
            feedback = evaluation.feedback
            last_reason = evaluation.reason
        except (ProtocolError, ToolError) as exc:
            last_reason = str(exc)
            feedback = f"이전 시도 실행 실패: {last_reason}. 실행 가능한 XML만 다시 작성하세요."
            run_dir.joinpath(f"error_attempt{attempt}.txt").write_text(last_reason, encoding="utf-8")
            append_failure(run_dir, request, model, "edit_existing", attempt, last_reason, feedback)
    raise AgentError(f"검증을 통과하지 못했습니다: {last_reason}")


def create_from_analysis(
    request: str,
    analysis_xml: str,
    source_context: DocxContext | None,
    run_dir: Path,
    model: str,
    max_attempts: int,
    mode: str,
    parallel_workers: int,
    worker_tasks: int,
) -> tuple[Path, Path]:
    feedback = ""
    last_reason = ""
    before_xml = EMPTY_CONTEXT_XML
    worker_context_xml = build_worker_context(
        request,
        analysis_xml,
        source_context,
        model,
        run_dir,
        max_tasks=worker_tasks,
        parallel_workers=parallel_workers,
    )
    run_dir.joinpath("worker_context.xml").write_text(worker_context_xml, encoding="utf-8")
    for attempt in range(1, max_attempts + 1):
        try:
            document_xml, root = generate_document_xml(
                request,
                analysis_xml,
                model,
                run_dir,
                feedback,
                attempt,
                source_context=source_context,
                worker_context_xml=worker_context_xml,
            )
            result_path = run_dir / f"document_attempt{attempt}.docx"
            write_document_xml(root, result_path)
            after = read_docx_context(result_path)
            run_dir.joinpath(f"after_context_attempt{attempt}.txt").write_text(after.text, encoding="utf-8")
            run_dir.joinpath(f"after_context_attempt{attempt}.xml").write_text(after.xml, encoding="utf-8")
            diff_context = build_diff_context(before_xml, after.xml)
            run_dir.joinpath(f"diff_context_attempt{attempt}.xml").write_text(diff_context, encoding="utf-8")

            evaluation_xml, evaluation = evaluate_result(
                request,
                analysis_xml,
                document_xml,
                before_xml,
                after.xml,
                diff_context,
                model,
                run_dir,
                attempt,
            )
            record = {
                "mode": mode,
                "request": request,
                "model": model,
                "attempt": attempt,
                "passed": evaluation.passed,
                "reason": evaluation.reason,
                "feedback": evaluation.feedback,
                "run_dir": str(run_dir),
                "analysis_xml": analysis_xml,
                "worker_context_xml": worker_context_xml,
                "document_xml": document_xml,
                "evaluation_xml": evaluation_xml,
            }
            append_jsonl(record)
            if evaluation.passed:
                return run_dir, result_path
            feedback = evaluation.feedback
            last_reason = evaluation.reason
        except ProtocolError as exc:
            last_reason = str(exc)
            feedback = f"이전 문서 XML 생성 실패: {last_reason}. 완전한 XML만 다시 작성하세요."
            run_dir.joinpath(f"error_attempt{attempt}.txt").write_text(last_reason, encoding="utf-8")
            append_failure(run_dir, request, model, mode, attempt, last_reason, feedback)
    raise AgentError(f"문서 생성을 완료하지 못했습니다: {last_reason}")


def append_failure(run_dir: Path, request: str, model: str, mode: str, attempt: int, reason: str, feedback: str) -> None:
    append_jsonl(
        {
            "mode": mode,
            "request": request,
            "model": model,
            "attempt": attempt,
            "passed": False,
            "reason": reason,
            "feedback": feedback,
            "run_dir": str(run_dir),
        }
    )


def edit_docx(input_path: Path, request: str, out_dir: Path, prefix: str, model: str, max_attempts: int = 2):
    run_dir, result_path, _mode, changed = run_agent(request, input_path, out_dir, prefix, model, max_attempts)
    return run_dir, result_path, changed or 0


def create_docx(request: str, out_dir: Path, prefix: str, model: str, max_attempts: int = 2):
    run_dir, result_path, _mode, _changed = run_agent(request, None, out_dir, prefix, model, max_attempts)
    return run_dir, result_path
