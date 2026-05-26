from pathlib import Path

from agent_v4.context import DocxContext, compact_context
from agent_v4.llm import run_llm
from agent_v4.xml_protocol import AnalysisResult, parse_analysis_result


ANALYZER_SYSTEM = (
    "당신은 DOCX 에이전트의 Analyzer입니다. "
    "문서를 분석하고 다음 실행 모드를 결정합니다. "
    "내부적으로 판단하되 분석 과정은 출력하지 마세요. "
    "반드시 완전한 <analysis_result> XML만 출력하세요."
)


def analyze_request(
    request: str,
    context: DocxContext | None,
    model: str,
    run_dir: Path,
    feedback: str = "",
    attempt: int = 1,
) -> tuple[str, AnalysisResult]:
    prompt = build_prompt(request, context, feedback)
    raw = run_llm(model, ANALYZER_SYSTEM, prompt, response_prefix="<analysis_result>\n", num_predict=768)
    (run_dir / f"analysis_attempt{attempt}_raw.txt").write_text(raw, encoding="utf-8")
    try:
        xml_text, result = parse_analysis_result(raw)
    except Exception:
        retry_raw = run_llm(
            model,
            ANALYZER_SYSTEM,
            build_retry_prompt(request, context, feedback),
            response_prefix="<analysis_result>\n",
            num_predict=768,
        )
        (run_dir / f"analysis_attempt{attempt}_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
        xml_text, result = parse_analysis_result(retry_raw)
    (run_dir / f"analysis_result_attempt{attempt}.xml").write_text(xml_text, encoding="utf-8")
    return xml_text, result


def build_prompt(request: str, context: DocxContext | None, feedback: str) -> str:
    context_block = ""
    if context is not None:
        context_block = (
            f"열린 DOCX 구조와 내용 요약:\n{compact_context(context.xml, limit=7000)}\n\n"
        )
    feedback_block = f"이전 평가 피드백:\n{feedback}\n\n" if feedback else ""
    return (
        f"사용자 요청:\n{request}\n\n"
        f"{context_block}"
        f"{feedback_block}"
        "역할: 열린 문서가 있으면 그 문서를 분석하고, 사용자 요청을 어떤 실행 모드로 처리할지 판단하세요.\n"
        "모드는 세 가지 중 하나입니다.\n"
        "- edit_existing: 열린 DOCX 자체의 특정 내용, 표, 문단을 수정해야 하는 요청\n"
        "- create_from_source: 열린 DOCX를 읽기 전용 자료로 분석해 새로운 DOCX를 만들어야 하는 요청\n"
        "- create_new: 열린 DOCX가 없거나, 열린 문서와 무관하게 새 DOCX를 만들어야 하는 요청\n"
        "\"요약된 새로운 문서\", \"분석해서 새 문서\", \"바탕으로 보고서 작성\"은 create_from_source가 우선입니다.\n"
        "\"팀명 변경\", \"이 항목 수정\", \"문서의 목적 보강\"처럼 원본 문서를 고치는 요청은 edit_existing입니다.\n"
        "XML 본문 값에는 꺾쇠괄호 문자나 태그 예시를 쓰지 마세요.\n"
        "값을 비워두거나 점 세 개 같은 placeholder를 쓰지 마세요.\n"
        "반드시 아래 형식만 출력하세요.\n"
        "<analysis_result>\n"
        "  <recommended_mode>edit_existing|create_from_source|create_new</recommended_mode>\n"
        "  <document_type>문서 유형</document_type>\n"
        "  <topic>핵심 주제</topic>\n"
        "  <summary>문서와 요청의 핵심 요약</summary>\n"
        "  <preserve>유지할 요소</preserve>\n"
        "  <change>바꿀 요소</change>\n"
        "  <reason>모드 선택 이유</reason>\n"
        "</analysis_result>\n"
        "마지막 </analysis_result> 뒤에는 아무것도 출력하지 마세요."
    )


def build_retry_prompt(request: str, context: DocxContext | None, feedback: str) -> str:
    return (
        build_prompt(request, context, feedback)
        + "\nXML 문법을 엄격히 지켜 다시 출력하세요. 설명문 안에 < 또는 > 문자를 쓰지 마세요."
    )
