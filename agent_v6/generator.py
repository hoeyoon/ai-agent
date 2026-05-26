from pathlib import Path

from agent_v6.context import DocxContext, compact_context
from agent_v6.llm import run_llm
from agent_v6.xml_protocol import parse_document_output


GENERATOR_SYSTEM = (
    "당신은 Writer Agent입니다. 분석 결과와 원본 문서 내용을 바탕으로 새 DOCX 문서 본문을 작성합니다. "
    "내부적으로 판단하되 분석 과정은 출력하지 마세요. "
    "반드시 완전한 <document> XML만 출력하세요."
)


def generate_document_xml(
    request: str,
    analysis_xml: str,
    model: str,
    run_dir: Path,
    feedback: str = "",
    attempt: int = 1,
    source_context: DocxContext | None = None,
    worker_context_xml: str = "",
):
    prompt = build_prompt(request, analysis_xml, feedback, source_context, worker_context_xml)
    raw = run_llm(model, GENERATOR_SYSTEM, prompt, response_prefix="<document>\n", num_predict=1536)
    (run_dir / f"writer_attempt{attempt}_raw.txt").write_text(raw, encoding="utf-8")
    try:
        xml_text, root = parse_document_output(raw)
    except Exception:
        retry_raw = run_llm(
            model,
            GENERATOR_SYSTEM,
            build_retry_prompt(request, analysis_xml, feedback, source_context, worker_context_xml),
            response_prefix="<document>\n",
            num_predict=1536,
        )
        (run_dir / f"writer_attempt{attempt}_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
        xml_text, root = parse_document_output(retry_raw)
    (run_dir / f"document_attempt{attempt}.xml").write_text(xml_text, encoding="utf-8")
    return xml_text, root


def build_prompt(
    request: str,
    analysis_xml: str,
    feedback: str,
    source_context: DocxContext | None,
    worker_context_xml: str = "",
) -> str:
    source_block = ""
    if source_context is not None:
        source_block = (
            f"원본 DOCX 분석 자료:\n{compact_context(source_context.xml, limit=9000)}\n\n"
        )
    feedback_block = f"이전 평가 피드백:\n{feedback}\n\n" if feedback else ""
    worker_block = f"Worker 분석 결과 XML:\n{worker_context_xml}\n\n" if worker_context_xml else ""
    return (
        f"사용자 요청:\n{request}\n\n"
        f"Analyzer 결과 XML:\n{analysis_xml}\n\n"
        f"{worker_block}"
        f"{source_block}"
        f"{feedback_block}"
        "새 DOCX 문서 내용을 작성하세요. 원본 DOCX가 제공되었다면 읽기 전용 자료로만 사용하고 원본 파일을 수정하지 않습니다.\n"
        "Worker 분석 결과가 있으면 그 결과를 우선 근거로 삼아 최종 문서를 구성하세요.\n"
        "요약 문서 요청이면 원본의 핵심 주제, 주요 근거, 결론을 압축하여 새 문서로 구성하세요.\n"
        "문서는 title, description, section, heading, paragraph 태그로만 구성하세요.\n"
        "본문에는 XML 태그 예시나 꺾쇠괄호 문자를 쓰지 마세요.\n"
        "placeholder나 점 세 개를 쓰지 말고 실제 문서 내용을 작성하세요.\n"
        "설명문, 마크다운, 코드블록을 출력하지 마세요.\n"
        "출력 형식:\n"
        "<document>\n"
        "  <title>문서 제목</title>\n"
        "  <description>문서 설명</description>\n"
        "  <section>\n"
        "    <heading>섹션 제목</heading>\n"
        "    <paragraph>본문 문단</paragraph>\n"
        "  </section>\n"
        "</document>\n"
        "마지막 </document> 뒤에는 아무것도 출력하지 마세요."
    )


def build_retry_prompt(
    request: str,
    analysis_xml: str,
    feedback: str,
    source_context: DocxContext | None,
    worker_context_xml: str = "",
) -> str:
    return (
        build_prompt(request, analysis_xml, feedback, source_context, worker_context_xml)
        + "\nXML 문법을 엄격히 지켜 다시 출력하세요. <document> 루트만 출력하세요."
    )

