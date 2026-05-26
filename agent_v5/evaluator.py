from pathlib import Path

from agent_v5.context import compact_context
from agent_v5.llm import run_llm
from agent_v5.xml_protocol import Evaluation, parse_evaluation_output


EVALUATOR_SYSTEM = (
    "당신은 DOCX 에이전트 결과를 검증하는 평가자입니다. "
    "내부적으로 판단하되 분석 과정은 출력하지 마세요. "
    "반드시 완전한 <evaluation> XML만 출력하세요."
)


def evaluate_result(
    request: str,
    request_analysis_xml: str,
    actions_or_document_xml: str,
    before_context_xml: str,
    after_context_xml: str,
    diff_context: str,
    model: str,
    run_dir: Path,
    attempt: int,
) -> tuple[str, Evaluation]:
    prompt = (
        f"사용자 요청:\n{request}\n\n"
        f"요청 분석 XML:\n{compact_context(request_analysis_xml, limit=700)}\n\n"
        f"실행 XML:\n{compact_context(actions_or_document_xml, limit=900)}\n\n"
        f"변경 증거 XML:\n{compact_context(diff_context, limit=3600)}\n\n"
        "위 증거만 근거로 사용자 요청이 충족되었는지 판단하세요.\n"
        "Python은 성공/실패를 판단하지 않고 changed/unchanged/missing 증거만 제공합니다.\n"
        "edit_existing 결과라면 사용자가 유지하라고 한 것은 유지되어야 하고, 바꾸라고 한 것은 변경되어야 합니다.\n"
        "edit_existing에서 요청 분석이나 실행 XML이 바꾸겠다고 한 문단/표 셀이 unchanged에 남아 있으면 실패입니다.\n"
        "create_from_source 또는 create_new 결과라면 새 문서의 제목, 설명, 섹션, 본문이 사용자 요청과 분석 결과를 충분히 반영했는지 판단하세요.\n"
        "새 문서 생성 결과에서 원본 문서가 unchanged로 남아 있지 않은 것은 정상입니다.\n"
        "사용자 요청의 새 주제와 맞지 않는 원본 제목, 라벨, 본문이 결과 문서에 남아 있으면 실패입니다.\n"
        "changed 증거에 없는 변경이 있었다고 추측하지 마세요.\n"
        "확실하지 않으면 passed=\"false\"로 판단하세요.\n"
        "반드시 <evaluation passed=\"true\"> 또는 <evaluation passed=\"false\"> XML만 출력해줘.\n"
        "<reason>은 2문장 이하로 짧게 작성하고, <feedback>은 실패 시 수정 지시만 작성해줘.\n"
        "마지막 닫는 태그인 </evaluation> 뒤에는 아무것도 출력하지 마."
    )
    run_dir.joinpath(f"evaluation_prompt_attempt{attempt}.txt").write_text(prompt, encoding="utf-8")
    raw = run_llm(model, EVALUATOR_SYSTEM, prompt, response_prefix='<evaluation passed="', num_predict=256)
    (run_dir / f"evaluation_attempt{attempt}_raw.txt").write_text(raw, encoding="utf-8")
    try:
        xml_text, evaluation = parse_evaluation_output(raw)
    except Exception:
        retry_raw = run_llm(
            model,
            EVALUATOR_SYSTEM,
            build_retry_prompt(request, request_analysis_xml, actions_or_document_xml, diff_context),
            response_prefix='<evaluation passed="',
            num_predict=256,
        )
        (run_dir / f"evaluation_attempt{attempt}_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
        xml_text, evaluation = parse_evaluation_output(retry_raw)
    (run_dir / f"evaluation_attempt{attempt}.xml").write_text(xml_text, encoding="utf-8")
    return xml_text, evaluation


def build_retry_prompt(request: str, request_analysis_xml: str, actions_or_document_xml: str, diff_context: str) -> str:
    return (
        f"사용자 요청:\n{request}\n\n"
        f"요청 분석 XML:\n{compact_context(request_analysis_xml, limit=500)}\n\n"
        f"실행 XML:\n{compact_context(actions_or_document_xml, limit=700)}\n\n"
        f"변경 증거 XML:\n{compact_context(diff_context, limit=2200)}\n\n"
        "변경 증거만 근거로 평가 결과를 완전한 <evaluation> XML만으로 다시 출력하세요. "
        "확실하지 않으면 passed=\"false\"로 판단하세요."
    )
