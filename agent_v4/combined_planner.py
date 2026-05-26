from pathlib import Path
import json

from agent_v4.config import BASE_DIR
from agent_v4.context import DocxContext, compact_context
from agent_v4.llm import run_llm
from agent_v4.xml_protocol import Action, parse_agent_result_output


COMBINED_PLANNER_SYSTEM = (
    "당신은 DOCX 문서 에이전트의 Editor Agent입니다. "
    "내부적으로 판단하되 분석 과정은 출력하지 마세요. "
    "반드시 완전한 <agent_result> XML만 출력하세요."
)


def plan_agent_result(
    request: str,
    context: DocxContext,
    model: str,
    run_dir: Path,
    analysis_xml: str = "",
    feedback: str = "",
    attempt: int = 1,
) -> tuple[str, str, str, list[Action]]:
    prompt = build_prompt(request, context, feedback, analysis_xml)
    raw = run_llm(model, COMBINED_PLANNER_SYSTEM, prompt, response_prefix="<agent_result>\n", num_predict=1536)
    (run_dir / f"combined_planner_attempt{attempt}_raw.txt").write_text(raw, encoding="utf-8")
    try:
        result_xml, analysis_xml, actions_xml, actions = parse_agent_result_output(raw)
    except Exception:
        retry_raw = run_llm(
            model,
            COMBINED_PLANNER_SYSTEM,
            build_retry_prompt(request, context, feedback, analysis_xml),
            response_prefix="<agent_result>\n",
            num_predict=1536,
        )
        (run_dir / f"combined_planner_attempt{attempt}_retry_raw.txt").write_text(retry_raw, encoding="utf-8")
        result_xml, analysis_xml, actions_xml, actions = parse_agent_result_output(retry_raw)

    (run_dir / f"agent_result_attempt{attempt}.xml").write_text(result_xml, encoding="utf-8")
    (run_dir / f"request_analysis_attempt{attempt}.xml").write_text(analysis_xml, encoding="utf-8")
    (run_dir / f"actions_attempt{attempt}.xml").write_text(actions_xml, encoding="utf-8")
    return result_xml, analysis_xml, actions_xml, actions


def build_prompt(request: str, context: DocxContext, feedback: str, analysis_xml: str = "") -> str:
    feedback_block = f"이전 평가 피드백:\n{feedback}\n\n" if feedback else ""
    analysis_block = f"Analyzer 결과 XML:\n{analysis_xml}\n\n" if analysis_xml else ""
    examples_block = load_few_shot_examples(request)
    return (
        f"사용자 요청:\n{request}\n\n"
        f"{analysis_block}"
        f"DOCX 구조 XML:\n{compact_context(context.xml, limit=3600)}\n\n"
        f"{examples_block}"
        f"{feedback_block}"
        "당신은 Editor Agent입니다. 열린 DOCX를 직접 수정할 때만 사용할 실행 액션을 작성하세요.\n"
        "Analyzer 결과가 있으면 그 판단을 우선 참고하세요.\n"
        "Python은 파일 처리와 액션 실행만 담당하고, 판단은 당신이 합니다.\n"
        "사용자 요청의 의미를 먼저 분석하고, 기존 문서의 어떤 부분을 유지하거나 바꿀지 스스로 판단하세요.\n"
        "기존 라벨, 제목, 본문이 유지 대상인지 교체 대상인지는 사용자 요청과 DOCX 구조를 근거로 request_analysis에 명시하세요.\n"
        "request_analysis 안에는 액션 태그 이름을 꺾쇠괄호로 쓰지 말고 일반 텍스트로만 쓰세요.\n"
        "editable_candidates의 target은 실제 수정 가능한 위치입니다. 필요한 위치만 선택하세요.\n"
        "tables 안의 각 cell target도 수정 가능한 위치입니다.\n"
        "문단을 바꿔야 하면 actions 안에서 replace_paragraph 액션의 index와 value를 사용하세요.\n"
        "설명문 안에 액션 태그를 꺾쇠괄호로 쓰지 마세요. XML 문법이 깨집니다.\n"
        "값을 작성할 때는 각 candidate의 chars 속성을 참고해서 원본 길이와 비슷하거나 더 짧게 작성하세요.\n"
        "지원 액션은 fill_template, replace_table_value, replace_paragraph, replace_text뿐입니다.\n"
        "가장 권장되는 액션은 replace_table_value이며 target과 value 속성을 사용합니다.\n"
        "placeholder나 점 세 개를 값으로 쓰지 마세요.\n"
        "출력 형식:\n"
        "<agent_result>\n"
        "  <request_analysis>\n"
        "    <intent>수정 의도</intent>\n"
        "    <topic>핵심 주제</topic>\n"
        "    <preserve>유지할 요소</preserve>\n"
        "    <change>변경할 요소</change>\n"
        "  </request_analysis>\n"
        "  <actions>\n"
        "    <replace_table_value target=\"table[0] row[0] cell[1]\" value=\"새 값\"/>\n"
        "  </actions>\n"
        "</agent_result>\n"
        "예시 문구나 기존 문서 내용을 복사하지 말고 현재 사용자 요청과 DOCX 구조에 맞게 작성하세요.\n"
        "반드시 <agent_result> XML만 출력하고 마지막 </agent_result> 뒤에는 아무것도 출력하지 마세요."
    )


def build_retry_prompt(request: str, context: DocxContext, feedback: str, analysis_xml: str = "") -> str:
    return (
        build_prompt(request, context, feedback, analysis_xml)
        + "\nXML 문법을 엄격히 지켜 다시 출력하세요. 설명문 안에 < 또는 > 문자를 쓰지 마세요."
    )


def load_few_shot_examples(request: str = "", limit: int = 1) -> str:
    examples = select_training_examples(request, success_limit=limit, failure_limit=1)
    if not examples:
        return ""

    success = [item for item in examples if item["kind"] == "success"]
    failure = [item for item in examples if item["kind"] == "failure"]
    parts = ["아래는 training logs에서 가져온 참고 예시입니다. 구조와 교훈만 참고하고 내용은 복사하지 마세요."]
    if success:
        parts.append("<few_shot_success_examples>")
        parts.extend(item["xml"] for item in success)
        parts.append("</few_shot_success_examples>")
    if failure:
        parts.append("<few_shot_failure_examples>")
        parts.extend(item["xml"] for item in failure)
        parts.append("</few_shot_failure_examples>")
    return "\n".join(parts) + "\n\n"


def select_training_examples(request: str, success_limit: int = 1, failure_limit: int = 1) -> list[dict[str, str]]:
    path = BASE_DIR / "training_logs" / "docx_agent_v4.jsonl"
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    successes = [
        (similarity_score(request, record.get("request", "")), record)
        for record in records
        if record.get("usable_for_training") and record.get("agent_result_xml")
    ]
    failures = [
        (similarity_score(request, record.get("request", "")), record)
        for record in records
        if record.get("human_reviewed") and record.get("usable_for_training") is False
    ]
    successes = [record for score, record in sorted(successes, key=lambda item: item[0], reverse=True) if score >= 3]
    failures = [record for score, record in sorted(failures, key=lambda item: item[0], reverse=True) if score >= 3]

    examples = []
    for record in successes[:success_limit]:
        examples.append({"kind": "success", "xml": build_success_example(record)})
    for record in failures[:failure_limit]:
        examples.append({"kind": "failure", "xml": build_failure_example(record)})
    return examples


def build_success_example(record: dict) -> str:
    return (
        "<sample>\n"
        f"  <request>{escape_xml(record.get('request', ''))}</request>\n"
        f"  <good_output><![CDATA[{safe_cdata(record.get('agent_result_xml', ''))}]]></good_output>\n"
        "</sample>"
    )


def build_failure_example(record: dict) -> str:
    return (
        "<sample>\n"
        f"  <request>{escape_xml(record.get('request', ''))}</request>\n"
        f"  <bad_summary><![CDATA[{safe_cdata(record.get('reason', ''))}]]></bad_summary>\n"
        f"  <lesson><![CDATA[{safe_cdata(record.get('feedback') or record.get('training_note', ''))}]]></lesson>\n"
        "</sample>"
    )


def similarity_score(left: str, right: str) -> int:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    return len(left_tokens & right_tokens)


def tokenize(text: str) -> set[str]:
    compact = "".join(text.lower().split())
    tokens = set()
    for token in ("표", "형식", "구조", "문서", "기획서", "설명", "수정", "생성", "docx", "ai", "에이전트"):
        if token in compact:
            tokens.add(token)
    return tokens


def escape_xml(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def safe_cdata(value: str) -> str:
    return str(value).replace("]]>", "]]]]><![CDATA[>")
