import json
from pathlib import Path
from typing import Any

from agent_v4.config import BASE_DIR
from agent_v4.logger import update_few_shot_examples
from agent_v4.xml_protocol import extract_required_xml


TRAINING_LOG = BASE_DIR / "training_logs" / "docx_agent_v4.jsonl"
FEW_SHOT = BASE_DIR / "training_logs" / "docx_agent_v4_few_shot_examples.xml"


def apply_human_review(
    run_dir: Path,
    passed: bool,
    feedback: str = "",
    error_type: str = "",
) -> dict[str, Any]:
    rows = load_rows()
    changed = False
    for row in rows:
        if same_run(row, run_dir):
            row["human_reviewed"] = True
            row["human_passed"] = passed
            row["passed"] = passed
            row["usable_for_training"] = passed
            row["human_feedback"] = feedback
            if feedback:
                row["feedback"] = feedback
            if error_type:
                row["error_type"] = error_type
            row["training_note"] = (
                "사람이 검토해 통과시킨 샘플입니다. few-shot 또는 후보 학습 데이터로 사용할 수 있습니다."
                if passed
                else "사람이 검토해 실패 처리한 샘플입니다. good_agent_result_xml이 추가되기 전에는 학습/few-shot에 사용하지 마세요."
            )
            changed = True
            break
    if not changed:
        raise ValueError(f"training log에서 실행 폴더를 찾지 못했습니다: {run_dir}")
    save_rows(rows)
    rebuild_few_shot_examples(rows)
    return row


def apply_human_correction(run_dir: Path, good_agent_result_xml: str, feedback: str = "") -> dict[str, Any]:
    rows = load_rows()
    changed = False
    for row in rows:
        if same_run(row, run_dir):
            root_name = detect_root_name(good_agent_result_xml)
            row["human_reviewed"] = True
            row["human_passed"] = True
            row["usable_for_training"] = True
            if root_name == "document":
                row["good_document_xml"] = good_agent_result_xml
                row["document_xml"] = good_agent_result_xml
            else:
                row["good_agent_result_xml"] = good_agent_result_xml
                row["agent_result_xml"] = good_agent_result_xml
            if feedback:
                row["human_feedback"] = feedback
                row["feedback"] = feedback
            row["training_note"] = (
                "사람이 교정한 정답 document_xml이 포함된 고품질 Writer 샘플입니다."
                if root_name == "document"
                else "사람이 교정한 정답 agent_result_xml이 포함된 고품질 Editor 샘플입니다."
            )
            changed = True
            break
    if not changed:
        raise ValueError(f"training log에서 실행 폴더를 찾지 못했습니다: {run_dir}")
    save_rows(rows)
    rebuild_few_shot_examples(rows)
    return row


def load_rows() -> list[dict[str, Any]]:
    if not TRAINING_LOG.exists():
        return []
    rows = []
    for line in TRAINING_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def save_rows(rows: list[dict[str, Any]]) -> None:
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def rebuild_few_shot_examples(rows: list[dict[str, Any]]) -> None:
    if FEW_SHOT.exists():
        FEW_SHOT.unlink()
    for row in rows:
        if row.get("usable_for_training") and row.get("agent_result_xml"):
            update_few_shot_examples(row)


def same_run(row: dict[str, Any], run_dir: Path) -> bool:
    left = str(row.get("run_dir", "")).replace("\\", "/").rstrip("/")
    right = str(run_dir).replace("\\", "/").rstrip("/")
    return left == right or left.endswith("/" + run_dir.name)


def detect_root_name(xml_text: str) -> str:
    for root_name in ("agent_result", "document"):
        try:
            extract_required_xml(xml_text, root_name)
            return root_name
        except Exception:
            continue
    raise ValueError("교정 XML은 <agent_result> 또는 <document> 루트여야 합니다.")
