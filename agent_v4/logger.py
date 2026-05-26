import json
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from agent_v4.config import BASE_DIR


def make_run_dir(out_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = out_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(record: dict[str, Any], path: Path | None = None) -> None:
    if path is None:
        path = BASE_DIR / "training_logs" / "docx_agent_v4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        payload = normalize_training_record({"created_at": datetime.now().isoformat(timespec="seconds"), **record})
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def update_few_shot_examples(record: dict[str, Any], max_examples: int = 3) -> None:
    if not record.get("passed"):
        return
    if record.get("usable_for_training") is False:
        return
    agent_result_xml = record.get("agent_result_xml")
    if not agent_result_xml:
        return

    path = BASE_DIR / "training_logs" / "docx_agent_v4_few_shot_examples.xml"
    existing = read_existing_samples(path)
    sample = build_sample_xml(record)
    samples = [sample] + [item for item in existing if item != sample]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<few_shot_examples>\n" + "\n".join(samples[:max_examples]) + "\n</few_shot_examples>\n",
        encoding="utf-8",
    )


def read_existing_samples(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    samples = []
    start = 0
    while True:
        open_index = text.find("  <sample", start)
        if open_index < 0:
            break
        close_index = text.find("  </sample>", open_index)
        if close_index < 0:
            break
        close_index += len("  </sample>")
        samples.append(text[open_index:close_index])
        start = close_index
    return samples


def build_sample_xml(record: dict[str, Any]) -> str:
    request = escape(str(record.get("request", "")))
    agent_result_xml = escape(str(record.get("agent_result_xml", "")))
    return (
        f"  <sample mode=\"{escape(str(record.get('mode', 'edit')))}\" model=\"{escape(str(record.get('model', '')))}\">\n"
        f"    <request>{request}</request>\n"
        f"    <good_output>{agent_result_xml}</good_output>\n"
        "  </sample>"
    )


def normalize_training_record(record: dict[str, Any]) -> dict[str, Any]:
    if "usable_for_training" not in record:
        record["usable_for_training"] = bool(record.get("passed"))
    if "training_note" not in record:
        if record.get("usable_for_training"):
            record["training_note"] = "검증 통과 샘플입니다. few-shot 또는 후보 학습 데이터로 사용할 수 있습니다."
        else:
            record["training_note"] = "실패 또는 미검토 샘플입니다. 원인 검토 전 학습 데이터로 사용하지 마세요."
    return record
