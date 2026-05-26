import argparse
import sys
from pathlib import Path

from agent_v4.config import BASE_DIR
from agent_v4.human_review import apply_human_correction, apply_human_review
from agent_v4.llm import create_models, shutdown_ollama
from agent_v4.loop import AgentError, run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="역할 분리형 DOCX AI 에이전트 v4")
    parser.add_argument("request", nargs="*", help="문서 생성/수정 요청")
    parser.add_argument("--input", "-i", type=Path, help="열어서 분석하거나 수정할 DOCX 파일")
    parser.add_argument("--model", choices=["q4", "q8"], default="q4")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "runs_v4")
    parser.add_argument("--prefix", default="agent_v4")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--worker-tasks", type=int, default=3, help="create_from_source에서 생성할 Worker 하위 작업 수")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Worker 병렬 실행 수. 16GB 환경에서는 1 권장")
    parser.add_argument("--chat", action="store_true", help="대화형 모드")
    parser.add_argument("--init-models", action="store_true", help="Modelfile로 Ollama 모델 등록")
    return parser.parse_args()


def run_once(args: argparse.Namespace, request: str, input_path: Path | None) -> tuple[Path, Path]:
    attempts = max(1, args.max_attempts)
    run_dir, result_path, mode, changed = run_agent(
        request,
        input_path,
        args.out_dir,
        args.prefix,
        args.model,
        attempts,
        parallel_workers=max(1, args.parallel_workers),
        worker_tasks=max(1, args.worker_tasks),
    )
    print(f"실행 폴더: {run_dir}")
    print(f"선택된 모드: {mode}")
    if mode == "edit_existing":
        print(f"DOCX 수정 완료: {result_path}")
        print(f"수정된 항목 수: {changed}")
    else:
        print(f"DOCX 생성 완료: {result_path}")
    return result_path, run_dir


def run_chat(args: argparse.Namespace) -> int:
    current_docx = args.input
    last_run_dir: Path | None = None
    print("DOCX AI 에이전트 v4 대화형 모드")
    print("명령: /open <파일.docx>, /new, /status, /review pass|fail <이유>, /correct <agent_result.xml|document.xml>, /exit")
    print("/open 문서는 Analyzer가 요청에 따라 수정 대상 또는 새 문서 생성용 분석 소스로 판단합니다.")
    if current_docx:
        print(f"현재 문서: {current_docx}")

    while True:
        try:
            user_input = input("\nagent_v4> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            shutdown_ollama()
            return 0
        if not user_input:
            continue
        if user_input in ("/exit", "exit", "quit", "/quit"):
            print("종료합니다.")
            shutdown_ollama()
            return 0
        if user_input.startswith("/open "):
            current_docx = Path(user_input[6:].strip().strip("\"'"))
            print(f"현재 문서: {current_docx}")
            continue
        if user_input == "/new":
            current_docx = None
            print("현재 문서를 비웠습니다. 다음 요청은 열린 문서 없이 새 DOCX 생성 모드로 판단됩니다.")
            continue
        if user_input == "/status":
            print(f"현재 문서: {current_docx if current_docx else '(없음, 생성 모드)'}")
            print(f"마지막 실행 폴더: {last_run_dir if last_run_dir else '(없음)'}")
            print(f"모델: {args.model}")
            print(f"최대 시도 횟수: {max(1, args.max_attempts)}")
            print(f"Worker 작업 수: {max(1, args.worker_tasks)}")
            print(f"Worker 병렬 수: {max(1, args.parallel_workers)}")
            print("구조: LLM Analyzer -> Orchestrator/Workers -> Editor/Writer -> Python Tools -> LLM Evaluator -> Human Review")
            continue
        if user_input.startswith("/review "):
            if last_run_dir is None:
                print("리뷰할 마지막 실행 폴더가 없습니다.")
                continue
            handle_review_command(user_input, last_run_dir)
            continue
        if user_input.startswith("/correct "):
            if last_run_dir is None:
                print("교정할 마지막 실행 폴더가 없습니다.")
                continue
            handle_correct_command(user_input, last_run_dir)
            continue
        try:
            current_docx, last_run_dir = run_once(args, user_input, current_docx)
            print(f"다음 작업 대상: {current_docx}")
            print("결과를 확인한 뒤 /review pass 또는 /review fail <이유>를 입력하면 training log에 사람 검토가 기록됩니다.")
        except Exception as exc:
            print(f"오류: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        if args.init_models:
            create_models()
            print("Ollama 모델 등록 완료")
            return 0
        if args.chat:
            return run_chat(args)
        request = " ".join(args.request).strip()
        if not request:
            raise ValueError("요청 문장을 입력하세요.")
        run_once(args, request, args.input)
        return 0
    except (AgentError, Exception) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.chat:
            shutdown_ollama()


def handle_review_command(command: str, last_run_dir: Path) -> None:
    parts = command.split(maxsplit=2)
    if len(parts) < 2 or parts[1] not in {"pass", "fail"}:
        print("사용법: /review pass [메모] 또는 /review fail <이유>")
        return
    passed = parts[1] == "pass"
    feedback = parts[2].strip() if len(parts) >= 3 else ""
    if not passed and not feedback:
        print("실패 리뷰에는 이유가 필요합니다. 예: /review fail 표 형식만 유지해야 하는데 라벨이 원본 그대로 남음")
        return
    error_type = "human_review_failed" if not passed else ""
    record = apply_human_review(last_run_dir, passed, feedback, error_type)
    print(f"사람 검토 저장 완료: human_passed={record.get('human_passed')}, usable_for_training={record.get('usable_for_training')}")


def handle_correct_command(command: str, last_run_dir: Path) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) < 2:
        print("사용법: /correct <agent_result.xml 또는 document.xml 경로>")
        return
    path = Path(parts[1].strip().strip("\"'"))
    if not path.exists():
        print(f"교정 XML 파일을 찾지 못했습니다: {path}")
        return
    good_xml = path.read_text(encoding="utf-8")
    record = apply_human_correction(last_run_dir, good_xml, "사람이 교정한 정답 XML을 등록했습니다.")
    print(f"교정 저장 완료: usable_for_training={record.get('usable_for_training')}")


if __name__ == "__main__":
    raise SystemExit(main())
