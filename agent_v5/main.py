import argparse
import sys
from pathlib import Path

from agent_v5.agent_loop import AgentLoopError, run_tool_loop
from agent_v5.config import BASE_DIR
from agent_v5.llm import create_models, shutdown_ollama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tool-loop DOCX AI 에이전트 v5")
    parser.add_argument("request", nargs="*", help="문서 생성/수정 요청")
    parser.add_argument("--input", "-i", type=Path, help="열어서 분석하거나 수정할 DOCX 파일")
    parser.add_argument("--model", choices=["q4", "q8"], default="q4")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "runs_v5")
    parser.add_argument("--prefix", default="agent_v5")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--chat", action="store_true", help="대화형 모드")
    parser.add_argument("--init-models", action="store_true", help="Modelfile로 Ollama 모델 등록")
    return parser.parse_args()


def run_once(args: argparse.Namespace, request: str, input_path: Path | None) -> tuple[Path | None, Path]:
    run_dir, result_path, final_observation = run_tool_loop(
        request=request,
        input_path=input_path,
        out_dir=args.out_dir,
        prefix=args.prefix,
        model=args.model,
        max_steps=max(1, args.max_steps),
    )
    print(f"실행 폴더: {run_dir}")
    if result_path:
        print(f"결과 DOCX: {result_path}")
    else:
        print("결과 DOCX: (없음)")
    (run_dir / "final_observation.xml").write_text(final_observation, encoding="utf-8")
    return result_path, run_dir


def run_chat(args: argparse.Namespace) -> int:
    current_docx = args.input
    last_run_dir: Path | None = None
    print("DOCX AI 에이전트 v5 대화형 모드")
    print("명령: /open <파일.docx>, /new, /status, /exit")
    print("구조: LLM tool_call -> Python tool 실행 -> observation -> LLM next tool_call")
    if current_docx:
        print(f"현재 문서: {current_docx}")

    while True:
        try:
            user_input = input("\nagent_v5> ").strip()
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
            print("현재 문서를 비웠습니다.")
            continue
        if user_input == "/status":
            print(f"현재 문서: {current_docx if current_docx else '(없음)'}")
            print(f"마지막 실행 폴더: {last_run_dir if last_run_dir else '(없음)'}")
            print(f"모델: {args.model}")
            print(f"최대 tool step: {max(1, args.max_steps)}")
            continue
        try:
            result_path, last_run_dir = run_once(args, user_input, current_docx)
            if result_path:
                current_docx = result_path
                print(f"다음 작업 대상: {current_docx}")
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
    except (AgentLoopError, Exception) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.chat:
            shutdown_ollama()


if __name__ == "__main__":
    raise SystemExit(main())
