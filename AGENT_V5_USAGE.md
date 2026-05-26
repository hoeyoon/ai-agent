# DOCX AI Agent v5

v5는 Codex/Claude Code 방식에 더 가까운 tool-loop 에이전트입니다.

```text
사용자 요청
  -> LLM이 다음 tool_call XML 생성
  -> Python이 도구 실행
  -> observation XML 저장
  -> observation을 보고 LLM이 다음 tool_call 생성
  -> evaluate_result
  -> finish
```

## 실행

```powershell
python -m agent_v5.main --chat --model q4
```

단발 실행:

```powershell
python -m agent_v5.main --model q4 --input ".\sample.docx" "이 문서의 내용을 분석해서 요약된 새로운 문서로 만들어줘"
```

## Chat 명령

```text
/open <파일.docx>
/new
/status
/exit
```

## Tool 목록

- `read_docx_context`: DOCX의 `word/document.xml`을 읽고 문단/표/셀 context를 생성합니다.
- `analyze_docx`: 읽은 context를 observation으로 요약합니다.
- `edit_docx`: LLM이 만든 actions XML을 실행해 기존 DOCX를 수정합니다.
- `create_docx`: LLM이 만든 document XML로 새 DOCX를 생성합니다.
- `evaluate_result`: 마지막 결과 문서를 LLM Evaluator로 검증합니다.
- `finish`: 작업을 종료합니다.

## v4와 차이

v4는 역할별 파이프라인입니다.

```text
Analyzer -> Editor/Writer -> Evaluator
```

v5는 매 단계마다 LLM이 다음 도구를 직접 선택합니다.

```text
tool_call -> observation -> tool_call -> observation
```

따라서 Codex/Claude Code처럼 중간 결과를 보고 다음 행동을 바꾸는 구조에 더 가깝습니다.

## 권장 설정

LG gram pro 2024, 16GB RAM 환경에서는 q4를 기본으로 쓰는 것을 권장합니다.

```powershell
python -m agent_v5.main --chat --model q4 --max-steps 8
```

q8도 같은 기능을 수행하지만 더 무겁습니다.

