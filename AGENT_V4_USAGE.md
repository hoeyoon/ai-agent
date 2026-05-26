# DOCX AI Agent v4

v4는 역할 분리형 DOCX 에이전트입니다.

```text
LLM Analyzer
  -> edit_existing이면 LLM Editor -> Python DOCX Tool -> LLM Evaluator
  -> create_from_source이면 LLM Orchestrator -> LLM Workers -> LLM Writer -> Python DOCX Writer -> LLM Evaluator
  -> create_new이면 LLM Writer -> Python DOCX Writer -> LLM Evaluator
```

q4와 q8은 같은 기능을 수행하는 모델 선택지입니다.

- q4: 더 가볍고 빠른 기본 선택
- q8: 같은 기능을 더 큰 양자화 모델로 실행

## 대화형 실행

```powershell
python -m agent_v4.main --chat --model q4
```

16GB 메모리 환경에서는 기본값을 권장합니다.

```powershell
python -m agent_v4.main --chat --model q4 --worker-tasks 3 --parallel-workers 1
```

## 열린 문서 사용

```text
agent_v4> /open sample.docx
```

`/open` 문서는 항상 수정 대상이 아닙니다. Analyzer가 사용자 요청을 보고 아래 중 하나로 판단합니다.

- `edit_existing`: 열린 DOCX 자체를 수정
- `create_from_source`: 열린 DOCX를 분석 자료로 사용해 새 DOCX 생성
- `create_new`: 열린 문서 없이 새 DOCX 생성

예:

```text
agent_v4> 팀명을 "docx 에이전트"로 바꿔줘
```

위 요청은 보통 `edit_existing`입니다.

```text
agent_v4> 이 문서의 내용을 분석해서 요약된 새로운 문서로 만들어줘
```

위 요청은 보통 `create_from_source`입니다.

## 단발 실행

```powershell
python -m agent_v4.main --model q4 --input ".\sample.docx" "이 문서의 내용을 분석해서 요약된 새로운 문서로 만들어줘"
```

결과는 기본적으로 `C:\models\runs_v4` 아래에 저장됩니다.

## Agentic 패턴 반영

- 프롬프트 체인: Analyzer 결과와 Evaluator feedback이 다음 단계 입력으로 전달됩니다.
- 라우팅: Analyzer가 `edit_existing`, `create_from_source`, `create_new` 중 실행 모드를 선택합니다.
- 병렬 처리: `create_from_source`에서 Worker 작업을 `--parallel-workers` 값만큼 병렬 실행할 수 있습니다.
- 오케스트레이터-워커: Orchestrator가 하위 작업을 만들고 Workers가 분석 결과를 생성한 뒤 Writer가 취합합니다.
- 평가 및 최적화: Evaluator가 결과를 평가하고 실패 시 feedback으로 재시도합니다.

## Worker 옵션

```powershell
python -m agent_v4.main --model q4 --worker-tasks 3 --parallel-workers 1 --input ".\sample.docx" "이 문서의 내용을 분석해서 요약된 새로운 문서로 만들어줘"
```

- `--worker-tasks`: Orchestrator가 만들 하위 작업 수입니다.
- `--parallel-workers`: Worker 병렬 실행 수입니다.

주의: 로컬 16GB 환경에서는 `--parallel-workers 1`이 가장 안정적입니다. 값을 2 이상으로 올리면 병렬 패턴은 강화되지만 메모리 사용량과 Ollama 대기 시간이 늘 수 있습니다.

## 로그

v4 로그는 v3와 분리됩니다.

```text
C:\models\training_logs\docx_agent_v4.jsonl
C:\models\training_logs\docx_agent_v4_few_shot_examples.xml
```
