# Phase 1: Data Augmentation & Teacher Dataset Generation [COMPLETED]

Phase 1에서 정교한 합성 데이터 생성 및 GPT-4o(Teacher) 파이프라인 가동을 완료하여, 로컬 LLM 학습을 위한 고품질 데이터셋을 확보했습니다.

## 주요 성과
- **스키마 준수 합성 데이터**: `050425.csv` 규격에 최적화된 30개의 구매 시나리오 생성 완료.
- **고정밀 히스로리 분리**: '공급사 이력'과 '제품 이력'을 엄격히 분리하여 합성 시나리오에 주입함으로써 데이터의 순도를 높였습니다.
- **강건성(Robustness) 강화**: 구체적으로 다음과 같은 실전형 에지 케이스를 포함하여 모델이 단순 패턴 매칭을 넘어 비판적 사고를 하도록 유도했습니다.
    - **데이터 결손 (Data Gaps)**: 특정 아이템의 `RiskLevel`이나 `WksToOOS`가 `null`인 상황에서의 추론 (예: Scenario 10).
    - **신뢰의 역설 (Reliability Paradox)**: 기본 정보는 '우수'하나 최근 히스토리에서 '중대한 결함/지연'이 발견되는 상황 (예: Scenario 3).
    - **외부 요인 반영 (External Factors)**: 운송 파업, 온라인 부정 리뷰, 갑작스러운 설계 변경 등 비구조적 데이터 반영 (예: Scenario 5, 6).
    - **재고-수요 불일치**: 재고량은 절대적으로 많지만 소진 속도가 너무 빨라 위기 상황인 경우 (예: Scenario 9).
- **Teacher-Student 데이터셋 확보**: GPT-4o가 작성한 분석 보고서, PR, 이메일이 포함된 `teacher_dataset_20260225.jsonl` (약 250KB, 30개 시나리오) 생성 완료.
- **CoT(Chain of Thought) 적용**: Analysis Agent가 분석 전 팩트 체크를 선행하도록 유도하여 할루시네이션을 획기적으로 줄였습니다.
- **고품질 데이터 선별**: 30개 시나리오 중 **7.5점 이상인 사례 15건(8.5점 3건 포함)**을 확보하여 SFT를 위한 최적의 '골드 스탠다드'를 마련했습니다.
- **영문 평가 보고서(Evaluation Audit) 포함**: 데이터 충실도와 논리적 추론을 평가한 영문 보고서를 포함했습니다.

## 데이터 고도화 시행착오 및 해결 과정
단순 데이터 생성을 넘어, 실제 '전문가급' 성능을 확보하기 위해 겪은 시행착오와 해결책을 기록합니다.

1. **상충 문제 (Strictness vs. Reasoning)**: 초기 Evaluation Agent에게 '할루시네이션(환각) 방지'를 강하게 지시하자, 전문가의 '합리적 추론'까지도 '없는 정보'로 간주하여 4.5~5.5점의 저득점을 남발하는 현상이 발생했습니다.
2. **해결책 1: CoT(Chain of Thought) 도입**: Analysis Agent가 답변 전 **[FACT CHECK] -> [GAP ANALYSIS] -> [LOGICAL INFERENCE]** 단계를 거치도록 프롬프트를 개조하여, 추론의 근거를 스스로 명시하게 했습니다.
3. **해결책 2: Balanced Evaluation**: Evaluator의 기준을 **'단순 날조(Fabrication)'**와 **'근거 있는 추론(Nuanced Reasoning)'**으로 이원화하여, 데이터 부재 상황에서의 정직성과 풍부한 추론 능력을 동시에 평가하도록 정교화했습니다.
4. **결과**: 이러한 상호 교정 루프를 통해 8.5점 이상의 'Gold Standard' 데이터를 확보할 수 있었습니다.

## 전략적 선택 및 의사결정 기록
데이터 구축 과정에서 내린 핵심적인 기술적 판단과 그 근거를 기록합니다.

1. **SFT (Supervised Fine-Tuning) 우선 채택**:
    - **이유**: 현재 목표는 모델이 복잡한 JSON 스키마를 100% 준수하고, 특정 비즈니스 톤을 습득하는 것입니다. DPO(Direct Preference Optimization)와 같은 선호도 학습보다는, 완벽한 정답지(Gold Standard)를 보여주며 '모방 학습'을 시키는 SFT가 초기 정밀도 확보에 더 유리하다고 판단했습니다.
2. **양(Quantity)보다 질(Quality) 전략**:
    - **이유**: 1,000개의 노이즈 섞인 데이터보다, 전문가급 사고(CoT)가 반영된 15~30개의 정예 데이터를 학습시키는 것이 모델의 '추론 능력' 전이에 더 효과적입니다. 이를 위해 30개의 시나리오를 엄선하고, 그중에서도 고득점 사례만 필터링하여 사용하기로 했습니다.
3. **합성 히스토리 주입(History Override)**:
    - **이유**: 단순 시뮬레이션을 위해 실제 DB를 구축하기보다, 검증하고 싶은 '에지 케이스'를 히스토리 형태의 텍스트로 직접 주입하여 학습 데이터의 다양성과 강건성을 효율적으로 확보했습니다.

## 기술 스택: LangGraph & LangSmith 선정 이유
단순 선형 파이프라인을 넘어 복잡한 구매 로직을 안정적으로 처리하기 위해 도입된 핵심 도구들입니다.

### 1. LangGraph: 임의 흐름 제어와 보안 가드 (Validator)
- **자유로운 흐름 제어**: LangGraph는 단순 선형 구조를 넘어 조건문(`if`), 반복문(`while/loop`), 병렬 처리 등 개발자가 원하는 **모든 임의의 흐름**을 DAG(Directed Acyclic Graph)나 순환 그래프로 구현할 수 있게 해줍니다.
- **DLP(Data Loss Prevention) 가드**: 본 프로젝트에서는 특히 **보안(Validator)**에 이 유연함을 사용했습니다.
    - **검증 로직**: `WksToOOS`, `Risk Level`, `Internal Analysis` 등 협력사에 노출되어서는 안 되는 **내부 전문 용어 및 수치**가 이메일에 포함되었는지 검사합니다.
    - **하이브리드 검증**: 단순 키워드 매칭(Heuristic)뿐만 아니라, **LLM이 직접 감사(Audit)**를 수행하여 문맥상 기밀 유출이 있는지 판단합니다.

### 2. LangSmith: 기록을 넘어선 'LLM 실험실'
- **단순 기록(Tracing) 그 이상**: LangSmith는 단순히 '무슨 일이 일어났나'를 기록하는 로그 저장소를 넘어, LLM 애플리케이션의 **전체 수명 주기를 관리**합니다.
- **디버깅 (Debug)**: 각 노드에서 에이전트가 어떤 도구를 호출했고, 토큰을 얼마나 썼으며, 왜 그런 답변을 했는지 파이프라인의 **'블랙박스'**를 열어보게 해줍니다.
- **데이터셋 및 테스트 (Testing)**: 이번 프로젝트에서 추출한 'Gold Standard' 데이터셋처럼, 좋은 사례들을 한곳에 모아 **회귀 테스트(Regression Test)**를 돌리거나 모델 성능을 정량적으로 비교(A/B Test)할 때 필수적입니다.
- **버그 해결 및 안정화**: 파이프라인 엔진(`services/agents.py`)의 변수 스코핑 오류를 해결하여 합성 데이터 주입 로직의 안정성을 확보했습니다.

## 생성된 데이터 예시
```json
{
  "instruction": "Analyze the purchasing data and supplier/item history...",
  "input": {
    "inventory": [...],
    "supplier_history": "Historically, SupplierX consistent...",
    "item_history": "WidgetPro experienced unexpected demand swings..."
  },
  "output": {
    "analysis": { ... },
    "report": "# Purchasing Analysis Report...",
    "pr": "Purchase Request for SupplierX...",
    "email": "Subject: Purchase Inquiry for SupplierX..."
  }
}
```

## AI Engineer의 핵심 역량과 프로젝트의 지향점
본 프로젝트를 통해 정의한 현대적 AI 엔지니어링의 핵심 가치입니다.

1. **로컬 모델 최적화 (Fine-tuning)**: 범용 API에 의존하는 것을 넘어, 도메인 특화 데이터를 활용해 Llama-3/Mistral 등 로컬 모델을 직접 훈련시켜 보안성과 성능을 동시에 확보합니다.
2. **지연 시간 및 효율성 관리 (Latency & Throughput)**: 단순히 '작동하는 것'을 넘어, 비즈니스 환경에 적합한 처리 속도를 내기 위해 가벼운 모델을 정교하게 튜닝합니다.
3. **평가 기반 자동 개선 (Eval-driven Iteration)**: 평가 점수(Evaluation Score)를 단순 로그가 아닌 **'모델 개선의 피드백 루프'**로 활용합니다. 낮은 점수의 사례를 분석하여 학습 데이터로 재투입하는 과정이 AI 엔지니어링의 정수입니다.

이러한 접근법을 통해 우리는 단순한 '프롬프트 유저'에서 **'독자적인 AI 엔진 구축자'**로 진화하고 있습니다.
- **Fine-tuning 환경 구축**: 확보된 데이터셋(`teacher_dataset_20260225.jsonl`)을 사용하여 Unsloth 또는 QLoRA 기반 미세 조정 수행 예정.
- **모델 성능 검증**: 학습된 모델의 성능을 GPT-4o 결과와 비교하여 Side-by-Side 평가 진행 예정.

---

# Phase 1.5: Code Quality, Git Strategy & Prompt Engineering

Phase 2(SFT 학습) 진입 전, 코드 품질 표준화와 프롬프트 보강을 수행하여 포트폴리오 완성도와 학습 데이터 품질을 동시에 높인 단계입니다.

## Git 전략 수립

포트폴리오 GitHub를 체계적으로 관리하기 위한 브랜치 전략을 수립하고 실행했습니다.

| 브랜치 | 역할 |
|:---|:---|
| `main` | 최신 개선 버전 (채용 담당자가 보는 곳) |
| `legacy/gpt4o` | 기존 GPT-4o 전용 버전 보존 |
| `feature/local-model-pipeline` | Phase 1 작업 PR 브랜치 |

**커밋 전략**: 한 번에 모든 변경을 올리는 대신, 논리적 단위(합성 데이터 → 평가 에이전트 → CoT 적용 → 영어화)로 4개의 커밋을 분리하여 **개발 과정의 사고 흐름**이 Git 히스토리에 남도록 설계했습니다.

## 코드 영어 표준화 (13개 파일)

GitHub 포트폴리오의 국제적 가독성을 위해 전체 Python 파일의 한국어 주석/docstring을 영문으로 변환했습니다.

- `services/`: `agents.py`, `prompts.py`, `vector_store.py`, `security.py`, `item_grouping.py`
- `utils/`: `pdf_utils.py`, `csv_utils.py`, `docx_utils.py`
- `scripts/`: `generate_synthetic_data.py`, `data_collector.py`
- 최상위: `config.py`, `schemas.py`

## 프롬프트 엔지니어링 보강

Analysis Agent 시스템 프롬프트에 3가지 핵심 개선을 적용했습니다. **이 프롬프트가 Llama-3의 "교과서"를 작성하는 역할**이므로, 학습 데이터 품질에 직접적 영향을 미칩니다.

### 1. 페르소나 부여
```diff
- You are the Analysis Agent for purchasing and inventory operations.
+ You are a senior procurement analyst with 15+ years of experience
+ in supply chain risk management.
```
**효과**: 범용 도우미가 아닌 베테랑 전문가의 시각으로 분석하도록 유도하여 출력의 깊이가 향상됩니다.

### 2. Edge Case Handling (Section 4 신설)
- `wks_to_oos`가 null/0일 때 → "Data gap — manual stock review needed" 플래그
- `risk_level` 누락 시 → "N/A" 기본값 + 갭 명시
- 양쪽 히스토리 부재 시 → 현재 데이터만으로 분석한다는 한계 명시

**효과**: 예외 상황에서 모델이 추측하지 않고 정직하게 한계를 밝히는 패턴을 강화합니다.

### 3. Few-shot 예시 (Section 6 신설)
히스토리가 있을 때와 없을 때의 올바른 출력 패턴을 시범으로 보여주어, 모델이 **패턴 매칭**으로 즉시 기대되는 품질 수준을 이해하도록 했습니다.

**효과**: JSON 구조 준수율 향상 + 할루시네이션 추가 감소 기대.

### 🐛 Critical Bug Fix: Evaluation Agent의 히스토리 컨텍스트 누락

데이터 재생성 후 분석 중 **심각한 채점 오류**를 발견했습니다.

**증상**: 5.5점 시나리오들의 Data Faithfulness가 일괄 3/10. Analysis Agent가 히스토리를 정확하게 인용했음에도 "날조(Fabrication)"로 판정됨.

**근본 원인**: `evaluation_node`가 `run_evaluation_agent`를 호출할 때 **히스토리 컨텍스트를 전달하지 않았음**.

```diff
# agents.py - evaluation_node
  out = run_evaluation_agent(
      state["supplier"],
      state["items"],
      state["analysis_output"],
+     supplier_history=state.get("supplier_history_override"),
+     item_history=state.get("item_history_override"),
  )
```

**추가 수정**: Evaluation Agent 프롬프트에 CRITICAL 지시 추가 — `provided_supplier_history`와 `provided_item_history`를 반드시 교차 검증하도록 강제.

**교훈**: Agent 간 데이터 전달 경로를 검증하지 않으면, 평가 시스템 자체가 오염되어 학습 데이터의 품질 판단이 왜곡될 수 있습니다. 이 수정은 SFT/DPO 학습 데이터의 정확성에 직접적 영향을 미칩니다.

---

# Phase 2: SFT Training on Vertex AI [COMPLETED]

GPT-4o 교사 데이터셋을 이용해 Llama-3-8B를 QLoRA로 미세 조정하고, Vertex AI Custom Training Job으로 클라우드 환경에서 훈련을 완료한 단계입니다.

## 인프라 구성

| 구성 요소 | 선택 | 이유 |
|---|---|---|
| 모델 | `unsloth/llama-3-8b-bnb-4bit` | 이미 4bit 양자화된 허깅페이스 공식 모델 — VRAM 절감 |
| 훈련 방식 | QLoRA (rank 16, alpha 32) | 전체 파라미터의 0.51%만 학습, T4(16GB)에서 실행 가능 |
| 훈련 프레임워크 | Unsloth + TRL SFTTrainer | Unsloth의 Flash Attention 2 최적화로 SFTTrainer 대비 ~2x 속도 |
| 클라우드 | Vertex AI Custom Training | T4 GPU 지원, 컨테이너 기반 재현성 확보 |
| 이미지 빌드 | Cloud Build + Artifact Registry | `cloudbuild.yaml`로 빌드, `sft-trainer:latest` 태그로 관리 |
| 아티팩트 저장 | GCS (`gs://purchasing-automation-models/`) | 컨테이너 스토리지는 에피머럴 — 훈련 후 즉시 GCS 업로드 |

## 최종 훈련 결과

- **5 에포크**, 총 **367초** (Tesla T4 1장)
- **Loss**: 1.14 → **0.41** (64% 감소)
- **어댑터 저장 위치**: `gs://purchasing-automation-models/sft-runs/lora_adapter/`
- **W&B 실험 기록**: `purchasing-automation-sft` 프로젝트

## 시행착오 및 해결 과정

### 🐛 Bug 1: `torch.int1` AttributeError (가장 치명적)

**증상**: 컨테이너 시작 직후 `AttributeError: module 'torch' has no attribute 'int1'` 발생, 잡 즉시 실패.

**원인 추적**:
```
unsloth_zoo >= 2026.3 → torchao >= 0.13 → torch.int1 사용
torch.int1은 PyTorch 2.6.0에서 추가됨
기존 베이스 이미지 pytorch/pytorch:2.5.1 에는 없음
```

**해결**: 베이스 이미지를 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`로 교체.

**교훈**: ML 패키지 버전 간 의존성은 문서화된 것과 달리 실제 실행 환경에서만 발견되는 경우가 많습니다. 훈련 전 항상 `python -c "import unsloth"` 수준의 smoke test를 컨테이너에서 실행해야 합니다.

### 🐛 Bug 2: Cloud Build 치환 변수 미동작

**증상**: `cloudbuild.yaml`의 `substitutions` 블록에서 `${PROJECT_ID}`가 확장되지 않아 이미지 URI가 잘못 생성됨.

**원인**: Cloud Build의 `substitutions` 블록 내에서는 `${PROJECT_ID}` 같은 built-in 변수가 자동 확장되지 않음 (user-defined substitutions 영역이기 때문).

**해결**: `_IMAGE_URI`에 프로젝트 ID를 하드코딩.
```yaml
substitutions:
  _IMAGE_URI: 'us-central1-docker.pkg.dev/purchasing-automation/purchasing-automation/sft-trainer:latest'
```

### 🐛 Bug 3: TRL이 `wandb.finish()` 자동 호출

**증상**: `trainer.train()` 이후 `wandb.log_artifact()`를 호출하면 `wandb run is not running` 에러 발생.

**원인**: TRL의 `SFTTrainer`는 훈련 완료 후 내부적으로 `wandb.finish()`를 자동 호출함. 이후 코드에서 wandb API를 다시 사용하면 실패.

**해결**: `trainer.train()` 이후 `wandb.finish()` 및 `wandb.log_artifact()` 호출 제거.

### 🐛 Bug 4: 컨테이너에서 `gsutil` 명령어 없음

**증상**: GCS 업로드를 `subprocess.run(["gsutil", "cp", ...])` 로 구현했더니 `FileNotFoundError`.

**원인**: PyTorch 공식 이미지에는 Google Cloud SDK가 설치되어 있지 않음.

**해결**: `google-cloud-storage` Python 패키지를 사용하는 방식으로 교체.
```python
from google.cloud import storage as gcs
client = gcs.Client()
for f in Path(adapter_path).rglob("*"):
    if f.is_file():
        bucket.blob(f"sft-runs/lora_adapter/{f.relative_to(adapter_path)}").upload_from_filename(str(f))
```

---

# Phase 3: Post-SFT 평가 (eval_sft.py) [COMPLETED]

훈련된 SFT 모델이 실제로 얼마나 GPT-4o에 근접한 품질을 내는지, holdout 테스트셋(마지막 5개 시나리오)으로 정량 평가한 단계입니다.

## 평가 방법론

각 holdout 예제마다:
1. **Llama SFT 추론**: LoRA 어댑터 로드 후 동일 프롬프트 입력
2. **GPT-4o 베이스라인**: 동일 입력, 기존 프로덕션 에이전트 실행
3. **GPT-4o-as-judge**: 두 결과를 각각 1~10점으로 채점
4. **JSON 유효성**: 실제 프로덕션 코드(`_extract_json_from_text`)와 동일한 파싱 로직 적용

## 최종 결과 (v4)

| 지표 | Llama SFT | GPT-4o |
|---|---|---|
| 평균 점수 | **7.4 / 10** | 9.3 / 10 |
| JSON 유효율 | **80%** (5개 중 4개) | 100% |
| 유효 출력 평균 점수 | **9.25 / 10** | — |

> 유효한 JSON을 생성했을 때의 품질은 GPT-4o와 거의 동등 (9.25 vs 9.3). 핵심 과제는 **일관성(consistency)** 확보.

## 시행착오: 4번의 파라미터 튜닝

### v1 → v2: Over-generation 발견

**문제**: `max_new_tokens=2048` 설정 시 일부 예제에서 JSON을 완성한 후에도 모델이 계속 텍스트를 생성 (EuroSupply, CraftySupplies 실패).

**원인**: SFT 소형 모델은 EOS 토큰 이후 생성을 멈추는 능력이 GPT-4o보다 약함. `max_new_tokens`가 충분히 크면 JSON 뒤에 설명 텍스트, 반복 패턴 등을 추가 생성.

**해결**: `max_new_tokens=900` (분석 JSON의 실제 최대 크기 ~700토큰보다 약간 여유 있게 설정) + `is_valid_json()`에 정규식 추출 로직 추가.

```python
# 생성된 텍스트에서 첫 번째 완전한 JSON 객체 추출 (trailing text 무시)
m = re.search(r"(\{[\s\S]*\})", text)
if m:
    return True, json.loads(m.group(1))
```

### v2 → v3: temperature=0 회귀

**문제**: `temperature=0` (greedy decoding)으로 변경했더니 JSON 유효율이 80% → 60%로 하락.

**원인**: Greedy decoding은 반복 루프(repetition loop)에 취약. 소형 SFT 모델은 가장 확률 높은 토큰을 계속 선택하다 보면 `"items": [{"items": [{"items":...` 같은 무한 중첩 패턴에 빠지는 경향이 있음.

**해결**: `temperature=0.1`, `do_sample=True`로 복원. 약간의 무작위성이 반복 루프 탈출에 도움이 됨.

### 발견된 잠재적 버그: SupplierName 키 오류

훈련 스크립트(`train_sft.py`)에서 공급사명을 가져올 때 `inp["inventory"][0].get("supplier")` (소문자)를 사용했지만, 실제 데이터셋의 키는 `"SupplierName"` (대소문자 혼합)임. 결과적으로 **모든 훈련 예제에서 프롬프트의 Supplier 필드가 "Unknown"** 으로 입력됨.

**영향 최소화**: 공급사명은 `Items` 필드의 JSON 내에도 포함되어 있어 모델이 컨텍스트에서 유추 가능. 재훈련 없이도 실용적 수준의 성능 확보. 단, DPO 훈련 시에는 동일한 키 오류를 유지해야 학습 일관성이 보장됨.

---

# 패키지 선택 전략 (Unsloth vs 표준 PEFT)

반복적으로 발생한 버전 호환성 문제의 근본 원인을 분석하여, 이후 모든 훈련 단계에서 적용할 명확한 기준을 수립합니다.

## 왜 Unsloth는 DPO에서 실패하는가 (그리고 해결책)

Unsloth는 훈련 속도를 높이기 위해 PyTorch 모델의 내부를 **패치(patch)**합니다. 구체적으로:
- Flash Attention 2를 커스텀 CUDA 커널로 대체
- forward pass를 최적화된 구현으로 교체
- 메모리 레이아웃을 최적화 목적에 맞게 재구성

DPOTrainer는 매 스텝마다 **같은 모델을 두 번** forward pass해야 합니다:
1. 현재 정책(policy) 모델로 chosen/rejected 로그 확률 계산
2. 참조(reference) 모델(`ref_model=None` 시 PEFT adapter 비활성화)로 동일 계산

Unsloth 단독으로는 이 이중 forward pass 패턴을 지원하지 않아 `CUDA error: an illegal memory access`가 발생합니다. **해결책: `PatchDPOTrainer()`** — Unsloth가 공식 제공하는 함수로, DPOTrainer의 이중 forward pass를 Unsloth 커널과 호환되도록 패치합니다.

## 확립된 패키지 전략

| 작업 | 도구 | 이유 |
|---|---|---|
| SFT 훈련 | **Unsloth** + TRL SFTTrainer | 단일 forward pass — Unsloth 최적화 안전하게 적용 가능 |
| 추론 (inference) | **Unsloth** FastLanguageModel.for_inference() | 추론 속도 최적화 |
| DPO 훈련 | **Unsloth** + `PatchDPOTrainer()` | `PatchDPOTrainer()`가 이중 forward pass를 Unsloth 커널과 호환되도록 패치 |
| RLHF/PPO 등 | **표준 PEFT** | Unsloth 공식 패치 미제공 시 사용 |

## TRL API 호환성 주의사항

TRL 버전이 올라가면서 API가 변경됩니다:
- `DPOTrainer(tokenizer=...)` → **`DPOTrainer(processing_class=...)`** (TRL >= 0.12)
- `SFTTrainer`는 훈련 후 `wandb.finish()` 자동 호출 — 이후 wandb API 재호출 금지

---

# Phase 4: DPO (Direct Preference Optimization) 훈련 [COMPLETED]

SFT만으로는 20%의 JSON 유효율 실패와 일관성 부족이 남음. DPO를 통해 "GPT-4o 출력(chosen) vs SFT 출력(rejected)" 선호도 쌍을 학습시켜 모델의 출력 일관성과 포맷 준수를 개선합니다.

## SFT → DPO 2단계 정렬 파이프라인

```
Base Llama-3-8B
    ↓ (Phase 2)
SFT LoRA Adapter  ← 도메인 지식 + JSON 스키마 학습
    ↓ (Phase 3.5: generate_dpo_pairs.py)
25개 선호도 쌍 생성
  chosen:  GPT-4o 교사 출력 (ground truth)
  rejected: SFT Llama 출력 (lower quality)
    ↓ (Phase 4: train_dpo.py)
DPO LoRA Adapter  ← 선호도 정렬 (hallucination ↓, consistency ↑)
```

## Phase 3.5: DPO 선호도 쌍 생성 (generate_dpo_pairs.py)

- holdout 5개를 제외한 **25개 훈련 예제**에 대해 SFT 모델로 추론 실행
- 각 예제마다 `(prompt, chosen=GPT-4o_output, rejected=SFT_output)` 쌍 생성
- 결과: `gs://purchasing-automation-models/dpo-data/dpo_preference_pairs.jsonl` (148KB, 25쌍)

**첫 번째 잡 실패**: 스크립트 작성 후 Docker 이미지를 재빌드하지 않아 컨테이너 내에 `generate_dpo_pairs.py`가 없었음. 이미지 재빌드 후 재제출으로 해결.

**로그의 `--- Logging error ---`**: `transformers` 내부 deprecation 경고 문자열에 `%`가 포함되어 Python logging 프레임워크가 포맷 오류를 냄. 실제 앱 크래시가 아니며 잡은 정상 실행됨. 육안으로는 에러처럼 보이지만 무시 가능.

## DPO 훈련 설정

| 하이퍼파라미터 | 값 | 이유 |
|---|---|---|
| β (beta) | 0.1 | 낮을수록 공격적인 선호도 학습. SFT 이후 단계이므로 안정성 우선 |
| Learning Rate | 5e-5 | SFT(2e-4)보다 낮게 설정 — 이미 학습된 지식 훼손 방지 |
| Epochs | 3 | 25개 소규모 데이터셋 과적합 방지 |
| ref_model | None (implicit) | PEFT 모델에서는 어댑터 비활성화로 자동 reference 생성 |

## train_dpo.py Vertex AI 적응 수정 사항

`train_dpo.py`는 로컬 경로에서 직접 읽도록 작성되어 있어, Vertex AI 에피머럴 컨테이너 환경에서 실행하려면 GCS 다운로드/업로드 로직이 필요했습니다.

```python
# if __name__ == "__main__": 블록에 추가
# 1. GCS에서 SFT 어댑터 다운로드
_download_gcs_prefix(GCS_ADAPTER_URI, SFT_MODEL_PATH)
# 2. GCS에서 선호도 쌍 다운로드
_download_gcs_file(GCS_PAIRS_URI, PREFERENCE_DATA_PATH)
# 3. 훈련
train_dpo()
# 4. DPO 어댑터 GCS 업로드
_upload_dir_to_gcs(adapter_path, f"{GCS_OUTPUT_URI}/lora_adapter")
```

또한 SFT에서 학습한 교훈 적용: `train_dpo()`내의 `wandb.finish()`와 `wandb.log_artifact()` 제거 (TRL이 `trainer.train()` 후 자동 호출하므로 중복 호출 시 실패).

## DPO 훈련 시행착오

### 🐛 Bug 1: `DPOTrainer` API 변경 (`tokenizer` → `processing_class`)

TRL >= 0.12에서 `DPOTrainer(tokenizer=...)` 파라미터명이 `processing_class`로 변경됨.

**해결**: `tokenizer=tokenizer` → `processing_class=tokenizer`

### 🐛 Bug 2: Unsloth CUDA 커널 충돌 (`CUDA error: an illegal memory access`)

**증상**: `RuntimeError: CUDA error: an illegal memory access` (step 0/12)

**원인**: 위 패키지 전략 섹션 참조. Unsloth 커스텀 커널 + DPOTrainer 이중 forward pass 충돌.

**잘못된 시도**: Unsloth 완전 제거 → 표준 PEFT + BitsAndBytes로 교체 → OOM 발생 (T4 16GB 초과)

**올바른 해결**: Unsloth 공식 `PatchDPOTrainer()` 사용. DPOTrainer import **전에** 호출해야 함.
```python
# train_dpo.py 최상단 (다른 import 전)
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()  # DPOTrainer를 Unsloth 커널과 호환되도록 패치

from trl import DPOTrainer, DPOConfig  # 반드시 PatchDPOTrainer() 호출 후
```

### 🐛 Bug 3 & 4: 잘못된 접근(표준 PEFT)에서 비롯된 부차적 문제들

> **참고**: Bug 3 & 4는 Bug 2의 잘못된 해결책(Unsloth 제거 → 표준 PEFT)을 시도하면서 발생한 문제들임. `PatchDPOTrainer()` 접근법에서는 발생하지 않음.

**Bug 3 — `model.warnings_issued` AttributeError**: 표준 PEFT `PeftModel`의 `__getattr__` 프록시가 `warnings_issued` 속성을 찾지 못함. `object.__setattr__(model, 'warnings_issued', {})` 우회로 해결 가능하지만 근본 해결책은 `PatchDPOTrainer()`.

**Bug 4 — `quantization_config` 중복**: `unsloth/llama-3-8b-bnb-4bit`는 4-bit 설정이 `config.json`에 내장됨. 표준 PEFT로 로드 시 `BitsAndBytesConfig`를 추가로 전달하면 충돌 경고 발생. `PatchDPOTrainer()` + `FastLanguageModel.from_pretrained(load_in_4bit=True)` 사용 시 자동 처리.

## DPO 훈련 최종 결과

- **3 에포크**, Tesla T4 1장, 약 14분
- **어댑터 저장 위치**: `gs://purchasing-automation-models/dpo-runs/lora_adapter/`
- **W&B 실험 기록**: `purchasing-automation-dpo` 프로젝트

---

# Phase 5: Before/After 비교 평가 (eval_dpo.py) [COMPLETED]

SFT 이후 DPO가 실질적인 품질 향상을 가져왔는지 검증하기 위해, Base Llama / SFT / SFT+DPO / GPT-4o 4개 모델을 동일 holdout셋(5개 시나리오)에서 비교 평가한 단계입니다.

## 평가 방법론

각 holdout 예제마다 4개 모델 모두 동일 프롬프트로 추론 후, GPT-4o-as-judge로 채점:
- **JSON 유효성 게이트**: JSON 파싱 실패 시 GPT-4o 판정 없이 0점 처리
- **data_accuracy (1-10)**: 공급사명, 아이템 코드, 재고 수준, 위험도를 정확히 참조했는가
- **reasoning_quality (1-10)**: 보충 분석과 핵심 질문이 논리적으로 타당한가
- **avg**: (data_accuracy + reasoning_quality) / 2
- **기준점**: GPT-4o ground truth를 자기 자신과 비교(상한선)

## 최종 결과

| 모델 | 평균 점수 | JSON 유효 |
|---|---|---|
| Base Llama-3-8B (미튜닝) | 0.0 / 10 | 0 / 5 |
| SFT | **9.3 / 10** | 5 / 5 |
| SFT+DPO | 7.4 / 10 | 5 / 5 |
| GPT-4o (상한선) | 10.0 / 10 | 5 / 5 |

### 시나리오별 상세

| 시나리오 | SFT avg | SFT+DPO avg | 차이 |
|---|---|---|---|
| SupplierTech | 9.5 | 7.0 | -2.5 |
| EuroSupply | 8.5 | 7.5 | -1.0 |
| MegaManufacturers | 9.5 | 7.5 | -2.0 |
| CraftySupplies | 9.5 | 8.0 | -1.5 |
| EnviroGoods | 9.5 | 7.0 | -2.5 |

## 분석 및 교훈

### Base Llama 0점에 대하여

Base Llama가 0/10인 이유는 **추론이 틀려서가 아니라, JSON을 전혀 생성하지 않아서**입니다. 미튜닝 모델은 프롬프트를 보고 free-form 텍스트를 출력합니다:

- `"Draft a report on the supplier's performance..."` — instruction 반복
- `"The company should consider increasing its stock of EcoBags..."` — 관련 내용이지만 비정형

JSON 유효성 게이트(`is_valid_json()`)에서 전부 탈락하여 자동으로 0점 처리됩니다. 이는 의도된 동작이며, **파인튜닝의 필요성(0 → 9.3)을 명확하게 드러내는 데이터**입니다.

### DPO 퇴화(Regression): SFT 9.3 → SFT+DPO 7.4

DPO가 SFT보다 성능이 낮아진 원인:

1. **학습 쌍 부족**: DPO 학습에 사용된 쌍은 고작 **7쌍** (12 총 예제 - 5 holdout). DPO는 통상 수백~수천 쌍이 있어야 안정적 개선이 가능.
2. **Rejected 샘플 품질 문제**: "거부(rejected)" 샘플이 SFT 초기 출력(주로 invalid JSON)이었음. 모델은 "JSON 형식을 지켜야 한다"는 신호는 학습했지만, 그 과정에서 **분석의 깊이(reasoning depth)를 잃음** — GPT-4o 판정 코멘트가 일관되게 *"lacks detailed replenishment analysis and critical questions"*로 나타남.
3. **데이터 불균형**: `data_accuracy`는 8-9를 유지하면서 `reasoning_quality`만 5-7로 급락. 형식은 맞추지만 내용이 얕아짐.

DPO 퇴화는 소규모 데이터셋에서 알려진 현상입니다. 더 많은 고품질 선호도 쌍이 확보되지 않는 한, **현 프로젝트 규모에서는 SFT 단독(9.3/10)이 최선의 모델**입니다.

### 전체 파이프라인 요약

| 단계 | 결과 | 의의 |
|---|---|---|
| Base Llama | 0/10 (JSON 불가) | 파인튜닝 없이는 도메인 태스크 수행 불가 |
| SFT | 9.3/10 | GPT-4o 지식 증류로 GPT-4o(10.0)에 근접 |
| DPO | 7.4/10 | 소규모 데이터(7쌍)에서의 DPO 한계 직접 확인 |
| GPT-4o | 10.0/10 | 상한선 (ground truth 자기 참조) |
