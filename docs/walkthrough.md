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

### 의사결정: Evaluation Agent는 수정하지 않음
Evaluation Agent 프롬프트는 이미 Balanced Scoring(날조 패널티 + 추론 보상)이 잘 구현되어 있어 추가 수정 없이 유지했습니다. Evaluation Agent는 GPT-4o로 유지될 예정이므로, 향후에도 높은 판별력을 유지할 수 있습니다.
