# Fine-tuning Gemma 4 12B as the analysis agent

Korean translation: [below](#한국어).

**2026-10-25 and 26.** Whether a self-hosted open-weight model can take `gpt-6.1-sol`'s place as
the analysis agent: the training data, what one training example is, the training and the result.
The analysis call receives the most company data (every item's figures and every history
document), so serving it in-house would keep that data away from model companies.

## 1. Base model

Five open-weight models, untuned, on the 25 evaluation cases with the production prompt and
checks:

| Untuned, 25 cases | Ready (every blocking check passed) | Lead-time changes applied (of Sol's 11) | Changed items with a wrong figure |
|---|---|---|---|
| `gpt-6.1-sol` (reference) | 24 | 11 | 0/22 |
| **Gemma 4 12B** | **14** | 1 | 1/1 |
| Gemma 4 E4B | 3 | 1 | 2/2 |
| Qwen3.5-9B | 8 | 11 | 21/22 |
| Ministral 3 14B / 8B | 0 / 0 | 11 / 11 | 22/22 / 22/22 |

Gemma 4 12B was the most often ready. Its main gap was a policy (it found the delays in the
history but put no allowance into the lead time), which training data can show; the Qwen and
Ministral models got nearly every changed figure wrong.

## 2. The training data

### How it was made

In short: invent suppliers, turn each one into exactly the input the production analysis agent
receives, let the production model answer it, and keep only the answers that pass the production
checks.

```
invented supplier  ->  production preprocessing  ->  gpt-6.1-sol answers  ->  production checks  ->  training example
(117 scenarios)        (code works out figures)      (production prompt)      (98 of 116 pass)
```

1. **Invent the situations** (`scripts/generate_sft_scenarios.py`). 117 made-up suppliers. Each
   has 2–6 items in stock and 0–11 history documents: emails, notices, incident reports and the
   like (598 documents in all). The stock figures are set in code to fit the scenario's type; the
   names and document texts are written by `gpt-6.1-sol`. No supplier or product name appears in
   the evaluation cases.
2. **Turn each into the production input** (`scripts/eval_scenarios.py`, `preprocess`). The stock
   rows go through the same preprocessing as a real stock CSV, so the risk level, gap, order and
   delivery dates and suggested quantity are what production code would work out. The model is
   given these figures; it does not calculate them.
3. **Get the answer to copy** (`scripts/build_sft_teacher_data.py`). `gpt-6.1-sol`, the production
   analysis model (medium effort), answers each scenario once with the production prompt
   (`prompts/v5/analysis.md`). This is the answer the open model is trained to reproduce, the
   "teacher" answer.
4. **Keep only good answers.** Every answer is scored by the production checks: code checks C1–C10,
   then the judge (`claude-haiku-5-5`, high effort). One scenario produced no usable answer, so 116
   were scored. An answer became a training example only if it passed every blocking check the
   first time: **98 of 116**. The 18 that failed are kept apart, for possible preference training
   (a good and a bad answer to the same input).

Each scenario has one of 13 types, nine scenarios per type, so every kind of situation the agent
must handle is equally represented, rather than whatever the generator tends to write:

- **History:** clean; none; a delay with no size; a lead-time notice with a size; an item quality
  incident; a supplier-wide quality problem; a demand change in the UK; a demand surge in Korea; a
  demand surge in an unstated market; ambiguous attribution.
- **Stock:** at least one Critical item with a minor or resolved issue; every item Low.
- **Data:** missing or invalid stock values.

### What one training example is

Three messages, exactly as the pipeline sends and receives them in production:

| Message | Content | Where it comes from |
|---|---|---|
| System | The production analysis prompt, `prompts/v5/analysis.md`, word for word (about 14,400 characters) | The prompt file |
| User | One JSON object: the snapshot date, the supplier, each item with the figures code worked out, and the numbered history documents with their header fields | Steps 1–2 |
| Assistant | The teacher's raw answer: one JSON object of findings, lead-time and quantity changes, figures after the changes, stock findings, questions, actions and summary | Step 3 |

Every field of the user message (the input):

```
user message
├─ snapshot_date          date of the stock snapshot, "2026-03-02" for every example
├─ supplier               supplier name
├─ items[]                one per usable stock row; every value worked out in code
│  ├─ item_code, item_name
│  ├─ risk_level                          Critical / High / Medium / Low
│  ├─ available_stock                     on hand + on order − committed
│  ├─ wks_to_oos                          weeks until stockout at current sales
│  ├─ expected_stockout_weeks             the gap: weeks without stock even if ordered today
│  ├─ suggested_quantity                  26 weeks of sales after arrival
│  ├─ recommended_latest_po_date          latest order date
│  ├─ recommended_latest_delivery_date    latest delivery date
│  ├─ recommended_latest_po_timing        the same as "within N weeks"
│  ├─ recommended_latest_delivery_timing  the same as "within N weeks"
│  ├─ data_issues[]                       problems in the row's values (empty here: rows
│  │                                      with unusable values are left out before the model)
│  └─ system_notes[]                      notes from code, e.g. "Critical: even if ordered
│                                         today, a stockout of about 14.1 weeks is expected
│                                         (lead time 18 weeks)."
└─ history[]              the scenario's documents, numbered (0–11)
   ├─ id                  the number the answer cites
   ├─ supplier            header: the supplier it is about ("" = any supplier of the items)
   ├─ items, item_codes   header: the items it is about ([] = the whole supplier)
   ├─ date                header: document date
   └─ text                the document body
```

How each item figure is worked out: `docs/stock_figures.md`. In production the history documents
are retrieved in code by these headers; in the training data the scenario's documents are given
directly, in the same form.

Every field of the assistant message (the answer to learn):

```
assistant message
├─ findings[]              facts in the history
│  └─ id, items, text, basis {stock: false, documents: [history ids]}
├─ supplier_lead_time      a lead-time change for every item, or null
│  └─ lead_time_weeks, reason, findings [finding ids]
├─ lead_time_changes[]     a lead-time change for some items
│  └─ items, lead_time_weeks, reason, findings
├─ quantity_changes[]
│  └─ item_code, original_quantity, factor, proposed_quantity, reason, findings
├─ figures_after_changes[] the model's own figures after a lead-time change
│  └─ item_code, new_lead_time_weeks, new_risk_level, new_gap_weeks, new_latest_po_date
├─ stock_findings[]        facts from the stock figures alone
│  └─ id, items, text, basis {stock: true, documents: []}
├─ critical_questions[]
│  └─ question, ask ("supplier" or "internal"), items, findings
├─ recommended_actions[]   internal actions
│  └─ action, items, findings
└─ summary
```

Each change, question and action cites the findings it rests on, and each finding cites its
documents; code checks every citation (C-checks) and recalculates the figures after a change.

The user message, shortened:

```json
{
  "snapshot_date": "2026-03-02",
  "supplier": "Jinora Care Co Ltd",
  "items": [
    {"item_code": "804600", "item_name": "CamelliaSilkBodyWash", "risk_level": "Low",
     "available_stock": 6720, "wks_to_oos": 56, "expected_stockout_weeks": 0.0,
     "suggested_quantity": 3120, "recommended_latest_po_date": "2026-11-23",
     "recommended_latest_delivery_date": "2027-03-29", "recommended_latest_po_timing": "within 38 weeks",
     "recommended_latest_delivery_timing": "within 56 weeks", "data_issues": [], "system_notes": []},
    ...
  ],
  "history": [
    {"id": 1, "supplier": "Jinora Care Co Ltd", "items": [], "item_codes": [], "date": "2025-05-14",
     "text": "Supplier onboarding review completed for the Korean personal care factory. ..."}
  ]
}
```

The assistant message, shortened:

```json
{
  "findings": [
    {"id": 1, "items": [], "text": "The May 2025 onboarding review requested ...",
     "basis": {"stock": false, "documents": [1]}}
  ],
  "supplier_lead_time": null, "lead_time_changes": [], "quantity_changes": [],
  "figures_after_changes": [], "stock_findings": [...], "critical_questions": [...],
  "recommended_actions": [...], "summary": "..."
}
```

What the model is **not** given: the checks, their verdicts or any explanation of why an answer
passed; the teacher's reasoning (its answer only); any company data. The example teaches the
answer to the production input, nothing more.

How it is rendered for training: Gemma 4's chat template with thinking off, as vLLM prompts the
model in serving, so training and serving see the same text. The loss is on the assistant
message only. A typical example is about 4,800 prompt tokens and 2,000 answer tokens.

### Split

| Set | Size | Use |
|---|---|---|
| Training | 85 of the 98 | Training, in a fixed order interleaving the types |
| Validation | 13 of the 98 (a tenth per type) | Validation loss, to choose the checkpoint |
| Generalisation | 7 scenarios of a type no training scenario has (a supplier pushing a newly launched product) | Evaluation only |
| Evaluation | The 25 cases every model was compared on | Evaluation only; never trained on |

## 3. Training

QLoRA (base in 4-bit, LoRA rank 16 on the language model's attention and MLP layers), learning
rate 2e-4, 3 epochs, the checkpoint with the lowest validation loss, on one RTX 5090
(`scripts/train_sft_v5.py`). A learning curve: 25, 50 and 85 training examples (each size contains
the smaller ones), two seeds each, six runs.

| Training examples | Validation loss (mean of two seeds) |
|---|---|
| 25 | 0.452 |
| 50 | 0.396 |
| 85 | 0.328 |

## 4. Result

| 25 evaluation cases | Ready |
|---|---|
| `gpt-6.1-sol` | 24 |
| Gemma 4 12B, untuned | 14 |
| Fine-tuned, six runs | 3–10 |

Validation loss fell steadily while readiness fell. Judged by loss alone, the fine-tune would have
looked like a success; the production checks caught the regression.

**The cause.** Fine-tuning taught the teacher's policy but not its wording. On the 18 evaluation
cases with a Critical item, the teacher asked the supplier about expedited or partial delivery 18
times and never named the internal gap; the untuned model asked twice; the fine-tuned models asked
19–21 times, as the teacher does, but in 15–20 of those questions wrote the phrase the internal
actions use ("to bridge the 8-week gap"). That reveals the company's stock position to the
supplier, a blocking failure (J5). The phrase is in 60 of the teacher's 300 actions and in none of
its 156 supplier questions; with 85 examples the model learned the shared shape of the two fields
but not which one may name the gap.

## 5. Decision

The analysis agent stays on `gpt-6.1-sol` (decided 2026-10-26). Serving was measured as feasible
(`docs/serving_gemma4_vllm.md`), but the open model, untuned or fine-tuned, falls short on the same
checks that chose every production model. A possible next step, not tried: train on the base
model's own passing outputs with their reasoning (rejection-sampling fine-tuning), aimed at both
the arithmetic after a change and the field mix-up.

## Files

| What | Where |
|---|---|
| Scenarios, teacher answers, validation and generalisation sets | `training_data/sft_v5/` |
| Scripts | `scripts/generate_sft_scenarios.py`, `scripts/build_sft_teacher_data.py`, `scripts/train_sft_v5.py`, `scripts/train_learning_curve.ps1`, `scripts/screen_base_models.py`, `scripts/report_sft_results.py` |
| Evaluation outputs | `eval_results/sft_screen_*.jsonl` |

---

# 한국어

## Gemma 4 12B 분석 에이전트 파인튜닝

**2026-10-25~26.** 자체 호스팅하는 오픈웨이트 모델이 분석 에이전트 자리에서 `gpt-6.1-sol`을 대신할 수
있는지 검토한 기록입니다. 학습 데이터, 학습 예시 하나의 구성, 학습, 결과 순으로 적습니다. 분석 호출은
회사 데이터를 가장 많이 받습니다(모든 품목의 수치와 모든 이력 문서). 사내에서 서빙하면 이 데이터를
모델 회사에 보내지 않아도 됩니다.

### 1. 베이스 모델

오픈웨이트 모델 5종을 튜닝 없이, 운영 프롬프트와 운영 검사로 평가 케이스 25건에 돌렸습니다.

| 튜닝 전, 25건 | Ready (차단 검사 전부 통과) | 리드타임 변경 반영 (Sol의 11건 중) | 변경한 품목 중 수치가 틀린 것 |
|---|---|---|---|
| `gpt-6.1-sol` (기준) | 24 | 11 | 0/22 |
| **Gemma 4 12B** | **14** | 1 | 1/1 |
| Gemma 4 E4B | 3 | 1 | 2/2 |
| Qwen3.5-9B | 8 | 11 | 21/22 |
| Ministral 3 14B / 8B | 0 / 0 | 11 / 11 | 22/22 / 22/22 |

Gemma 4 12B가 Ready가 가장 많았습니다. 주된 부족함은 방침이었습니다. 이력에서 지연을 찾아내고도
리드타임에 여유분을 반영하지 않았는데, 이런 방침은 학습 데이터로 보여줄 수 있습니다. 반면 Qwen과
Ministral은 변경한 수치를 거의 전부 틀렸습니다.

### 2. 학습 데이터

#### 만든 과정

한 줄로 말하면, 가상의 공급업체를 만들고, 각각을 운영 분석 에이전트가 실제로 받는 입력 형태로 바꾼 뒤,
운영 모델이 답하게 하고, 운영 검사를 통과한 답만 남겼습니다.

```
가상 공급업체   ->  운영 전처리         ->  gpt-6.1-sol 답변  ->  운영 검사         ->  학습 예시
(시나리오 117개)    (코드가 수치 계산)       (운영 프롬프트)         (116건 중 98건 통과)
```

1. **상황 만들기** (`scripts/generate_sft_scenarios.py`). 가상 공급업체 117개를 만들었습니다. 각각 재고
   품목 2~6개와 이력 문서 0~11개(이메일, 공지, 사고 보고서 등, 총 598개)를 가집니다. 재고 수치는
   시나리오 유형에 맞게 코드가 정하고, 이름과 문서 본문은 `gpt-6.1-sol`이 씁니다. 평가 케이스에 나오는
   공급업체명이나 제품명은 쓰지 않았습니다.
2. **운영 입력으로 바꾸기** (`scripts/eval_scenarios.py`, `preprocess`). 재고 행을 실제 재고 CSV와 같은
   전처리에 통과시킵니다. 그래서 위험도, 재고 공백, 발주일과 입고일, 권장 수량이 운영 코드가 계산하는
   값과 같습니다. 모델은 이 수치를 받기만 하고 직접 계산하지 않습니다.
3. **따라 배울 답 받기** (`scripts/build_sft_teacher_data.py`). 운영 분석 모델인 `gpt-6.1-sol`(medium
   effort)이 운영 프롬프트(`prompts/v5/analysis.md`)로 시나리오마다 한 번 답합니다. 오픈 모델이 그대로
   재현하도록 학습할 답, 즉 "교사" 답변입니다.
4. **좋은 답만 남기기.** 모든 답변을 운영 검사로 채점합니다. 코드 검사 C1~C10, 그다음 judge
   (`claude-haiku-5-5`, high effort)입니다. 한 시나리오는 쓸 수 있는 답이 나오지 않아 116건을
   채점했습니다. 첫 답변에서 차단 검사를 모두 통과한 것만 학습 예시가 됐습니다: **116건 중 98건**.
   통과하지 못한 18건은 나중에 선호 학습(같은 입력에 대한 좋은 답과 나쁜 답의 쌍)에 쓸 수 있도록 따로
   보관합니다.

시나리오마다 13개 유형 중 하나가 정해져 있고 유형당 9개입니다. 생성기가 쓰기 쉬운 상황으로 쏠리지 않고,
에이전트가 다뤄야 할 상황이 모두 같은 비중으로 들어가게 하기 위해서입니다.

- **이력:** 문제 없음, 이력 없음, 규모 없는 지연, 규모가 명시된 리드타임 통지, 품목 품질 사고, 공급업체
  전체의 품질 문제, 영국 수요 변화, 한국 수요 급증, 시장이 명시되지 않은 수요 급증, 귀속이 모호한 경우.
- **재고:** Critical 품목이 하나 이상 있으나 사소하거나 해결된 문제만 있는 경우, 모든 품목이 Low인 경우.
- **데이터:** 재고 값이 비었거나 잘못된 경우.

#### 학습 예시 하나의 구성

메시지 세 개이며, 운영에서 파이프라인이 보내고 받는 그대로입니다.

| 메시지 | 내용 | 출처 |
|---|---|---|
| System | 운영 분석 프롬프트 `prompts/v5/analysis.md` 원문 그대로(약 14,400자) | 프롬프트 파일 |
| User | JSON 객체 하나: 기준일, 공급업체, 코드가 계산한 수치가 붙은 각 품목, 헤더 정보가 붙은 번호 매긴 이력 문서 | 1~2단계 |
| Assistant | 교사의 원래 답변: 사실(findings), 리드타임·수량 변경, 변경 후 수치, 재고 관련 사실, 질문, 조치, 요약을 담은 JSON 객체 하나 | 3단계 |

User 메시지(입력)의 전체 필드:

```
user message
├─ snapshot_date          재고 스냅샷 날짜. 모든 예시에서 "2026-03-02"
├─ supplier               공급업체명
├─ items[]                쓸 수 있는 재고 행마다 하나. 모든 값은 코드가 계산
│  ├─ item_code, item_name
│  ├─ risk_level                          Critical / High / Medium / Low
│  ├─ available_stock                     보유 + 입고 예정 − 출고 확정
│  ├─ wks_to_oos                          현재 판매 속도로 재고 소진까지 남은 주
│  ├─ expected_stockout_weeks             재고 공백: 오늘 발주해도 재고가 없는 주
│  ├─ suggested_quantity                  입고 후 26주 판매분
│  ├─ recommended_latest_po_date          최종 발주일
│  ├─ recommended_latest_delivery_date    최종 입고일
│  ├─ recommended_latest_po_timing        같은 내용을 "within N weeks"로
│  ├─ recommended_latest_delivery_timing  같은 내용을 "within N weeks"로
│  ├─ data_issues[]                       행 값의 문제(이 데이터에서는 비어 있음: 쓸 수
│  │                                      없는 값의 행은 모델에 가기 전에 코드가 제외)
│  └─ system_notes[]                      코드가 붙이는 메모. 예: "Critical: even if ordered
│                                         today, a stockout of about 14.1 weeks is expected
│                                         (lead time 18 weeks)."
└─ history[]              시나리오의 이력 문서, 번호를 붙여서(0~11개)
   ├─ id                  답변이 인용하는 번호
   ├─ supplier            헤더: 문서가 다루는 공급업체("" = 공급업체 무관, 그 품목 전체)
   ├─ items, item_codes   헤더: 문서가 다루는 품목([] = 공급업체 전체)
   ├─ date                헤더: 문서 날짜
   └─ text                문서 본문
```

품목 수치의 계산 방법은 `docs/stock_figures.md`에 있습니다. 운영에서는 이력 문서를 이 헤더로 코드가 검색해 오고, 학습 데이터에서는 시나리오의 문서를 같은 형태로
직접 넣었습니다.

Assistant 메시지(배울 답)의 전체 필드:

```
assistant message
├─ findings[]              이력에서 찾은 사실
│  └─ id, items, text, basis {stock: false, documents: [이력 id]}
├─ supplier_lead_time      전 품목 리드타임 변경, 없으면 null
│  └─ lead_time_weeks, reason, findings [사실 id]
├─ lead_time_changes[]     일부 품목 리드타임 변경
│  └─ items, lead_time_weeks, reason, findings
├─ quantity_changes[]      수량 변경
│  └─ item_code, original_quantity, factor, proposed_quantity, reason, findings
├─ figures_after_changes[] 리드타임 변경 후 모델이 직접 다시 낸 수치
│  └─ item_code, new_lead_time_weeks, new_risk_level, new_gap_weeks, new_latest_po_date
├─ stock_findings[]        재고 수치만으로 나온 사실
│  └─ id, items, text, basis {stock: true, documents: []}
├─ critical_questions[]    질문
│  └─ question, ask ("supplier" 또는 "internal"), items, findings
├─ recommended_actions[]   내부 조치
│  └─ action, items, findings
└─ summary                 요약
```

변경, 질문, 조치는 근거가 된 사실을 인용하고, 사실은 근거 문서를 인용합니다. 코드가 모든 인용을
검사하고(C 검사), 변경 후 수치를 다시 계산합니다.

User 메시지(축약):

```json
{
  "snapshot_date": "2026-03-02",
  "supplier": "Jinora Care Co Ltd",
  "items": [
    {"item_code": "804600", "item_name": "CamelliaSilkBodyWash", "risk_level": "Low",
     "available_stock": 6720, "wks_to_oos": 56, "expected_stockout_weeks": 0.0,
     "suggested_quantity": 3120, "recommended_latest_po_date": "2026-11-23",
     "recommended_latest_delivery_date": "2027-03-29", "recommended_latest_po_timing": "within 38 weeks",
     "recommended_latest_delivery_timing": "within 56 weeks", "data_issues": [], "system_notes": []},
    ...
  ],
  "history": [
    {"id": 1, "supplier": "Jinora Care Co Ltd", "items": [], "item_codes": [], "date": "2025-05-14",
     "text": "Supplier onboarding review completed for the Korean personal care factory. ..."}
  ]
}
```

Assistant 메시지(축약):

```json
{
  "findings": [
    {"id": 1, "items": [], "text": "The May 2025 onboarding review requested ...",
     "basis": {"stock": false, "documents": [1]}}
  ],
  "supplier_lead_time": null, "lead_time_changes": [], "quantity_changes": [],
  "figures_after_changes": [], "stock_findings": [...], "critical_questions": [...],
  "recommended_actions": [...], "summary": "..."
}
```

모델에게 **주지 않은** 것: 검사 항목, 판정 결과, 답변이 왜 통과했는지에 대한 설명, 교사의 추론 과정
(답변만 줍니다), 회사 데이터. 학습 예시는 운영 입력에 대한 답만 가르치고, 그 이상은 없습니다.

학습용 렌더링: Gemma 4 채팅 템플릿에 thinking을 끈 형태로, vLLM이 서빙에서 모델에 프롬프트를 넣는 방식과
같습니다. 따라서 학습과 서빙에서 모델이 보는 텍스트가 같습니다. 손실은 assistant 메시지에만 걸었습니다.
예시 하나는 대략 프롬프트 4,800토큰, 답변 2,000토큰입니다.

#### 분할

| 세트 | 크기 | 용도 |
|---|---|---|
| 학습 | 98건 중 85건 | 학습. 유형이 섞이도록 고정된 순서 |
| 검증 | 98건 중 13건(유형별 10분의 1) | 체크포인트 선택용 검증 손실 |
| 일반화 | 학습 시나리오에 없는 유형 7건(신규 출시 제품을 미는 공급업체) | 평가 전용 |
| 평가 | 모든 모델을 비교한 25건 | 평가 전용. 학습에 쓰지 않음 |

### 3. 학습

QLoRA(베이스 4비트, 언어 모델의 어텐션·MLP 레이어에 LoRA rank 16), 학습률 2e-4, 3 epoch, 검증 손실이
가장 낮은 체크포인트, RTX 5090 한 장(`scripts/train_sft_v5.py`). 학습 곡선을 보기 위해 학습 예시 25, 50,
85건(큰 세트가 작은 세트를 포함), 시드 2개씩, 총 6회 학습했습니다.

| 학습 예시 수 | 검증 손실 (시드 2개 평균) |
|---|---|
| 25 | 0.452 |
| 50 | 0.396 |
| 85 | 0.328 |

### 4. 결과

| 평가 25건 | Ready |
|---|---|
| `gpt-6.1-sol` | 24 |
| Gemma 4 12B, 튜닝 전 | 14 |
| 파인튜닝, 6회 | 3~10 |

검증 손실은 꾸준히 내려갔지만 Ready는 줄었습니다. 손실만 봤다면 파인튜닝은 성공처럼 보였을 것이고,
운영 검사가 이 퇴보를 잡아냈습니다.

**원인.** 파인튜닝은 교사의 방침은 가르쳤지만 표현은 가르치지 못했습니다. Critical 품목이 있는 평가
케이스 18건에서, 교사는 공급업체에 긴급·분할 납품을 18번 물었고 내부의 재고 공백은 한 번도 언급하지
않았습니다. 튜닝 전 모델은 2번 물었습니다. 파인튜닝 모델은 교사처럼 19~21번 물었지만, 그중 15~20개의
질문에 내부 조치에서 쓰는 문구("to bridge the 8-week gap")를 썼습니다. 이는 회사의 재고 상황을
공급업체에 드러내는 것으로, 차단 실패(J5)입니다. 이 문구는 교사의 조치 300개 중 60개에 있고, 공급업체
질문 156개에는 하나도 없습니다. 예시 85건으로 모델은 두 필드의 공통된 형태는 배웠지만, 어느 쪽에서
공백을 말해도 되는지는 배우지 못했습니다.

### 5. 결정

분석 에이전트는 `gpt-6.1-sol`을 유지합니다(2026-10-26 결정). 서빙은 가능한 것으로 측정됐지만
(`docs/serving_gemma4_vllm.md`), 오픈 모델은 튜닝 전이든 후든 모든 운영 모델을 고른 바로 그 검사에서
기준에 못 미칩니다. 시도하지 않은 다음 단계: 베이스 모델이 직접 낸 통과 답변을 추론 과정과 함께 학습하는
방법(rejection-sampling fine-tuning). 변경 후 계산과 필드 혼동을 함께 겨냥합니다.

### 파일

| 내용 | 위치 |
|---|---|
| 시나리오, 교사 답변, 검증·일반화 세트 | `training_data/sft_v5/` |
| 스크립트 | `scripts/generate_sft_scenarios.py`, `scripts/build_sft_teacher_data.py`, `scripts/train_sft_v5.py`, `scripts/train_learning_curve.ps1`, `scripts/screen_base_models.py`, `scripts/report_sft_results.py` |
| 평가 출력 | `eval_results/sft_screen_*.jsonl` |
