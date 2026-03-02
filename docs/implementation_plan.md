# [Roadmap] Purchasing Automation: GPT-4o → Fine-tuned Llama-3 전환

기존 GPT-4o 기반 파이프라인을 **Llama-3-8B SFT + DPO**로 전환하여 독자적인 구매 특화 AI 엔진을 구축하는 로드맵입니다.

---

## 전략적 의사결정

| 선택지 | 결정 | 이유 |
|:---|:---|:---|
| SFT vs DPO | **SFT 우선 → DPO 후속** | SFT로 JSON 규격/톤을 습득 후, DPO로 할루시네이션 교정 |
| Llama-3 vs Mistral | **Llama-3-8B** | 업계 인지도, 커뮤니티 자료, QLoRA 지원 풍부 |
| Colab vs Vertex AI | **Vertex AI** | 기존 GCP 인프라와 통합, 엔터프라이즈급 MLOps 경험 |
| 대량 vs 소량 데이터 | **소량 고품질 (30개 Gold Standard)** | CoT가 반영된 정예 데이터가 추론 능력 전이에 효과적 |

---

## Phase 1: Data Pipeline & Teacher Dataset [COMPLETED] ✅

- 합성 시나리오 30개 생성 (에지 케이스 포함)
- CoT(Chain of Thought) 기반 Analysis Agent 개조
- Balanced Evaluation Agent (할루시네이션 방지 + 논리적 추론 보상)
- 고품질 데이터 15건(7.5+) 확보, Gold Standard 3건(8.5)

## Phase 2: Fine-tuning on Vertex AI
- **SFT**: Llama-3-8B + QLoRA, `teacher_dataset_20260225.jsonl` 활용
- **DPO**: 1차 생성(4.5점, Rejected) + 3차 생성(8.5점, Chosen) Preference Pair 활용
- **W&B**: 학습 중 loss/eval score 실험 추적
- **프롬프트 보강**: Few-shot, 페르소나 추가 → 학습 데이터 재생성

## Phase 3: Serving & Self-Correction
- **vLLM**: 파인튜닝된 모델 고성능 서빙
- **Backend Bridge**: FastAPI 엔드포인트만 로컬 서버로 전환
- **Eval-driven Retry Loop**: Eval 점수 < 7 → Analysis 자동 재시도 (LangGraph 분기)
- **Validator 확장**: 이메일 외 보고서/PR 문서까지 보안 검증

## Phase 4: Benchmarking & Documentation
- **Before/After 비교**: 기본 Llama vs SFT vs SFT+DPO vs GPT-4o 정량 비교표
- **README 아키텍처 다이어그램** + Prompt Design Principles 기술
- **코드 영어화**: 주석, docstring 전체 영문 통일

---

## Git 전략 및 커밋 플랜

### 브랜치 구조
```
main ───────────── 개선 버전 (최종 포트폴리오)
 └── legacy/gpt4o ─ 기존 GPT-4o 전용 버전 (보존)
```

### PR #1: Phase 1 완료 (Data Pipeline)
```
commit: "feat: synthetic data generation pipeline (30 scenarios)"
commit: "feat: evaluation agent with anti-hallucination scoring"
commit: "refactor: add Chain of Thought to analysis agent"
commit: "refactor: convert all comments and docstrings to English"
```

### PR #2: Phase 2 완료 (Model Training)
```
commit: "feat: add SFT training script for Vertex AI"
commit: "feat: add W&B experiment tracking"
commit: "feat: add DPO training with preference pairs"
```

### PR #3: Phase 3 완료 (Serving & Polish)
```
commit: "feat: add eval-driven retry loop in LangGraph"
commit: "feat: add vLLM serving + backend bridge"
commit: "docs: update README with architecture diagram + results"
```

---

## 최종 기술 스택

| 영역 | 스택 |
|:---|:---|
| Multi-Agent | LangGraph, LangChain |
| RAG | ChromaDB |
| 관측성 | LangSmith, W&B |
| 학습 | Vertex AI, QLoRA/PEFT, SFT + DPO |
| 모델 | Llama-3-8B |
| 서빙 | vLLM |
| 배포 | GCP Cloud Run, Docker |
| CI/CD | GitHub Actions |
| 백엔드 | FastAPI, Pydantic |

> [!TIP]
> **CV 한 줄 요약**: "Built an autonomous purchasing system with multi-agent orchestration (LangGraph), fine-tuned Llama-3-8B via SFT + DPO on Vertex AI, and deployed on GCP with vLLM serving and CI/CD."
