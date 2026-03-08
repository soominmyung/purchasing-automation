# 구매 자동화 AI 시스템 (Purchasing Automation Suite)

---

## 프로젝트 목적

기업 구매 담당자가 수십~수백 개 SKU의 재고 데이터를 수작업으로 검토하고 문서화하는 데 소요되는 시간을 제거하기 위해 개발된 엔터프라이즈급 AI 파이프라인입니다.

CSV 형식의 재고 스냅샷 한 장이 입력되면, 공급업체별 분석 보고서 · 구매 요청서(PR) · 공급업체 이메일 초안까지 전 과정을 자동으로 생성합니다. 실무 사용을 전제로 설계되어 보안, 스트리밍 피드백, CI/CD 배포까지 포함합니다.

현업 운영 환경에서는 재고 인풋과 공급업체 이력 등 컨텍스트 데이터를 내부 데이터베이스에서 자동으로 조회합니다. 포트폴리오 데모에서는 해당 데이터를 드래그앤드롭으로 직접 업로드하는 방식으로 시연합니다.

추가로, GPT-4o 기반 파이프라인을 **자체 파인튜닝 모델(Llama-3-8B)**로 대체 가능한지 검증하는 SFT+DPO 연구를 병행하였습니다.

---

## 요약

| 항목 | 내용 |
|:---|:---|
| 기간 | 2025.07 ~ 2026.03 |
| 역할 | 개인 프로젝트 (설계 · 개발 · 배포 전담) |
| 시작점 | n8n 저코드 프로토타입 → FastAPI + LangGraph 전면 재설계 |
| 배포 | GCP Cloud Run (서버리스, 자동 스케일링) |
| 데모 | https://soominmyung.com/purchasing-automation |

---

## 기술 스택

| 영역 | 기술 |
|:---|:---|
| 백엔드 | Python, FastAPI (비동기, SSE 스트리밍) |
| AI 오케스트레이션 | LangGraph (상태 기반 멀티 에이전트), LangChain |
| LLMOps / 관측성 | LangSmith (`@traceable` 수동 계측) |
| LLM | GPT-4o / GPT-4o-mini → Llama-3-8B (QLoRA 파인튜닝) |
| 파인튜닝 | Unsloth + TRL SFTTrainer / DPOTrainer, Vertex AI Custom Training |
| 실험 추적 | Weights & Biases (W&B) |
| 벡터 DB | ChromaDB (RAG) |
| 프론트엔드 | React / TypeScript, Framer Custom Code Component |
| 데이터 처리 | Pandas, PyPDF, python-docx |
| 인프라 | GCP Cloud Run, Vertex AI, Google Artifact Registry, Cloud Storage |
| CI/CD | GitHub Actions (main 브랜치 push → 자동 빌드 · 배포) |
| 컨테이너 | Docker |

---

## 세부 구성

### 1. 멀티 에이전트 파이프라인 (LangGraph)

5개의 특화 에이전트가 LangGraph 상태 그래프로 연결됩니다.

```
[인풋] 재고 CSV + 공급업체 이력 컨텍스트
       ↳ 현업: 내부 DB에서 자동 조회
       ↳ 데모: 드래그앤드롭으로 수동 업로드
  → 공급업체별 그룹화 (리드타임 기반 우선순위 계산)
  → Analysis Agent   — 재고 위험도 분석 + RAG 컨텍스트 검색
  → Evaluator Agent  — 분석 품질 자동 심사 (논리·데이터 정합성)
  → PR Draft Agent   — 구매 요청서 초안 생성
  → PR Doc Agent     — 공식 구매 문서 완성
  → Email Agent      — 공급업체 발송용 이메일 초안
  → Validator Node   — 내부 정보 유출 탐지 → 자동 재작성 루프
```

### 2. 실시간 스트리밍 (SSE)

단순 요청-응답이 아닌 Server-Sent Events로 각 단계의 진행 상황을 클라이언트에 실시간 전송합니다. 사용자는 파이프라인 전 과정을 즉시 확인할 수 있습니다.

### 3. RAG 기반 지식 베이스

과거 공급업체 이력 · 아이템 히스토리 PDF를 ChromaDB에 임베딩하여 분석 시 자동 참조합니다.

- **현업 환경**: 컨텍스트 문서가 내부 데이터베이스에 사전 등록되어 파이프라인 실행 시 자동으로 조회됩니다.
- **데모 환경**: ZIP 파일 일괄 업로드 또는 폴더 드래그앤드롭으로 컨텍스트를 수동 제공합니다.

### 4. 보안 및 LLMOps

- Header 기반 API Key 인증 + IP별 Rate Limiting
- LangSmith로 모든 에이전트 노드의 토큰 소비 · 지연 시간 · 프롬프트 성능 추적
- 민감 정보 유출 탐지 Validator — 이메일 초안에 내부 데이터가 포함될 경우 자동 재생성

### 5. SFT + DPO 파인튜닝 연구

GPT-4o를 자체 호스팅 모델로 대체할 수 있는지 검증하기 위한 2단계 파인튜닝 파이프라인을 구축했습니다.

**Stage 1 — SFT (Supervised Fine-Tuning)**

- 모델: Llama-3-8B + QLoRA (rank 16, 학습 파라미터 0.51%)
- 데이터: GPT-4o에서 지식 증류한 12개 고품질 예제 (CoT 포함)
- 학습: Vertex AI Tesla T4, 5 epochs, 367초
- Loss: 1.14 → 0.41

**Stage 2 — DPO (Direct Preference Optimization)**

- Preference pair: GPT-4o 출력(Chosen) vs SFT 출력(Rejected), 25쌍
- 학습: Vertex AI Tesla T4, 3 epochs, ~14분
- 구현: Unsloth `PatchDPOTrainer()` + TRL DPOTrainer

---

## 결과

### Before / After 평가 (GPT-4o-as-judge, 홀드아웃 5개)

| 모델 | 평균 점수 | JSON 유효율 | 비고 |
|:---|:---:|:---:|:---|
| Base Llama-3-8B | 0.0 / 10 | 0% | 자유형식 텍스트 출력 → JSON 파싱 실패 |
| **Llama-3-8B SFT** | **9.3 / 10** | **100%** | GPT-4o 품질의 93% 달성 |
| Llama-3-8B SFT+DPO | 7.4 / 10 | 100% | 소량 데이터(7쌍 유효) DPO 회귀 |
| GPT-4o (기준 상한) | 10.0 / 10 | 100% | 레퍼런스 (자기 참조) |

**핵심 결론**: SFT만으로도 GPT-4o 대비 93% 수준의 품질에 도달 — 자체 호스팅 모델로의 전환 가능성을 실증적으로 검증.

DPO 회귀(9.3→7.4)는 유효 학습 쌍 부족(7쌍)에 기인한 소량 데이터 한계이며, 데이터 규모 확장 시 개선 가능한 알려진 현상입니다.

### 아키텍처 성과

- 구매 기반 분석 및 보고서 생성 업무 자동화 — 수작업 문서화 시간을 AI 파이프라인으로 대체
- 공급업체 이력·재고 추이·리드타임 등 다층적 컨텍스트를 RAG로 통합함으로써, 단순 재고 조회 수준을 넘어 위험도 우선순위·공급망 연속성까지 고려한 고도화된 구매 분석이 가능해짐
- 데이터 기반 구매·재고 충당 업무 지원 및 데이터 중심 업무 문화 구축 기여
- n8n 저코드 프로토타입을 FastAPI + LangGraph 엔터프라이즈 아키텍처로 전환
- CI/CD 파이프라인 구축 — main 브랜치 push 시 자동 빌드·배포 (GCP Cloud Run)
- SSE 스트리밍, Validator 자동 재작성 루프, RAG 기반 지식 검색 통합
