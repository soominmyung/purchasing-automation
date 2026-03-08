# Purchasing Automation Suite (FastAPI + LangGraph + Vertex AI)

---

## Project Purpose

An enterprise-grade AI pipeline built to eliminate the manual effort purchasing teams spend reviewing and documenting hundreds of SKU-level inventory records.

Given a single CSV inventory snapshot, the system automatically generates supplier-level analysis reports, purchase request documents (PR), and supplier email drafts end-to-end. Designed for production use, it includes security hardening, real-time streaming feedback, and a full CI/CD deployment pipeline.

In production, inventory input and context data (supplier history, item records) are automatically retrieved from an internal database. The portfolio demo simulates this by allowing manual drag-and-drop upload of the same data.

Additionally, a parallel research track validated replacing the GPT-4o Analysis Agent with a **self-hosted fine-tuned model (Llama-3-8B)** via SFT + DPO.

---

## Summary

| Item | Detail |
|:---|:---|
| Period | Jul 2025 – Mar 2026 |
| Role | Solo project (architecture, development, and deployment) |
| Starting point | n8n low-code prototype → re-engineered into FastAPI + LangGraph |
| Deployment | GCP Cloud Run (serverless, auto-scaling) |
| Demo | https://soominmyung.com/purchasing-automation |

---

## Tech Stack

| Area | Technologies |
|:---|:---|
| Backend | Python, FastAPI (async, SSE streaming) |
| AI Orchestration | LangGraph (state-based multi-agent), LangChain |
| LLMOps / Observability | LangSmith (`@traceable` manual instrumentation) |
| LLM | GPT-4o / GPT-4o-mini → Llama-3-8B (QLoRA fine-tuned) |
| Fine-Tuning | Unsloth + TRL SFTTrainer / DPOTrainer, Vertex AI Custom Training |
| Experiment Tracking | Weights & Biases (W&B) |
| Vector Database | ChromaDB (RAG) |
| Frontend | React / TypeScript, Framer Custom Code Component |
| Data Processing | Pandas, PyPDF, python-docx |
| Infrastructure | GCP Cloud Run, Vertex AI, Google Artifact Registry, Cloud Storage |
| CI/CD | GitHub Actions (push to main → automated build & deploy) |
| Container | Docker |

---

## Architecture

### 1. Multi-Agent Pipeline (LangGraph)

Five specialized agents connected via a LangGraph state graph.

```
[Input] Inventory CSV + Supplier context
        ↳ Production: auto-retrieved from internal DB
        ↳ Demo: manually uploaded via drag-and-drop
  → Supplier Grouping  (lead-time-based priority calculation)
  → Analysis Agent     — inventory risk analysis + RAG context retrieval
  → Evaluator Agent    — automated quality audit (logic & data consistency)
  → PR Draft Agent     — purchase request draft generation
  → PR Doc Agent       — formal procurement document completion
  → Email Agent        — supplier-facing email drafts
  → Validator Node     — sensitive data leak detection → auto-rewrite loop
```

### 2. Real-Time Streaming (SSE)

Rather than a simple request-response model, the system uses **Server-Sent Events (SSE)** to stream live progress updates to the client at every stage of the pipeline — from CSV parsing through to document generation.

### 3. RAG-Based Knowledge Base

Historical supplier records and item history PDFs are embedded into ChromaDB and automatically referenced during analysis.

- **Production**: Context documents are pre-registered in the internal database and automatically retrieved at pipeline runtime.
- **Demo**: Context is provided manually via ZIP file upload or folder drag-and-drop.

### 4. Security & LLMOps

- Header-based API Key authentication + IP-level rate limiting
- LangSmith tracing across all agent nodes — token consumption, latency, and prompt performance
- Sensitive data leak Validator — if internal data is detected in an email draft, the graph automatically loops back for a rewrite

### 5. SFT + DPO Fine-Tuning Research

A two-stage fine-tuning pipeline was built to validate replacing GPT-4o with a self-hosted model.

**Stage 1 — Supervised Fine-Tuning (SFT)**

- Model: Llama-3-8B + QLoRA (rank 16, 0.51% trainable parameters)
- Dataset: 12 high-quality teacher examples distilled from GPT-4o (with Chain-of-Thought)
- Training: Vertex AI Tesla T4, 5 epochs, 367 seconds
- Loss: 1.14 → 0.41

**Stage 2 — Direct Preference Optimization (DPO)**

- Preference pairs: GPT-4o output (Chosen) vs. SFT output (Rejected), 25 pairs
- Training: Vertex AI Tesla T4, 3 epochs, ~14 minutes
- Implementation: Unsloth `PatchDPOTrainer()` + TRL DPOTrainer

---

## Results

### Before / After Evaluation (GPT-4o-as-judge, 5 holdout examples)

| Model | Avg Score | JSON Valid | Notes |
|:---|:---:|:---:|:---|
| Base Llama-3-8B | 0.0 / 10 | 0% | Free-form text output → JSON parsing failure |
| **Llama-3-8B SFT** | **9.3 / 10** | **100%** | 93% of GPT-4o quality achieved |
| Llama-3-8B SFT+DPO | 7.4 / 10 | 100% | DPO regression on small dataset (7 valid pairs) |
| GPT-4o (reference ceiling) | 10.0 / 10 | 100% | Reference only (self-referential) |

**Key finding**: SFT alone reached 93% of GPT-4o quality — empirically validating the feasibility of replacing the cloud LLM with a self-hosted model.

DPO regression (9.3 → 7.4) is a known limitation of preference optimization on small datasets (only 7 valid pairs); data accuracy was preserved (8–9/10) but reasoning depth declined. Addressable with more training pairs.

### Key Achievements

- **Purchasing analysis and reporting fully automated** — manual documentation replaced by AI pipeline end-to-end
- **Advanced procurement analysis enabled** by integrating multi-layered context (supplier history, inventory trends, lead times) via RAG — moving beyond simple stock checks to risk-prioritized, supply-chain-aware decision support
- **Data-driven purchasing culture** — supports data-backed procurement and replenishment decisions across the organization
- Re-engineered n8n low-code prototype into a production FastAPI + LangGraph enterprise architecture
- CI/CD pipeline — every push to main triggers automated build and deployment to GCP Cloud Run
- SSE streaming, Validator auto-rewrite loop, and RAG-based knowledge retrieval fully integrated
