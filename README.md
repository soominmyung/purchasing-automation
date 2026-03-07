# Purchasing Automation Suite (FastAPI + LangGraph + Docker)

This project is an **enterprise-grade AI solution** designed to automate the end-to-end purchasing workflow. By combining a high-performance **FastAPI** backend with a **LangGraph Multi-agent** architecture, it transforms raw inventory data into deep analytical reports and ready-to-use procurement documents.

It demonstrates how **CSV-based stock snapshots, supplier and item history documents, and structured examples** are orchestrated through a multi-agent LLM pipeline to produce:

**1. Purchasing Analysis Report** — structured insights and risk assessments   
**2. AI Quality Evaluation Report** — critical review of analysis logic and data adherence  
**3. Purchase Request Document** — grouped by supplier for approval workflows  
**4. Supplier Email Drafts** — external communication requesting timelines or availability

---

## 🚀 Architecture Evolution: From n8n to FastAPI
This project originally started as a low-code automation workflow in **n8n**. To ensure enterprise-grade performance, complex logic handling, and seamless containerized deployment, it was **re-engineered** and refactored into a **pure Python FastAPI** architecture.

This evolution demonstrates:
*   **Rapid Prototyping**: Leveraging low-code tools (n8n) for initial workflow validation.
*   **Scalable Engineering**: Translating node-based logic into high-performance, asynchronous Python code.
*   **Production Readiness**: Moving from internal automation to a public-facing, secured, and containerized API suite.

---

## Interactive Demo: Manual Context Upload
**Note:** For this demonstration, context files must be uploaded manually. In the production environment, data is automatically retrieved from the internal database.

**Link**: https://soominmyung.com/purchasing-automation

![Purchasing_AI](https://github.com/user-attachments/assets/d2770c1e-c08a-4341-8dee-86885387ae71)

---

## Fine-Tuning Research: SFT + DPO on Llama-3-8B

To validate replacing the GPT-4o Analysis Agent with a self-hosted model, a two-stage fine-tuning pipeline was built on Vertex AI using GPT-4o knowledge distillation.

### Stage 1: SFT (Supervised Fine-Tuning)

Fine-tuned **Llama-3-8B** via QLoRA (4-bit quantization + LoRA adapters) using [Unsloth](https://github.com/unslothai/unsloth).

| Epoch | Loss   |
|-------|--------|
| 1     | 1.141  |
| 2     | 0.844  |
| 3     | 0.582  |
| 4     | 0.498  |
| 5     | **0.409** |

- **Runtime**: 367s on a single NVIDIA Tesla T4 (Vertex AI Custom Training)
- **Trainable parameters**: ~41M / 8B (0.51% — LoRA rank 16)
- **Dataset**: 12 teacher examples distilled from GPT-4o
- **Artifact**: `gs://purchasing-automation-models/sft-runs/lora_adapter/`
- **Experiment tracking**: [W&B — purchasing-automation-sft](https://wandb.ai/msm1640-/purchasing-automation-sft)

### Stage 2: DPO (Direct Preference Optimization)

Generated 25 preference pairs (chosen: GPT-4o output, rejected: SFT output) and ran DPO alignment training on the SFT adapter.

- **Runtime**: ~14 min on NVIDIA Tesla T4
- **Artifact**: `gs://purchasing-automation-models/dpo-runs/lora_adapter/`
- **Experiment tracking**: [W&B — purchasing-automation-dpo](https://wandb.ai/msm1640-/purchasing-automation-dpo)

### Before/After Evaluation

Evaluated on 5 holdout examples using GPT-4o-as-judge (data_accuracy + reasoning_quality, scored 1–10 against GPT-4o ground truth as reference).

| Model | Avg Score | JSON Valid |
|-------|-----------|-----------|
| Base Llama-3-8B (no fine-tuning) | 0.0 / 10 | 0% |
| **Llama-3-8B SFT** | **9.3 / 10** | **100%** |
| Llama-3-8B SFT+DPO | 7.4 / 10 | 100% |
| GPT-4o (reference ceiling) | 10.0 / 10 | 100% |

- SFT reached **93% of GPT-4o quality** — validates local model viability
- DPO regressed (9.3 → 7.4): only 7 training pairs available; model retained data accuracy (8–9) but lost reasoning depth — a known limitation of DPO on small datasets
- Eval results: `gs://purchasing-automation-models/eval-results/eval_dpo_20260306_0549.json`

### Infrastructure

| Component | Detail |
|-----------|--------|
| Base image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` |
| SFT framework | Unsloth 2026.3 + TRL SFTTrainer |
| DPO framework | Unsloth + PatchDPOTrainer() + TRL DPOTrainer |
| GPU | NVIDIA Tesla T4 (16GB VRAM) |
| Build | GCP Cloud Build (E2_HIGHCPU_8, ~4 min) |
| Artifact storage | Google Cloud Storage |
| Experiment tracking | Weights & Biases (W&B) |

---

## Tech Stack

This project implements a production-ready architecture using the following technologies:

* **Backend**: Python, FastAPI (Asynchronous API, SSE Streaming)
* **AI Framework**: LangGraph (Multi-agent Orchestration), LangChain (Tool Binding), LangSmith (LLMOps)
* **LLM**: OpenAI GPT-4o / GPT-4o-mini → Fine-tuned Llama-3-8B (QLoRA)
* **Fine-tuning**: Unsloth + TRL SFTTrainer/DPOTrainer, QLoRA (4-bit), Vertex AI Custom Training, W&B
* **Vector Database**: ChromaDB (RAG - Retrieval Augmented Generation)
* **Frontend Interface**: React, Framer (Custom Code Components), TypeScript
* **Data Processing**: Pandas (CSV), PyPDF (Extraction), Python-docx (Word Generation)
* **Infrastructure**: **GCP Cloud Run** (Serverless), **Vertex AI** (Custom Training), **Google Artifact Registry**
* **CI/CD**: **GitHub Actions** (Automated build and deployment pipeline)

---

## Key Technical Competencies

### 1️⃣ Intelligent Inventory-to-Procurement Orchestration
Beyond simple CSV parsing, the system implements complex business logic to transform raw data into actionable intelligence:
*   **Contextual Supplier Grouping**: Automatically clusters hundreds of individual SKU requirements into consolidated supplier-based batches, reducing procurement overhead.
*   **Predictive Lead-Time Reasoning**: Analyzes `WksToOOS` (Weeks to Out of Stock) and `CurrentStock` to prioritize urgent replenishments, ensuring supply chain continuity.
*   **Dynamic Document Routing**: Orchestrates different document types (Analysis vs. PR vs. Email) based on the specific risk score and history of each supplier group.

### 2️⃣ Real-time Event Streaming (SSE)
Beyond a simple request-response model, this system utilizes **Server-Sent Events (SSE)** to provide real-time feedback to the user at every stage of the pipeline: CSV parsing → Item Grouping → AI Analysis → Document Generation.

### 3️⃣ Scalable RAG-based Data Ingestion
* **Bulk Processing**: Developed a scalable API capable of ingesting multiple PDFs or **ZIP archives** for high-volume data training.
* **Robust Folder Traversal**: Implemented an advanced recursive folder traversal algorithm in React/TypeScript to handle large-scale document uploads from local directories.
* **Automated Metadata Extraction**: Utilizes Regex to automatically identify Supplier names and ItemCodes within documents, mapping them to Vector DB metadata for high-precision retrieval.

### 4️⃣ Agentic Workflow with LangGraph
Evolved from a linear pipeline to a **state-based agentic graph** using **LangGraph**. This architecture supports complex, non-linear workflows including:
*   **Self-Correction Loop**: A dedicated **Validator Node** audits generated emails for sensitive internal data. If a leak is detected, the graph automatically loops back for a rewrite.
*   **AI Quality Evaluator**: An independent agent node that critiques the primary analysis, scoring it across data accuracy, reasoning logic, and operational readiness to ensure boardroom-ready output.
*   **State Management**: Orchestrates shared state across specialized agents, ensuring consistency throughout the analysis.

### 5️⃣ Cloud-Native Architecture (GCP Cloud Run)
Optimized for serverless deployment on **Google Cloud Platform**, utilizing a scale-to-zero model for cost efficiency. The system supports a **memory-first approach** where documents are generated as bytes and encoded to **Base64** for instant client-side download.

### 6️⃣ Automated CI/CD Pipeline (GitHub Actions)
Fully automated deployment pipeline using **GitHub Actions**. Every push to the `main` branch triggers an automated build, containerization (Docker), and deployment to GCP Cloud Run, ensuring stable and repeatable delivery.

### 7️⃣ Production Security & LLMOps (LangSmith)
Implemented a robust observability and security layer to ensure reliability:
*   **LangSmith Integration**: Full telemetry tracing for every node in the graph, providing deep visibility into token consumption, latency, and prompt performance.
*   **Manual Instrumentation**: Utilizes `@traceable` for end-to-end tracing of custom business logic, ensuring that even non-LangChain code is fully visible in the observability stack.
*   **Intelligent Security**: Combines **Header-based Authentication** (`X-API-Key`) with **Stricter API Key Hardening** (Regex-based sanitization) to prevent environment-specific injection errors.
*   **Robust Error Handling**: Graceful management of OpenAI API rate limits and data validation errors.

---

## 💻 Local Setup & Quick Start

Get the project running on your local machine in minutes using Docker.

### 1. Clone the repository
```bash
git clone https://github.com/soominmyung/purchasing-automation.git
cd purchasing-automation
```

### 2. Set environment variables
```bash
cp .env.example .env
```
Open the `.env` file and add your `OPENAI_API_KEY`. You can also set a custom `API_ACCESS_TOKEN` for header-based authentication.

### 3. Run with Docker
```bash
# Build the image
docker build -t purchasing-ai .

# Run the container
docker run -p 8080:8080 --env-file .env purchasing-ai
```
The API will be available at `http://localhost:8080`. You can explore the interactive docs at `http://localhost:8080/docs`.

---

## Project Structure

* **main.py**: FastAPI Entry point & Environment configuration
* **routers/**: API Layer (Pipeline, Ingest, Output)
* **services/**: Business Logic (AI Agents, Vector Store, Security, Grouping)
* **utils/**: Utilities (CSV Parsing, PDF Extraction, Word Generation)
* **scripts/**: Fine-tuning pipeline (`train_sft.py`, `train_dpo.py`, `generate_dpo_pairs.py`, `eval_dpo.py`)
* **training_data/**: GPT-4o distilled teacher dataset (JSONL)
* **docs/**: Project documentation, walkthrough, and **Sample Dataset (Examples.zip)**
* **.env.example**: Template for environment variables
* **.github/workflows/deploy.yml**: CI/CD pipeline configuration
* **Dockerfile**: Containerization configuration for Cloud Run (Port 8080)

---

## Key API Endpoints

* **POST /api/run/stream**: Upload inventory CSV and execute real-time streaming analysis.
* **POST /api/ingest/{type}**: Batch-learn historical documents via PDF or folder upload (CORS-friendly).
* **POST /api/run/embed**: Generate and return documents as Base64 encoded strings for client-side download.

