# Purchasing Automation Suite (FastAPI + LLM + Docker)

This project is an **enterprise-grade AI solution** designed to automate the end-to-end purchasing workflow. By combining a high-performance **FastAPI** backend with a **LangChain Multi-agent** architecture, it transforms raw inventory data into deep analytical reports and ready-to-use procurement documents.

It demonstrates how **CSV-based stock snapshots, supplier and item history documents, and structured examples** are orchestrated through a multi-agent LLM pipeline to produce:

**1. Purchasing Analysis Report** — structured insights and risk assessments   
**2. Purchase Request Document** — grouped by supplier for approval workflows  
**3. Supplier Email Drafts** — external communication requesting timelines or availability

## Interactive Demo: Manual Context Upload
**Note:** For this demonstration, context files must be uploaded manually. In the production environment, data is automatically retrieved from the internal database.

**Link**: https://soominmyung.com/purchasing-automation

![Purchasing_AI](https://github.com/user-attachments/assets/d2770c1e-c08a-4341-8dee-86885387ae71)

---

## Tech Stack

This project implements a production-ready architecture using the following technologies:

* **Backend**: Python, FastAPI (Asynchronous API, SSE Streaming)
* **AI Framework**: LangChain (Multi-agent Orchestration, Tool Binding)
* **LLM**: OpenAI GPT-4o / GPT-4o-mini
* **Vector Database**: ChromaDB (RAG - Retrieval Augmented Generation)
* **Frontend Interface**: React, Framer (Custom Code Components), TypeScript
* **Data Processing**: Pandas (CSV), PyPDF (Extraction), Python-docx (Word Generation)
* **Infrastructure**: Docker (Containerization for Hugging Face Spaces)

---

## Key Technical Competencies

### 1️⃣ Real-time Event Streaming (SSE)
Beyond a simple request-response model, this system utilizes **Server-Sent Events (SSE)** to provide real-time feedback to the user at every stage of the pipeline: CSV parsing → Item Grouping → AI Analysis → Document Generation.

### 2️⃣ Scalable RAG-based Data Ingestion
* **Bulk Processing**: Developed a scalable API capable of ingesting multiple PDFs or **ZIP archives** for high-volume data training.
* **Automated Metadata Extraction**: Utilizes Regex to automatically identify Supplier names and ItemCodes within documents, mapping them to Vector DB metadata for high-precision retrieval.

### 3️⃣ Multi-Agent Orchestration
Instead of relying on a single prompt, the system orchestrates **five specialized agents**. The Analysis Agent uses Tools to search the knowledge base, while specialized Documentation Agents transform those findings into various professional formats.

### 4️⃣ Ephemeral Environment Optimization (Memory-first)
Designed specifically for serverless/ephemeral environments like Hugging Face Spaces, the system supports a **memory-first approach** where documents are generated as bytes and encoded to **Base64** for instant client-side download, bypassing persistent disk requirements.

### 5️⃣ Production Security & Usage Control (Hybrid Auth)
Implemented a robust security layer to prevent API abuse and cost overruns:
*   **Header-based Authentication**: Secure access via `X-API-Key` validation for all API endpoints.
*   **Intelligent Rate Limiting**: IP-based daily request limits (e.g., max 5/day) to ensure availability and prevent malicious exploitation.
*   **Environment-level Secret Management**: Zero hardcoded credentials; fully driven by external secrets.

---

## Project Structure

* **main.py**: FastAPI Entry point & CORS configuration
* **routers/**: API Layer (Pipeline, Ingest, Output)
* **services/**: Business Logic (AI Agents, Vector Store, Security, Grouping)
* **utils/**: Utilities (CSV Parsing, PDF Extraction, Word Generation)
* **schemas.py**: Data Models & Validation (Pydantic)
* **config.py**: Environment Variables & System Settings
* **.env**: Local secret management (hidden from version control)
* **docs/**: Sample documents for testing

---

## Key API Endpoints

* **POST /api/run/stream**: Upload inventory CSV and execute real-time streaming analysis.
* **POST /api/ingest/{type}/zip**: Batch-learn historical documents via ZIP upload.
* **GET /api/output/download**: Download generated reports and email drafts.

