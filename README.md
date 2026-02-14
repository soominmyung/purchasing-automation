# Purchasing Automation Suite (FastAPI + LLM + Docker)

This project is an **enterprise-grade AI solution** designed to automate the end-to-end purchasing workflow. By combining a high-performance **FastAPI** backend with a **LangChain Multi-agent** architecture, it transforms raw inventory data into deep analytical reports and ready-to-use procurement documents.

It demonstrates how **CSV-based stock snapshots, supplier and item history documents, and structured examples** are orchestrated through a multi-agent LLM pipeline to produce:

**1. Purchasing Analysis Report** — structured insights and risk assessments   
**2. Purchase Request Document** — grouped by supplier for approval workflows  
**3. Supplier Email Drafts** — external communication requesting timelines or availability

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

## Tech Stack

This project implements a production-ready architecture using the following technologies:

* **Backend**: Python, FastAPI (Asynchronous API, SSE Streaming)
* **AI Framework**: LangChain (Multi-agent Orchestration, Tool Binding)
* **LLM**: OpenAI GPT-4o / GPT-4o-mini
* **Vector Database**: ChromaDB (RAG - Retrieval Augmented Generation)
* **Frontend Interface**: React, Framer (Custom Code Components), TypeScript
* **Data Processing**: Pandas (CSV), PyPDF (Extraction), Python-docx (Word Generation)
* **Infrastructure**: **GCP Cloud Run** (Serverless Container Platform), **Google Artifact Registry**
* **CI/CD**: **GitHub Actions** (Automated build and deployment pipeline)

---

## Key Technical Competencies

### 1️⃣ Real-time Event Streaming (SSE)
Beyond a simple request-response model, this system utilizes **Server-Sent Events (SSE)** to provide real-time feedback to the user at every stage of the pipeline: CSV parsing → Item Grouping → AI Analysis → Document Generation.

### 2️⃣ Scalable RAG-based Data Ingestion
* **Bulk Processing**: Developed a scalable API capable of ingesting multiple PDFs or **ZIP archives** for high-volume data training.
* **Robust Folder Traversal**: Implemented an advanced recursive folder traversal algorithm in React/TypeScript to handle large-scale document uploads from local directories.
* **Automated Metadata Extraction**: Utilizes Regex to automatically identify Supplier names and ItemCodes within documents, mapping them to Vector DB metadata for high-precision retrieval.

### 3️⃣ Multi-Agent Orchestration
Instead of relying on a single prompt, the system orchestrates **five specialized agents**. The Analysis Agent uses Tools to search the knowledge base, while specialized Documentation Agents transform those findings into various professional formats.

### 4️⃣ Cloud-Native Architecture (GCP Cloud Run)
Optimized for serverless deployment on **Google Cloud Platform**, utilizing a scale-to-zero model for cost efficiency. The system supports a **memory-first approach** where documents are generated as bytes and encoded to **Base64** for instant client-side download.

### 6️⃣ LLMOps & Observability
Integrated **LangSmith** for comprehensive telemetry across the multi-agent pipeline. This enables real-time monitoring of agent token consumption, execution latency, and prompt tracing to ensure cost-efficiency and mitigate LLM hallucination risks in a production environment.

### 7️⃣ Production Security & Error Handling
Implemented a robust security layer to protect resources and ensure reliability:
*   **Header-based Authentication**: Secure access via `X-API-Key` validation for all API endpoints.
*   **Intelligent Rate Limiting**: Selective IP-based daily limits; allowing unlimited file ingestion while strictly controlling expensive LLM-powered pipeline executions.
*   **Permissive CORS for Portfolio**: Configured for seamless interaction with Framer frontend while maintaining backend security.
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
* **docs/**: Project documentation and **Sample Dataset (Examples.zip)**
* **.env.example**: Template for environment variables
* **.github/workflows/deploy.yml**: CI/CD pipeline configuration
* **Dockerfile**: Containerization configuration for Cloud Run (Port 8080)

---

## Key API Endpoints

* **POST /api/run/stream**: Upload inventory CSV and execute real-time streaming analysis.
* **POST /api/ingest/{type}**: Batch-learn historical documents via PDF or folder upload (CORS-friendly).
* **POST /api/run/embed**: Generate and return documents as Base64 encoded strings for client-side download.

