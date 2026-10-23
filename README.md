# Adaptive Hybrid RAG

Adaptive Hybrid RAG is a research-oriented Retrieval-Augmented Generation system that combines BM25 lexical retrieval, dense semantic retrieval, and query-dependent adaptive fusion. The project is designed to support both academic benchmarking and a working application for uploading documents, asking questions, and inspecting retrieval behavior in real time.

This repository contains the full project structure for a final-year submission: backend services, frontend UI, evaluation scripts, trained model artifacts, and supporting research files.

## Project overview

The system addresses a common hybrid-RAG limitation: a single fixed weighting between BM25 and dense retrieval does not work well for all query types. Keyword-heavy queries often favor lexical retrieval, while semantic or paraphrased queries favor dense retrieval.

The implementation uses a learned predictor to estimate a query-specific BM25 weight and then computes the dense weight as:

- predicted_dense_weight = 1 - predicted_bm25_weight

The final fused score combines independently normalized BM25 and dense retrieval scores. This is implemented in the retrieval and fusion services used by both the general RAG endpoint and the user-document RAG workflow.

## Main features

- BM25 lexical retrieval using rank-bm25
- Dense semantic retrieval using SentenceTransformers and ChromaDB
- Adaptive score fusion with query-dependent BM25/dense weighting
- Retrieval feature extraction for V4 Rich query analysis
- Local LLM generation via Ollama using qwen2.5:7b
- User uploads and user-specific document indexing
- Persistent authentication, chat history, and conversation management
- Research dashboard and retrieval benchmark scripts
- Domain-aware evaluation and adaptive-weight analysis

## Technology stack

Frontend
- React 19
- Vite 8
- JavaScript / JSX
- CSS
- Lucide React

Backend
- Python 3.10+
- FastAPI
- SQLite
- ChromaDB
- rank-bm25
- SentenceTransformers
- scikit-learn
- PyMuPDF / pypdf / python-docx
- Ollama

## Repository structure

```text
adaptive-hybrid-rag/
├── backend/
│   ├── api/
│   ├── config/
│   ├── evaluation/
│   ├── models/
│   ├── services/
│   ├── tests/
│   ├── training/
│   ├── app.py
│   ├── main.py
│   ├── requirements.txt
│   └── sample_user_document*.txt
├── data/
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   └── src/
├── chroma_db/
├── bm25_cache/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
├── LICENSE
└── ...
```

## Prerequisites

The project was developed and validated in a local Windows environment with the following expected setup:

- Git
- Python 3.10 or newer
- Node.js 18+ and npm
- Ollama installed locally
- A local model cache for sentence-transformers and a compatible CPU/GPU environment

This project uses:

- `all-MiniLM-L6-v2` as the embedding model
- `qwen2.5:7b` as the local LLM model in Ollama

## Clone and setup

```powershell
git clone https://github.com/Parasfunde21/adaptive-hybrid-rag.git
cd adaptive-hybrid-rag
```

## Python environment

Create and activate a local virtual environment in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

If you want a single root-level install entrypoint, the repository root `requirements.txt` now points to the backend dependency file:

```powershell
pip install -r requirements.txt
```

## Frontend setup

Use the lockfile for reproducible installation:

```powershell
cd frontend
npm ci
```

If `npm ci` is unavailable because the lockfile has changed, use:

```powershell
npm install
```

## Environment configuration

Create a local environment file from the example template:

```powershell
copy .env.example .env
```

Use the values in `.env.example` and do not commit real secrets or credentials. The frontend uses `VITE_API_BASE_URL` to reach the FastAPI server when it is not running on the default address.

## Ollama setup and model pull

Install Ollama from the official source for your platform and start the local service. Then pull the required model:

```powershell
ollama pull qwen2.5:7b
```

If Ollama is running locally on its standard host/port, the backend code can call it without additional configuration. An optional override is permitted through `OLLAMA_HOST` in the environment if needed.

## Embedding-model cache

The project expects the `all-MiniLM-L6-v2` model to be available in the local sentence-transformers cache.

If the model is not already cached, a one-time download is required from Hugging Face. In a network-enabled environment, use:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

This writes the model into the local cache directory configured by the sentence-transformers tooling. The runtime also sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the embedding loader to ensure offline local-cache behavior.

## Required model artifacts

The adaptive predictor relies on local trained artifacts under `backend/models`.

Expected files include:

- `backend/models/adaptive_weight_model_v4_rich.pkl`
- `backend/models/adaptive_weight_model_v4.pkl`
- other historical V3/V4 model files if you want to compare research variants

If a model file is missing, regenerate it from the training pipeline.

Example:

```powershell
cd backend
python training/train_v4_rich_model.py
```

This expects the feature training data to exist under `backend/evaluation/v4_analysis/` and writes the trained model back to `backend/models`.

## Database and runtime initialization

The backend expects a ChromaDB vector store, a BM25 cache, and a local SQLite database to be initialized or created on first run.

From the backend directory:

```powershell
cd backend
python app.py
```

This starts the FastAPI app and loads the ChromaDB collection and the embedding model. If the vector collection is empty, the project will fail fast with a clear error message rather than silently continuing.

The BM25 index is built lazily from ChromaDB into the `bm25_cache/` folder when it is missing or stale. Local SQLite databases and other generated runtime files are intentionally excluded from version control.

## Vector database and BM25 cache setup

The ChromaDB collection is configured in `backend/config/settings.py` and points to the `chroma_db/` folder at the repository root.

The BM25 cache is generated under:

```text
bm25_cache/
```

If this is your first local run, the system will create and populate the cache automatically when the backend initializes. The cache is versioned by collection and is rebuilt when the ChromaDB collection changes.

## Startup commands

Start the backend in one terminal:

```powershell
cd adaptive-hybrid-rag
.\venv\Scripts\Activate.ps1
cd backend
python app.py
```

The backend uses FastAPI and is usually served at:

- `http://127.0.0.1:8000`
- health endpoint: `http://127.0.0.1:8000/health`

Start the frontend in a second terminal:

```powershell
cd adaptive-hybrid-rag\frontend
npm ci
npm run dev -- --host 0.0.0.0
```

The Vite dev server typically runs on:

- `http://localhost:5173`

The frontend is configured to use `VITE_API_BASE_URL` with a fallback of `http://127.0.0.1:8000`.

## Smoke test

A basic end-to-end workflow is:

1. Start the backend and frontend.
2. Open the frontend in the browser.
3. Create an account or log in if authentication is enabled.
4. Upload a document or use the sample text in `backend/sample_user_document.txt`.
5. Ask a question such as:
   - `What is adaptive hybrid retrieval?`
   - `How does BM25 compare with dense retrieval in this project?`
6. Confirm that the response contains a grounded answer, retrieval results, and latency metrics.

You can also test the API directly:

```powershell
curl http://127.0.0.1:8000/health
```

For question submission:

```powershell
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is adaptive hybrid retrieval?"}'
```

## Evaluation commands

Benchmark scripts are available under `backend/evaluation`.

To run the current V4 benchmark:

```powershell
cd backend
python -m evaluation.benchmark_v4_rich_actual
```

Other relevant scripts include:

```powershell
python -m evaluation.benchmark_v3
python -m evaluation.benchmark_v4_rich
python evaluation/build_v4_rich_features.py
```

The evaluation scripts rely on the local benchmark datasets and feature-generation outputs already included in the repo under `backend/evaluation/` and `backend/training/`.

## Common errors and troubleshooting

### `ModuleNotFoundError` or package import errors

Install the backend dependencies again inside the virtual environment:

```powershell
pip install -r backend\requirements.txt
```

### ChromaDB collection is empty

The system expects a populated vector collection. Initialize or regenerate the local ChromaDB content before running the app.

### BM25 cache not found

The app automatically builds the BM25 cache the first time it runs. If the cache is stale or corrupted, delete the local `bm25_cache/` directory and restart the backend.

### Missing model artifact

If the adaptive predictor model is missing, recreate it using the training pipeline:

```powershell
cd backend
python training/train_v4_rich_model.py
```

### Embedding model missing from cache

Populate the local sentence-transformers cache once from a network-enabled environment using the command above. After that, the app can run in offline mode.

### Ollama model not found

Pull the model explicitly:

```powershell
ollama pull qwen2.5:7b
```

### Frontend cannot reach backend

Confirm:

- backend is running on port 8000
- frontend `VITE_API_BASE_URL` points to the correct host
- no firewall or port conflict is blocking local traffic

## Notes on generated files and local artifacts

This project intentionally excludes large local artifacts from version control. Some of them are generated on first run and can be very large:

- ChromaDB vector indexes under `chroma_db/`
- BM25 cache under `bm25_cache/`
- SQLite databases under `data/` or `backend/data/`
- model downloads and caches for Hugging Face / sentence-transformers
- local Ollama models and generated outputs

These files should remain local to the machine unless a project-specific distribution mechanism is explicitly approved.

## Security notes

- Never commit real `.env` files or credentials.
- Keep secrets, API keys, login tokens, and local environment data out of Git history.
- Use `.env.example` only for safe placeholders and non-secret sample values.

## Limitations and verification status

The repository contains the complete research and application codebase, but some setup steps require local resources that are not always pre-populated in a fresh environment:

- Ollama and the `qwen2.5:7b` model must be installed locally.
- The embedding cache for `all-MiniLM-L6-v2` must be available.
- Local ChromaDB and BM25 indexes are generated at runtime.
- Adaptive predictor model files are local artifacts that may need to be reproduced from training data.

The instructions in this README are based on the actual code in the repository and are intended to be followed as a reproducible setup guide. Where a step depends on a local artifact or a model download, the README labels that requirement clearly rather than implying it is already bundled in the repo.

## License

The repository currently serves as an academic and research project. Add a formal license if you plan to release the project outside the project or final-year academic context.

nDCG:
  Improved : 209
  Equal    : 675
  Decreased: 116

Precision:
  Improved : 67
  Equal    : 904
  Decreased: 29

Recall:
  Improved : 67
  Equal    : 904
  Decreased: 29

These query-level results demonstrate that the adaptive predictor is actively changing the retrieval balance rather than behaving as a fixed-weight hybrid.

Domain Evaluation

The 1,000-query benchmark contains four evaluated domains.

Domain

Precision@10

Recall@10

MRR

nDCG@10

Argumentation

0.0792

0.7920

0.2499

0.3800

Finance

0.0984

0.4468

0.4391

0.3697

Medical

0.1960

0.1345

0.4303

0.2611

Scientific

0.0856

0.7792

0.6170

0.6527

The domain results illustrate that retrieval behavior varies substantially by domain, which is one of the motivations for query-dependent hybrid weighting.

System Architecture

Backend

The backend is implemented in Python and provides:

FastAPI
├── Authentication
├── User routes
├── Conversation management
├── User chat
├── Document ingestion
├── User-specific indexing
├── BM25 retrieval
├── Dense retrieval
├── Adaptive hybrid fusion
├── Context selection
└── LLM generation

Frontend

The frontend is implemented using:

React
Vite
CSS
Lucide React

The interface provides:

Authentication

Conversation sidebar

Document upload

Chat interface

Retrieval pipeline visualization

Retrieval metrics

Adaptive weights

Research/evaluation dashboard

User Document RAG

The application supports user-provided documents.

The workflow is:

Upload Document
      ↓
Document Extraction
      ↓
Text Cleaning
      ↓
Chunking
      ↓
Embedding Generation
      ↓
User-specific Index
      ↓
BM25 + Dense Retrieval
      ↓
Adaptive Fusion
      ↓
Relevant Context
      ↓
LLM Generation

Uploaded document metadata is preserved so that retrieved chunks can be associated with their original document.

Authentication

The application includes a persistent SQLite authentication layer.

Supported functionality includes:

User registration

Login using username/email

Password hashing

Sessions

Email verification

Password reset

Last-login tracking

Passwords are stored as password hashes with individual salts rather than plaintext passwords.

Local runtime databases are intentionally excluded from Git.

Conversation Persistence

User conversations and messages are stored persistently.

The application tracks:

User
 └── Conversations
      └── Messages
           ├── User queries
           └── Assistant responses

This allows users to return to previous conversations instead of losing chat history between sessions.

Latency Metrics

The RAG pipeline reports timing information for individual stages, including:

BM25 retrieval
Dense retrieval
Fusion
Reranking
Context selection
Prompt preparation
LLM generation
Total latency

Generation latency is measured from the actual LLM generation stage rather than being treated as a fixed or placeholder value.

Project Structure

adaptive-hybrid-rag/
│
├── backend/
│   ├── api/
│   │   ├── auth.py
│   │   ├── conversations.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   ├── user_chat.py
│   │   └── user_routes.py
│   │
│   ├── evaluation/
│   │   ├── dataset_v2/
│   │   ├── benchmark_v4_rich_actual.py
│   │   └── ...
│   │
│   ├── services/
│   │   ├── adaptive_predictor_v4.py
│   │   ├── auth_service.py
│   │   ├── conversation_service.py
│   │   ├── hybrid_service.py
│   │   ├── llm_service.py
│   │   ├── model_loader.py
│   │   ├── retrieval_service.py
│   │   ├── ingestion/
│   │   └── user_rag/
│   │
│   ├── app.py
│   └── ...
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── bm25_cache/          # local/generated
├── chroma_db/           # local/generated
├── .gitignore
└── README.md

Installation

1. Clone the repository

git clone https://github.com/Parasfunde21/adaptive-hybrid-rag.git
cd adaptive-hybrid-rag

2. Create and activate a Python environment

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install backend dependencies

If a backend requirements file is present:

pip install -r backend/requirements.txt

4. Install frontend dependencies

cd frontend
npm install

Running the Application

Start the backend

From the project root:

.\venv\Scripts\Activate.ps1
cd backend
python app.py

The backend is expected to run on:

http://localhost:8000

Start the frontend

Open another terminal:

cd frontend
npm run dev

The Vite development server will provide the frontend URL shown in the terminal.

Database Initialization

The project includes a database migration utility.

From the backend directory:

python services/migrate_database.py

The local SQLite databases are runtime data and are intentionally excluded from version control.

Evaluation

The actual V4 benchmark can be run from the backend directory using:

python -m evaluation.benchmark_v4_rich_actual

The benchmark compares:

BM25

Dense

Fixed Hybrid

Adaptive Hybrid

and produces summary, domain, query-comparison, adaptive-weight, and improvement-analysis outputs under the evaluation results directory.

Evaluation Metrics

Precision@10

Measures the proportion of the top 10 retrieved documents that are relevant.

Recall@10

Measures how much of the relevant material is retrieved within the top 10 results.

MRR

Mean Reciprocal Rank measures how highly the first relevant result is ranked.

nDCG@10

Normalized Discounted Cumulative Gain evaluates ranking quality while giving higher importance to relevant documents appearing near the top.

Research Interpretation

The benchmark should be interpreted as evidence that adaptive weighting can improve several retrieval metrics compared with a fixed hybrid baseline on the evaluated dataset.

The results should not be interpreted as proof that adaptive retrieval is better for every possible dataset or query.

In particular:

MRR is slightly lower for Adaptive Hybrid than Fixed Hybrid in the 1,000-query benchmark.

Query-level improvements are not universal.

Benchmark performance depends on the evaluated datasets and retrieval configuration.

Model artifacts and large local indexes are intentionally excluded from Git.

The evaluation is intended to demonstrate the behavior and effectiveness of the adaptive retrieval approach.

This conservative interpretation is important for reproducible research and for avoiding overstating the contribution.

Technology Stack

Frontend

React

Vite

JavaScript

CSS

Lucide React

Backend

Python

FastAPI

SQLite

ChromaDB

Sentence Transformers

BM25 / rank-bm25

scikit-learn

Retrieval

BM25

Dense vector retrieval

Adaptive weighted fusion

Retrieval feature extraction

Generation

Qwen-based local LLM pipeline

Current Project Status

[x] BM25 retrieval
[x] Dense retrieval
[x] Hybrid retrieval
[x] Adaptive retrieval
[x] V4 Rich predictor
[x] User authentication
[x] Email verification backend
[x] Password reset backend
[x] User document upload
[x] User-specific RAG
[x] Conversation persistence
[x] Retrieval metrics
[x] Generation latency
[x] Research dashboard
[x] 1,000-query benchmark
[x] Domain evaluation
[x] GitHub repository

Research Pipeline Summary

                    ┌──────────────────┐
                    │    User Query    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Query Processing  │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
       ┌──────▼──────┐               ┌──────▼──────┐
       │     BM25    │               │    Dense    │
       │   Top-50    │               │   Top-50    │
       └──────┬──────┘               └──────┬──────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ Retrieval Features  │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ Adaptive Predictor  │
                  └──────────┬──────────┘
                             │
                       BM25 / Dense
                          weights
                             │
                  ┌──────────▼──────────┐
                  │  Adaptive Fusion    │
                  └──────────┬──────────┘
                             │
                          Top-10
                             │
                  ┌──────────▼──────────┐
                  │ Context Selection   │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Qwen Generation   │
                  └──────────┬──────────┘
                             │
                     ┌───────▼───────┐
                     │ Final Answer  │
                     └───────────────┘

License

This repository is currently intended as an academic/research project.

Add an explicit license here if the project is later released for external reuse.

Acknowledgement

This project was developed as a final-year research project focused on adaptive hybrid retrieval and Retrieval-Augmented Generation.