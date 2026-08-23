Adaptive Hybrid RAG

A research-oriented Retrieval-Augmented Generation (RAG) system that combines BM25 lexical retrieval, dense semantic retrieval, and query-dependent adaptive fusion to improve retrieval quality across different query types.

The project also provides a complete user-facing workflow for uploading documents, asking questions, inspecting retrieval behavior, and viewing retrieval/generation metrics.

Overview

Traditional hybrid retrieval commonly uses a fixed BM25/Dense weighting such as:

BM25 = 0.50
Dense = 0.50

The problem is that different queries can favor different retrieval signals. Keyword-heavy queries may benefit more from lexical retrieval, while semantic or paraphrased queries may benefit more from dense retrieval.

This project therefore uses a learned retrieval-weight predictor to estimate the appropriate BM25/Dense balance for each query.

                         User Query
                              |
                              v
                    +-------------------+
                    | Query Processing  |
                    +-------------------+
                              |
                 +------------+------------+
                 |                         |
                 v                         v
        +----------------+        +----------------+
        | BM25 Retrieval |        | Dense Retrieval|
        |  Top-50        |        |  Top-50        |
        +----------------+        +----------------+
                 |                         |
                 +------------+------------+
                              |
                              v
                  Retrieval Feature Extraction
                              |
                              v
                 +---------------------------+
                 | Adaptive Weight Predictor |
                 |     Extra Trees / V4      |
                 +---------------------------+
                              |
                    BM25 / Dense weights
                              |
                              v
                     Adaptive Fusion
                              |
                              v
                       Top-K Results
                              |
                              v
                     Context Selection
                              |
                              v
                         LLM / Qwen
                              |
                              v
                         Final Answer

Key Features

BM25 lexical retrieval

Dense semantic retrieval using Sentence Transformers

Adaptive BM25/Dense fusion

Retrieval-diagnostic feature extraction

Learned adaptive weight prediction

Candidate retrieval before final Top-K fusion

User PDF/document upload and indexing

User-specific document retrieval

Conversational chat history

Persistent users and conversations

Email verification backend

Password reset backend

Retrieval and generation latency measurement

Research dashboard with retrieval metrics

Experimental benchmark visualization

Domain-wise evaluation

1,000-query multi-domain evaluation

Research Contribution

The main research contribution is query-dependent retrieval weighting.

Instead of assuming that BM25 and dense retrieval should always receive the same weight, the system extracts retrieval and query characteristics and predicts a BM25 fusion weight:

predicted_bm25_weight = f(query_features, retrieval_features)

predicted_dense_weight = 1 - predicted_bm25_weight

The final fused score is based on the independently normalized retrieval scores:

Fused Score =
    BM25 Weight × Normalized BM25 Score
    +
    Dense Weight × Normalized Dense Score

The model does not directly predict document relevance. It predicts the retrieval-fusion weight used to combine the two retrieval signals.

Adaptive Retrieval Features

The V4 Rich feature set contains 17 features derived from query and retrieval behavior.

BM25 features

bm25_top_score

bm25_mean_score

bm25_score_std

bm25_score_gap

bm25_top_mean_ratio

Dense features

dense_top_similarity

dense_mean_similarity

dense_similarity_std

dense_similarity_gap

dense_top_mean_ratio

Cross-retriever features

overlap_ratio

rank_agreement

Query features

query_token_count

query_unique_token_count

query_avg_token_length

Lexical coverage

bm25_top_lexical_coverage

dense_top_lexical_coverage

These features describe how confident and consistent the retrieval systems appear for a particular query. The technical analysis found dense_top_lexical_coverage to be the strongest individual feature in the V4 Rich model.

V4 Rich Model

The V4 Rich adaptive predictor uses an Extra Trees regression model.

Configuration

Model: ExtraTreesRegressor
Estimators: 500
Max features: 0.8
Min samples leaf: 2
Random state: 42
Features: 17

The model predicts a BM25 weight, which is clipped to the valid range:

0 <= BM25 weight <= 1
Dense weight = 1 - BM25 weight

The project also retains earlier V3 adaptive components as part of the research lineage and comparison.

Retrieval Configuration

The current retrieval configuration uses:

Candidate K : 50
BM25 candidates : 50
Dense candidates : 50
Final K : 10

The larger candidate pool allows the adaptive fusion stage to select from a wider set of lexical and semantic candidates before producing the final Top-10 results.

Experimental Evaluation

The current actual V4 benchmark contains:

Queries: 1,000
Candidate K: 50
Final K: 10
Adaptive model: Extra Trees V4 Rich

Fixed Hybrid vs Adaptive Hybrid

Metric

Fixed Hybrid

Adaptive Hybrid

Relative Change

Precision@10

0.1111

0.1148

+3.33%

Recall@10

0.5189

0.5381

+3.70%

MRR

0.4358

0.4340

-0.40%

nDCG@10

0.4073

0.4159

+2.09%

The results show improvements in Precision@10, Recall@10, and nDCG@10 over the fixed hybrid configuration. MRR is slightly lower, so the result should be reported honestly rather than presenting the adaptive method as universally better.

Adaptive Effect Across Queries

Out of 1,000 benchmark queries:

Adaptive weights changed: 1000 / 1000

MRR:
  Improved : 138
  Equal    : 768
  Decreased: 94

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