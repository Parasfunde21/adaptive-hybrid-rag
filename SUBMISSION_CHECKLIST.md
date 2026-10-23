# Final-year submission checklist

## Source code
- [ ] Confirm the final backend and frontend code are committed in the repo.
- [ ] Review the main application entry points: `backend/app.py`, `backend/api/routes.py`, `frontend/src/App.jsx`.
- [ ] Confirm the research pipeline and user-document pipeline remain intact.

## Supporting files and model artifacts
- [ ] Confirm `.env.example` is in place and contains only safe placeholders.
- [ ] Confirm `.gitignore` excludes local secrets, databases, caches, and generated indexes.
- [ ] Verify required local artifacts are present or reproducible:
  - `backend/models/adaptive_weight_model_v4_rich.pkl`
  - `chroma_db/` collection data if available locally
  - `bm25_cache/` if generated locally
  - local Ollama model `qwen2.5:7b`
  - local embedding cache for `all-MiniLM-L6-v2`

## Installation and launch
- [ ] Python environment created and activated.
- [ ] Backend dependencies installed from `backend/requirements.txt`.
- [ ] Frontend dependencies installed from `frontend/package-lock.json` using `npm ci`.
- [ ] Ollama is installed and the model is pulled.
- [ ] Local app launches without blocking errors.

## End-to-end test
- [ ] Backend health endpoint responds successfully.
- [ ] Frontend loads at the local dev URL.
- [ ] User upload flow works.
- [ ] Question submission returns a response and retrieval metadata.
- [ ] Generated answer is inspectable in the UI or API response logs.

## Evaluation and benchmarks
- [ ] Evaluation scripts under `backend/evaluation/` are available.
- [ ] Datasets required for reproduction are present.
- [ ] Benchmark or training script can be rerun if a grader needs to verify results.
- [ ] Performance claims are aligned with the actual benchmark outputs and not overstated.

## Academic documents
- [ ] Final report, thesis text, or academic documentation is prepared separately as required by the course.
- [ ] Any figures or benchmark tables are consistent with repository output.

## Git and submission
- [ ] Working tree is clean after commit or intentionally contains only approved changes.
- [ ] Final commit message is clear and references the submission work.
- [ ] Push is attempted only if permissions and credentials are available.
- [ ] Final repository state matches the accepted submission package.
