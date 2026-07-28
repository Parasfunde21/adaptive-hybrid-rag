from fastapi import FastAPI

app = FastAPI(title="Adaptive Hybrid RAG")

@app.get("/")
def root():
    return {"message": "Adaptive Hybrid RAG API is running"}