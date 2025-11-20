# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.chunk_router import chunk_router
from backend.api.llm_router import llm_router
from backend.api.search_router import search_router

# Import app_state so it loads config, LLM, embeddings once
from backend.core import app_state

app = FastAPI(title="PDF_ChatBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chunk_router, prefix="/api")
app.include_router(search_router, prefix="/api/search")
app.include_router(llm_router, prefix="/api/llm")

@app.get("/")
def root():
    return {"message": "API is up!"}

@app.on_event("startup")
def startup_event():
    # trigger initialization
    _ = app_state.config
    _ = app_state.llm
    _ = app_state.embedding_model
