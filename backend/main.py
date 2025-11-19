from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.chunk_router import chunk_router
from backend.api.llm_router import llm_router
from backend.api.search_router import search_router

app = FastAPI(title="PDF_ChatBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chunk_router, prefix="/api")
app.include_router(llm_router, prefix="/llm")
app.include_router(chunk_router, prefix="/search")

@app.get("/")
def root():
    return {"message": "API is up!"}