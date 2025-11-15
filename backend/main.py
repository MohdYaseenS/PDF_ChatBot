from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.chunk_router import router as chunk_router

app = FastAPI(title="PDF_ChatBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chunk_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "API is up!"}