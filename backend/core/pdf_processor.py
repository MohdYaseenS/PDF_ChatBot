# backend/core/pdf_processor.py
import os
import shutil
import tempfile
import atexit
import requests
from pypdf import PdfReader
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger("core.pdf_processor")

class PDFProcessor:
    def __init__(self, api_url: Optional[str] = None, index_key: str = "default"):
        port = os.environ.get("CHUNK_API_PORT", "8081")
        self.api_url = api_url or f"http://localhost:{port}"
        self.session_dir = tempfile.mkdtemp(prefix="gradio_pdf_session_")
        self.pdf_path: Optional[str] = None
        self.pdf_text: str = ""
        self.chunks = []
        self.vectors = None
        self.index_key = index_key
        atexit.register(self.cleanup)

    def cleanup(self):
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir)
            logger.info(f"Cleaned up {self.session_dir}")

    def upload_pdf(self, file) -> str:
        if file is None:
            return "Please upload a PDF first."
        self.pdf_path = os.path.join(self.session_dir, os.path.basename(file.name))
        shutil.copy(file.name, self.pdf_path)
        self.pdf_text = ""
        try:
            reader = PdfReader(self.pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.pdf_text += text + "\n"
        except Exception as e:
            return f"Error reading PDF: {e}"
        if not self.pdf_text.strip():
            return "The PDF contains no extractable text."

        try:
            r = requests.post(f"{self.api_url}/api/chunk", json={"text": self.pdf_text}, timeout=120)
            r.raise_for_status()
            data = r.json()
            self.chunks = data.get("chunks", [])
            self.vectors = np.array(data.get("vectors", []), dtype=np.float32)
            build_resp = requests.post(f"{self.api_url}/api/search/build_index", json={
                "key": self.index_key,
                "chunks": self.chunks,
                "vectors": self.vectors.tolist(),
            }, timeout=60)
            build_resp.raise_for_status()
        except Exception as e:
            return f"Error processing PDF via API: {e}"
        return f"PDF uploaded and indexed successfully! {len(self.chunks)} chunks processed."

    def ask(self, question: str, top_k: int = 3) -> str:
        if not self.chunks or self.vectors is None:
            return "Please upload and process a PDF before asking a question."
        try:
            qresp = requests.post(f"{self.api_url}/api/search/query", json={
                "key": self.index_key,
                "query": question,
                "top_k": top_k,
            }, timeout=30)
            qresp.raise_for_status()
            data = qresp.json()
            matches = data.get("matches", [])
            context = "\n\n---\n\n".join(matches)
            ans_resp = requests.post(f"{self.api_url}/api/llm/answer", json={
                "context": context,
                "question": question,
            }, timeout=60)
            ans_resp.raise_for_status()
            answer = ans_resp.json().get("answer")
            return answer
        except Exception as e:
            return f"Error handling question: {e}"
    
    
    def ask_stream(self, question: str, top_k: int = 3):

        logger = logging.getLogger(__name__)

        if not self.chunks or self.vectors is None:
            yield "Please upload and process a PDF before asking a question."
            return

        # Step 1: Search for relevant chunks
        try:
            resp = requests.post(
                f"{self.api_url}/api/search/query",
                json={"key": self.index_key, "query": question, "top_k": top_k},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            matches = data.get("matches", [])
            context = "\n\n---\n\n".join(matches)
        except Exception as e:
            yield f"Error fetching context: {e}"
            return

        # Step 2: Stream from LLM backend
        try:
            with requests.post(
                f"{self.api_url}/api/llm/answer",
                json={"context": context, "question": question},
                stream=True,
                timeout=60
            ) as ans_resp:
                ans_resp.raise_for_status()
                answer = ""

                # Iterate over lines (FastAPI streaming usually sends bytes ending with b'\n')
                for chunk in ans_resp.iter_lines(decode_unicode=True):
                    if chunk:  # ignore keep-alive or empty lines
                        answer += chunk + "\n"
                        yield answer  # yield cumulative text for Gradio

                # Ensure final yield in case last chunk didn't end with newline
                if answer:
                    yield answer

        except Exception as e:
            logger.exception("Error during LLM streaming")
            yield f"Streaming error: {e}"