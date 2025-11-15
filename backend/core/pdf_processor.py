import os
import shutil
import tempfile
import atexit
import requests
from pypdf import PdfReader
import numpy as np
from typing import Optional

import os
import requests

class PDFProcessor:
    def __init__(self, api_url: str = None):
        if api_url is None:
            port = os.environ.get("CHUNK_API_PORT", "8000")
            api_url = f"http://localhost:{port}/chunk_and_vectorize"
        
        self.api_url = api_url

        # Check if API is reachable
        try:
            response = requests.get(self.api_url.replace("/chunk_and_vectorize", "/health"), timeout=3)
            if response.status_code != 200:
                raise APIConnectionError(f"Chunk API returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Cannot connect to Chunk API at {self.api_url}: {e}")

        # Initialize other attributes
        self.session_dir = tempfile.mkdtemp(prefix="gradio_pdf_session_")
        self.pdf_path: Optional[str] = None
        self.pdf_text: str = ""
        self.chunks: list = []
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index: Optional[object] = None
        atexit.register(self.cleanup)

    def cleanup(self):
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir)
            print(f"Cleaned up session directory: {self.session_dir}")

    def upload_pdf(self, file):
        if file is None:
            return "Please upload a PDF first."

        # Save PDF
        self.pdf_path = os.path.join(self.session_dir, os.path.basename(file.name))
        shutil.copy(file.name, self.pdf_path)

        # Extract text
        self.pdf_text = ""
        try:
            reader = PdfReader(self.pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    self.pdf_text += page_text + "\n"
        except Exception as e:
            return f"Error reading PDF: {e}"

        if not self.pdf_text.strip():
            return "The PDF contains no extractable text."

        # Call the API endpoint to chunk + vectorize
        try:
            self._call_chunk_api()
        except Exception as e:
            return f"Error processing PDF via API: {e}"

        return f"PDF uploaded and indexed successfully! {len(self.chunks)} chunks processed."

    def _call_chunk_api(self):
        """
        Sends the PDF text to the FastAPI endpoint to chunk and vectorize,
        then stores the results in class attributes.
        """
        if not self.pdf_text.strip():
            raise ValueError("No text available to send to chunk API.")

        payload = {"text": self.pdf_text}

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Expected: the endpoint returns chunks and vectors
            self.chunks = data.get("chunks", [])
            self.vectors = np.array(data.get("vectors", []))
            self.faiss_index = data.get("faiss_index",[])

            if not self.chunks or self.vectors.size == 0:
                raise ValueError("Empty chunks or vectors returned from API.")

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to chunk API: {e}")

        except Exception as e:
            raise RuntimeError(f"Error calling chunk API: {e}")

    
    def handle_question(self, question, top_k: int = 3):
        if self.vectors is None or not self.chunks:
            return "Please upload and process a PDF before asking a question."

        try:
            # Embed question using the same API or locally
            question_vector = np.array([self.vectors[0]])  # placeholder
            # Ideally, call your model embedding function here
            # Then search FAISS index if available
            # matched_chunks = ...

            return f"Top matches for your question:\n\n" + "\n---\n".join(self.chunks[:top_k])

        except Exception as e:
            return f"Error handling question: {e}"
