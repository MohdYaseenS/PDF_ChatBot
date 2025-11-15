import os
import shutil
import tempfile
import atexit
from pypdf import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class PDFProcessor:
    def __init__(self):
        # Create temporary session folder
        self.session_dir = tempfile.mkdtemp(prefix="gradio_pdf_session_")
        self.pdf_path = None
        self.pdf_text = ""
        self.chunks = []
        self.vectors = None
        self.faiss_index = None

        # Cleanup temp folder on exit
        atexit.register(self.cleanup)

        # Load embedding model once
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def cleanup(self):
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir)
            print(f"Cleaned up session directory: {self.session_dir}")

    # Recursive text splitter
    def recursive_text_splitter(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start_idx = 0
        while start_idx < len(text):
            end_idx = start_idx + chunk_size
            chunks.append(text[start_idx:end_idx])
            start_idx = end_idx - overlap
        return chunks

    # Upload and process PDF
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

        # Chunk text
        self.chunks = self.recursive_text_splitter(self.pdf_text, chunk_size=1000, overlap=200)

        # Vectorize chunks
        self.vectors = self.model.encode(self.chunks, show_progress_bar=False)
        self.vectors = np.array(self.vectors)

        # Create FAISS index
        dimension = self.vectors.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.vectors)

        return f"PDF uploaded and indexed successfully! {len(self.chunks)} chunks processed."

    # Handle question using FAISS
    def handle_question(self, question):
        if self.faiss_index is None:
            return "Please upload and process a PDF before asking a question."

        # Embed question
        question_vector = self.model.encode([question])
        
        # Search FAISS index
        D, I = self.faiss_index.search(np.array(question_vector), k=3)
        matched_chunks = [self.chunks[i] for i in I[0] if i < len(self.chunks)]

        return f"Top matches for your question:\n\n" + "\n---\n".join(matched_chunks)
