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
        """
        Stream PDF Q&A response from the backend LLM API.

        Yields partial answer as it arrives.
        """
        logger.info(f"ask_stream called with question: {question[:50]}...")
        
        if not self.chunks or self.vectors is None:
            error_msg = "Please upload and process a PDF before asking a question."
            logger.warning(error_msg)
            yield error_msg
            return

        try:
            # 1️⃣ Search for relevant chunks
            logger.info("Searching for relevant chunks...")
            qresp = requests.post(
                f"{self.api_url}/api/search/query",
                json={
                    "key": self.index_key,
                    "query": question,
                    "top_k": top_k,
                },
                timeout=30,
            )
            qresp.raise_for_status()
            try:
                data = qresp.json()
            except ValueError:
                error_msg = f"Invalid JSON from search API: {qresp.text}"
                logger.error(error_msg)
                yield error_msg
                return

            matches = data.get("matches", [])
            context = "\n\n---\n\n".join(matches)
            logger.info(f"Found {len(matches)} matching chunks, context length: {len(context)}")

            # 2️⃣ Streaming LLM call
            # Use stream=True and ensure no buffering
            logger.info("Calling LLM API for streaming response...")
            ans_resp = requests.post(
                f"{self.api_url}/api/llm/answer",
                json={"context": context, "question": question},
                stream=True,
                timeout=60,
            )
            ans_resp.raise_for_status()
            
            # Log for debugging
            logger.info(f"Streaming response received. Status: {ans_resp.status_code}")
            logger.info(f"Content-Type: {ans_resp.headers.get('Content-Type')}")
            logger.info(f"Transfer-Encoding: {ans_resp.headers.get('Transfer-Encoding')}")

            # 3️⃣ Stream content incrementally
            # For plain text streaming, read and accumulate chunks
            answer = ""
            buffer = b""  # Use bytes buffer for UTF-8 character handling
            chunk_count = 0
            
            try:
                # Read streaming response - FastAPI StreamingResponse sends bytes
                # Use very small chunk size (1 byte) to get updates as fast as possible
                # This ensures we get chunks immediately as they arrive
                for chunk_bytes in ans_resp.iter_content(chunk_size=1, decode_unicode=False):
                    if chunk_bytes:
                        chunk_count += 1
                        buffer += chunk_bytes
                        
                        # Try to decode complete UTF-8 sequences
                        try:
                            # Attempt to decode the buffer
                            decoded = buffer.decode('utf-8')
                            # Successfully decoded - add to answer and clear buffer
                            answer += decoded
                            buffer = b""
                            # Yield accumulated answer for Gradio (Gradio updates UI with each yield)
                            logger.debug(f"Yielding answer chunk #{chunk_count}, length: {len(answer)}")
                            yield answer
                        except UnicodeDecodeError:
                            # Incomplete UTF-8 sequence - keep in buffer
                            # Continue accumulating bytes until we have a complete character
                            pass
                    
                # Handle any remaining buffer at the end
                if buffer:
                    try:
                        decoded = buffer.decode('utf-8')
                        answer += decoded
                    except UnicodeDecodeError:
                        # Force decode with error replacement for any remaining bytes
                        decoded = buffer.decode('utf-8', errors='replace')
                        answer += decoded
                    logger.info(f"Final yield, total answer length: {len(answer)}")
                    yield answer
                else:
                    logger.info(f"Streaming complete, total answer length: {len(answer)}, chunks: {chunk_count}")
                    if answer:
                        yield answer  # Final yield even if no buffer
                    
            except Exception as stream_error:
                logger.exception(f"Error during streaming: {stream_error}")
                yield f"Error during streaming: {str(stream_error)}"

        except requests.exceptions.RequestException as e:
            logger.exception("Request error in ask_stream()")
            yield f"Request error: {e}"
        except Exception as e:
            logger.exception("Unexpected error in ask_stream()")
            yield f"Unexpected error: {e}"