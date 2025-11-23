import os
import shutil
import tempfile
import atexit
import httpx
import asyncio
import logging
import numpy as np
from typing import Optional, Generator
from pypdf import PdfReader

logger = logging.getLogger("core.pdf_processor")


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop that is safe from nested-loop errors."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class PDFProcessor:
    def __init__(self, api_url: Optional[str] = None, index_key: str = "default"):
        port = os.environ.get("CHUNK_API_PORT", "8081")
        self.api_url = api_url or f"http://localhost:{port}"

        self.session_dir = tempfile.mkdtemp(prefix="gradio_pdf_session_")
        self.pdf_path: Optional[str] = None
        self.pdf_text: str = ""
        self.chunks: list[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.index_key = index_key

        self.client = httpx.AsyncClient(timeout=120.0)
        atexit.register(self.cleanup)

        logger.info(f"PDFProcessor initialized with API URL: {self.api_url}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def cleanup(self):
        """Cleanup session directories and close async HTTP client safely."""
        try:
            if os.path.isdir(self.session_dir):
                shutil.rmtree(self.session_dir)
                logger.info(f"Cleaned up session dir: {self.session_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning session directory: {e}")

        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.client.aclose())
            loop.close()
            logger.debug("HTTP client closed successfully.")
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")

    # -------------------------------------------------------------------------
    # Upload + Process PDF
    # -------------------------------------------------------------------------
    def upload_pdf(self, file) -> str:
        if file is None:
            return "Please upload a PDF first."

        logger.info(f"Uploading PDF: {file.name}")

        # Copy PDF
        self.pdf_path = os.path.join(self.session_dir, os.path.basename(file.name))
        shutil.copy(file.name, self.pdf_path)

        # Extract PDF text
        try:
            reader = PdfReader(self.pdf_path)
            self.pdf_text = ""
            for page in reader.pages:
                text = page.extract_text() or ""
                self.pdf_text += text + "\n"

            logger.info(
                f"Extracted {len(self.pdf_text)} characters "
                f"from {len(reader.pages)} PDF pages."
            )
        except Exception as e:
            logger.exception(f"Error reading PDF: {e}")
            return f"Error reading PDF: {e}"

        if not self.pdf_text.strip():
            return "The PDF contains no extractable text."

        # Send to chunker + build index
        loop = get_event_loop()
        try:
            logger.info("Sending text to chunking API...")

            # Chunk request
            r = loop.run_until_complete(
                self.client.post(
                    f"{self.api_url}/api/chunk",
                    json={"text": self.pdf_text},
                )
            )
            r.raise_for_status()

            data = r.json()
            self.chunks = data.get("chunks", [])
            self.vectors = np.array(data.get("vectors", []), dtype=np.float32)

            logger.info(f"Received {len(self.chunks)} chunks. Building index...")

            # Build index
            build_resp = loop.run_until_complete(
                self.client.post(
                    f"{self.api_url}/api/search/build_index",
                    json={
                        "key": self.index_key,
                        "chunks": self.chunks,
                        "vectors": self.vectors.tolist(),
                    },
                )
            )
            build_resp.raise_for_status()

            logger.info(f"Index built successfully for key: {self.index_key}")
            return f"PDF uploaded and indexed successfully! {len(self.chunks)} chunks processed."

        except Exception as e:
            logger.exception(f"Error processing PDF: {e}")
            return f"Error processing PDF via API: {e}"

    # -------------------------------------------------------------------------
    # Ask (non-streaming)
    # -------------------------------------------------------------------------
    def ask(self, question: str, top_k: int = 3) -> str:
        if not self.chunks or self.vectors is None:
            return "Please upload and process a PDF before asking a question."

        loop = get_event_loop()

        try:
            logger.info(f"Getting context for question: {question[:50]}...")

            # Search matches
            resp = loop.run_until_complete(
                self.client.post(
                    f"{self.api_url}/api/search/query",
                    json={"key": self.index_key, "query": question, "top_k": top_k},
                )
            )
            resp.raise_for_status()

            matches = resp.json().get("matches", [])
            context = "\n\n---\n\n".join(matches)

            # Get answer from LLM
            ans_resp = loop.run_until_complete(
                self.client.post(
                    f"{self.api_url}/api/llm/answer",
                    json={"context": context, "question": question},
                )
            )
            ans_resp.raise_for_status()

            answer = ans_resp.json().get("answer", "")
            return answer

        except Exception as e:
            logger.exception(f"Error handling question: {e}")
            return f"Error handling question: {e}"

    # -------------------------------------------------------------------------
    # Ask (streaming)
    # -------------------------------------------------------------------------
    def ask_stream(self, question: str, top_k: int = 3) -> Generator[str, None, None]:
        if not self.chunks or self.vectors is None:
            yield "Please upload and process a PDF before asking a question."
            return

        loop = get_event_loop()

        # Step 1: Context Retrieval
        try:
            resp = loop.run_until_complete(
                self.client.post(
                    f"{self.api_url}/api/search/query",
                    json={"key": self.index_key, "query": question, "top_k": top_k},
                )
            )
            resp.raise_for_status()

            matches = resp.json().get("matches", [])
            context = "\n\n---\n\n".join(matches)

        except Exception as e:
            logger.exception(f"Error fetching context: {e}")
            yield f"Error fetching context: {e}"
            return

        # Step 2: Stream LLM Response
        try:
            async def stream_answer():
                async with self.client.stream(
                    "POST",
                    f"{self.api_url}/api/llm/answer",
                    json={"context": context, "question": question},
                ) as ans_resp:
                    ans_resp.raise_for_status()
                    async for chunk in ans_resp.aiter_bytes():
                        if chunk:
                            yield chunk.decode("utf-8", errors="ignore")

            # Run async generator synchronously
            agen = stream_answer()

            while True:
                try:
                    piece = loop.run_until_complete(agen.__anext__())
                    yield piece
                except StopAsyncIteration:
                    break

        except Exception as e:
            logger.exception("Streaming error")
            yield f"Streaming error: {e}"
