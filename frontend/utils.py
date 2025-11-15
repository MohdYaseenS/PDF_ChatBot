import os 
import shutil
import tempfile
import atexit
from pypdf import PdfReader
import requests

session_dir = tempfile.mkdtemp(prefix="gradio_pdf_session_")
pdf_path = None
pdf_text = ""  # <-- store extracted text here

# Cleanup function — removes temp folder when app exits
def cleanup():
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        print(f"Cleaned up session directory: {session_dir}")

atexit.register(cleanup)

# Handle PDF upload
def upload_pdf(file):
    global pdf_path, pdf_text
    if file is None:
        return "Please upload a PDF first."
    pdf_path = os.path.join(session_dir, os.path.basename(file.name))
    shutil.copy(file.name, pdf_path)
    try:
        reader = PdfReader(pdf_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
    except Exception as e:
        return f"Error reading PDF: {e}" 
        
    return f"PDF uploaded successfully, PDF processed successfully.Stored at: {pdf_path} with content: {pdf_text}"


# Handle question submission
def handle_question(question):
    global pdf_path 
    if pdf_path is None:
        return "Please upload a PDF before asking a question."
    # Here you can process the question or send it to an API
    # For demonstration, we just echo the question
    return f"Received your question: '{question}'\nPDF in use: {os.path.basename(pdf_path)}"


def text_cunk_request(pdf_text):
    url = "http://localhost:8081/api/chunk_and_vectorize"  # URL of the backend API endpoint
    payload = {"text": pdf_text}
    try:
        response = requests.post(url, json=payload) # Send POST request to the backend API
        return response  
    except requests.exceptions.RequestException as e:
        yield f"Connection error: {str(e)}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"

