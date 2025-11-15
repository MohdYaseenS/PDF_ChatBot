import os 
import shutil
import tempfile
import atexit
from pypdf import PdfReader

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

