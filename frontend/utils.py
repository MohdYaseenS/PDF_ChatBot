import os 
import shutil
import tempfile
import atexit

session_dir = tempfile.mkdtemp(prefix="gradio_pdf_session_")
pdf_path = None

# Cleanup function — removes temp folder when app exits
def cleanup():
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        print(f"Cleaned up session directory: {session_dir}")

atexit.register(cleanup)

# Handle PDF upload
def upload_pdf(file):
    global pdf_path
    if file is None:
        return "Please upload a PDF first."
    pdf_path = os.path.join(session_dir, os.path.basename(file.name))
    shutil.copy(file.name, pdf_path)
    return "PDF uploaded successfully"


# Handle question submission
def handle_question(question):
    global pdf_path 
    if pdf_path is None:
        return "Please upload a PDF before asking a question."
    # Here you can process the question or send it to an API
    # For demonstration, we just echo the question
    return f"Received your question: '{question}'\nPDF in use: {os.path.basename(pdf_path)}"

