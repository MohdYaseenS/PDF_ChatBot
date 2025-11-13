import gradio as gr
from utils import upload_pdf, handle_question

# Only launch app if script is run directly
if __name__ == "__main__":
    with gr.Blocks() as app:
        gr.Markdown("## 📄 PDF Question Interface")
        
        with gr.Tab("Upload PDF"):
            pdf_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
            upload_output = gr.Textbox(label="Upload status", interactive=False, placeholder="Load your PDF")
            upload_btn = gr.Button("Upload")
            upload_btn.click(upload_pdf, inputs=pdf_input, outputs=upload_output)
        
        with gr.Tab("Ask Question"):
            question_input = gr.Textbox(label="Type your question", placeholder="Enter your question here...")
            submit_btn = gr.Button("Submit Question")
            response_output = gr.Textbox(label="Response")
            submit_btn.click(handle_question, inputs=question_input, outputs=response_output)
    
    app.launch()