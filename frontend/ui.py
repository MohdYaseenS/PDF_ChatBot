import gradio as gr
from backend.core.pdf_processor import PDFProcessor

# Create an instance of the processor
pdf_processor = PDFProcessor()

if __name__ == "__main__":
    with gr.Blocks() as app:
        gr.Markdown("## 📄 PDF Question Interface")
        
        # Upload PDF tab
        with gr.Tab("Upload PDF"):
            pdf_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
            upload_output = gr.Textbox(label="Upload status", interactive=False)
            upload_btn = gr.Button("Upload")
            upload_btn.click(pdf_processor.upload_pdf, inputs=pdf_input, outputs=upload_output)
        
        # Ask question tab
        with gr.Tab("Ask Question"):
            question_input = gr.Textbox(label="Type your question", placeholder="Enter your question here...")
            submit_btn = gr.Button("Submit Question")
            response_output = gr.Textbox(label="Response")
            submit_btn.click(pdf_processor.ask, inputs=question_input, outputs=response_output)
    
    app.launch()
