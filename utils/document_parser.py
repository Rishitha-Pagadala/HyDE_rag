from PyPDF2 import PdfReader
from docx import Document

def parse_document(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file.read().decode("utf-8")
    return text
