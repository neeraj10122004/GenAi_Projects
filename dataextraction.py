from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from PyPDF2 import PdfReader

def pdf_to_txt(pdf_path, txt_path):
    """Converts a PDF file to a TXT file."""
    try:
        # Read the PDF
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extract text from each page
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Save the text to a file
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
        
        print(f"PDF converted to TXT successfully! File saved at: {txt_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Paths for the input PDF and output TXT file
pdf_path = "resume.pdf"  # Replace with your PDF file path
txt_path = "output.txt"  # Replace with your desired TXT file path

# Convert PDF to TXT
pdf_to_txt("resume.pdf","resume.txt")



loader = WebBaseLoader("http://localhost:8000/resume.txt")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)


question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)

print(docs[0])

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
)

response_message = model.invoke(
    "tell about skills of neeraj sai teja akula"
)

print(response_message.content)