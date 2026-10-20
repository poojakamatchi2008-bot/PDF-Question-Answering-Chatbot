# PDF Question Answering Chatbot

## About the Project
A PDF Question Answering Chatbot built using Python that allows users to upload PDF documents and ask questions based on their content. It retrieves relevant information and generates answers using a locally running Large Language Model (LLM).

## Features
- Upload PDF documents through a Streamlit interface
- Extract text from PDF files
- Split text into smaller chunks
- Generate text embeddings using Hugging Face
- Retrieve relevant information using FAISS vector search
- Generate answers using Ollama and a local LLM
- Display answers through an interactive web interface

## Technologies Used
- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Ollama
- Llama 3.2

## Project Workflow
PDF Upload → Text Extraction → Text Chunking → Embeddings → FAISS Similarity Search → LLM Answer Generation

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/PDF-Question-Answering-Chatbot.git
cd PDF-Question-Answering-Chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
Download Ollama from https://ollama.com/

Pull the model:
```bash
ollama pull llama3.2
```

### 4. Run the Application
```bash
python -m streamlit run app.py
```

Upload a PDF document and enter a question to receive an answer based on the document.

## Note
The language model runs locally using Ollama. Response speed depends on the selected model and available system resources.
