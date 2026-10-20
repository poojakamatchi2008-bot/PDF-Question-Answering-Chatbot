import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 PDF Question Answering Chatbot")

st.write("Upload a PDF and ask questions based on its content.")


# ============================================================
# LOAD EMBEDDING MODEL ONLY ONCE
# ============================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ============================================================
# LOAD LLAMA ONLY ONCE
# ============================================================

@st.cache_resource
def load_llm():

    llm = OllamaLLM(
        model="llama3.2"
    )

    return llm


# ============================================================
# PROCESS PDF
# ============================================================

@st.cache_resource
def process_pdf(pdf_bytes):

    # Save PDF temporarily
    pdf_path = "uploaded_file.pdf"

    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)


    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()


    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)


    # Embeddings
    embeddings = load_embeddings()


    # FAISS
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )


    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )


    return retriever, len(documents), len(chunks)


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ============================================================
# AFTER PDF UPLOAD
# ============================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()


    # Process only when required
    with st.spinner("Processing PDF..."):

        retriever, page_count, chunk_count = process_pdf(
            pdf_bytes
        )


    st.success(
        f"✅ PDF ready! Pages: {page_count} | Chunks: {chunk_count}"
    )


    # ========================================================
    # LOAD LLAMA
    # ========================================================

    llm = load_llm()


    # ========================================================
    # QUESTION
    # ========================================================

    question = st.text_input(
        "Ask a question from the PDF:"
    )


    # ========================================================
    # ANSWER
    # ========================================================

    if question:

        with st.spinner("Generating answer..."):

            # Retrieve relevant documents
            relevant_docs = retriever.invoke(question)


            # Create context
            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )


            # Prompt
            prompt = f"""
You are a PDF question-answering assistant.

Use the PDF information below to answer the question.

Rules:

1. Give a correct and clear answer based only on the PDF information.
2. Include the key points necessary to answer the question.
3. You can explain the information in your own simple words.
4. Keep the original meaning of the PDF correct.
5. Include important points directly related to the question.
6. If the PDF gives an explanation, definition, steps, types,
   examples, causes, effects, or solutions, include them when relevant.
7. Do not add unrelated information.
8. Do not add facts that are not supported by the PDF.
9. Do not make unsupported connections between concepts.
10. Do not unnecessarily make the answer very long.
11. If the PDF context does not contain enough information, say:
"I couldn't find enough information to answer this question from the PDF."

PDF INFORMATION:
--------------------
{context}
--------------------

QUESTION:
{question}

ANSWER:
"""


            # Generate answer
            answer = llm.invoke(prompt)


        # Display answer
        st.subheader("💡 Answer")

        st.write(answer)