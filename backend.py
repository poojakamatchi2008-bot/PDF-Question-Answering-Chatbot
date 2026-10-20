# ============================================================
# PDF QUESTION ANSWERING CHATBOT - BACKEND
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# ------------------------------------------------------------
# 1. GET PDF PATH
# ------------------------------------------------------------

pdf_path = input("Enter PDF file path: ")


# ------------------------------------------------------------
# 2. LOAD PDF
# ------------------------------------------------------------

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("\nPDF loaded successfully!")
print("Pages:", len(documents))


# ------------------------------------------------------------
# 3. SPLIT PDF INTO CHUNKS
# ------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# ------------------------------------------------------------
# 4. CREATE HUGGINGFACE EMBEDDINGS
# ------------------------------------------------------------

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings created!")


# ------------------------------------------------------------
# 5. CREATE FAISS VECTOR DATABASE
# ------------------------------------------------------------

print("\nCreating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS database created!")


# ------------------------------------------------------------
# 6. CREATE RETRIEVER
# ------------------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)

print("Retriever ready!")


# ------------------------------------------------------------
# 7. LOAD LOCAL LLAMA 3.2
# ------------------------------------------------------------

print("\nLoading Llama 3.2...")

llm = OllamaLLM(
    model="llama3.2"
)

print("Llama 3.2 loaded!")


# ------------------------------------------------------------
# 8. QUESTION-ANSWER LOOP
# ------------------------------------------------------------

while True:

    question = input(
        "\nAsk a question (type 'exit' to stop): "
    )

    # Stop chatbot
    if question.lower().strip() == "exit":
        print("\nChatbot stopped.")
        break


    # --------------------------------------------------------
    # 9. RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    relevant_docs = retriever.invoke(question)


    # --------------------------------------------------------
    # 10. CREATE CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )


    # --------------------------------------------------------
    # 11. CREATE PROMPT
    # --------------------------------------------------------
    prompt = f"""
You are a PDF question-answering assistant.

Answer the question using ONLY the information provided in the PDF.

IMPORTANT RULES:

1. Give the answer directly to the question.
2. Use the PDF content as the main source.
3. Keep the important technical meaning and terminology from the PDF.
4. You may rewrite the PDF content in simple words, but do not change its meaning.
5. Include the necessary key points from the PDF that are directly related to the question.
6. If the question asks "What is", first give a clear definition/explanation.
7. If the PDF contains important points, characteristics, steps, types, operations,
   causes, effects, examples, or solutions related to the question, include them when relevant.
8. Do not give unrelated information.
9. Do not add general knowledge that is not supported by the PDF.
10. Do not make unsupported connections between concepts.
11. Do not unnecessarily make the answer very long.
12. If the PDF does not contain enough information, say:
"I couldn't find enough information to answer this question from the PDF."
13. Do not mention PDF context, chunks, embeddings, FAISS, retriever, or Llama.
14. Give ONLY the final answer.

PDF INFORMATION:
--------------------
{context}
--------------------

QUESTION:
{question}

ANSWER:
"""



    # --------------------------------------------------------
    # 12. GENERATE ANSWER
    # --------------------------------------------------------

    answer = llm.invoke(prompt)


    # --------------------------------------------------------
    # 13. DISPLAY ONLY THE ANSWER
    # --------------------------------------------------------

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(answer)