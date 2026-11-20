import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_PERSIST_DIR = "./chroma_db"


def build_vector_store(pdf_path: str):
    """Loads a PDF, splits text recursively into chunks, and stores embeddings in ChromaDB."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"[1/3] Loading PDF document: {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"      Loaded {len(documents)} page(s).")

    print("[2/3] Splitting text into recursive chunks ...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"      Created {len(chunks)} text chunks.")

    print("[3/3] Generating vector embeddings and storing in ChromaDB ...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("      Vector store built and persisted successfully!\n")
    return vector_store


def format_docs_with_citations(docs):
    """Formats retrieved document chunks along with their 1-indexed page numbers."""
    formatted_chunks = []
    for doc in docs:
        # PyPDFLoader stores 0-indexed page numbers in metadata['page']
        page_num = doc.metadata.get("page", 0) + 1
        source_file = os.path.basename(doc.metadata.get("source", "Document"))
        formatted_chunks.append(
            f"[Source: {source_file}, Page {page_num}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def create_rag_chain(vector_store):
    """Creates a strict RAG chain that answers only from retrieved context with page citations."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1  # Low temperature to prevent hallucinations
    )

    system_prompt = (
        "You are a strict Document Q&A Assistant. Answer the user's question ONLY using "
        "the provided context below. Do not use any outside knowledge.\n"
        "Rules:\n"
        "1. Every fact or claim in your answer MUST include an inline source page citation "
        "in the format (Page X).\n"
        "2. If the answer cannot be found in the provided context, respond strictly with: "
        "'I could not find the answer to that question in the provided document.'\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | format_docs_with_citations,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please set it in your .env file or environment.")
        sys.exit(1)

    print("==========================================================")
    print("   DATHEON TASK 2: DOCUMENT Q&A ASSISTANT (RAG + CHROMA)  ")
    print("==========================================================")

    pdf_path = input("Enter the path to your PDF file (e.g., sample.pdf): ").strip()

    try:
        vector_store = build_vector_store(pdf_path)
    except Exception as e:
        print(f"Failed to process PDF: {e}")
        sys.exit(1)

    rag_chain, retriever = create_rag_chain(vector_store)

    print("Document indexed! Ask questions below (type 'exit' to quit).\n")
    while True:
        query = input("Your Question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting Document Q&A Assistant.")
            break

        # Retrieve top matching chunks to show explicit source pages
        retrieved_docs = retriever.invoke(query)
        cited_pages = sorted({doc.metadata.get("page", 0) + 1 for doc in retrieved_docs})

        print("\nThinking...")
        answer = rag_chain.invoke(query)

        print(f"\nAnswer:\n{answer}")
        print(f"\nRetrieved Context Pages: {cited_pages}")
        print("-" * 58 + "\n")


if __name__ == "__main__":
    main()
