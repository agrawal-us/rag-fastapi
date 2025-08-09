import os, glob, pathlib, argparse
from typing import List

from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# OpenAI wrappers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS
from langchain_community.vectorstores import FAISS

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
FAISS_DIR   = os.getenv("FAISS_DIR",   "./faiss_index")

# -------- sample fallback --------
RAW_DOCS = [
    ("Embeddings guide (toy)",
     "Embeddings turn text into dense vectors that capture semantic meaning. "
     "They enable similarity search, clustering, and RAG retrieval. "
     "Choose smaller, cheaper models for fast retrieval and larger ones for accuracy."),
    ("Chunking notes",
     "Chunk text into overlapping windows so the retriever can match user queries. "
     "Common sizes are 300-800 characters with overlaps of 50-200. "
     "Keep chunks self-contained and include source metadata for citation."),
    ("Prompting tips",
     "When generating answers from retrieved context, instruct the model to use only the provided chunks. "
     "Ask for citations and avoid fabricating facts that are not present in the context.")
]

def load_documents() -> List[Document]:
    paths = glob.glob("data/*.txt")
    if not paths:
        return [Document(page_content=t, metadata={"title": title}) for title, t in RAW_DOCS]
    docs = []
    for p in paths:
        text = pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"title": pathlib.Path(p).name}))
    return docs

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
    return splitter.split_documents(docs)

def build_faiss(chunks: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.from_documents(chunks, embedding=embeddings)

def save_faiss(db: FAISS, path: str = FAISS_DIR):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    db.save_local(path)

def load_faiss(path: str = FAISS_DIR) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    # allow_dangerous_deserialization is required in newer LC versions
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

SYSTEM_PROMPT = (
    "You are a precise assistant. Use ONLY the provided context to answer. "
    "If the answer isn't in the context, say you don't know."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user",
     "Question: {question}\n\n"
     "Context:\n{context}\n\n"
     "Answer succinctly and cite sources like [title] where relevant.")
])

def format_docs(docs: List[Document]) -> str:
    return "\n---\n".join(f"[{d.metadata.get('title','Source')}] {d.page_content}" for d in docs)

def build_chain(retriever):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
    )
    return chain

def ingest():
    print("Ingest: loading and chunking docs…")
    docs = load_documents()
    chunks = split_docs(docs)
    print(f"Ingest: {len(docs)} docs -> {len(chunks)} chunks")

    db = build_faiss(chunks)
    save_faiss(db)
    print(f"Ingest: FAISS saved to {FAISS_DIR} (files like index.faiss / index.pkl)")

def query_once(q: str):
    db = load_faiss()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    chain = build_chain(retriever)

    out = chain.invoke(q)
    print("\nAnswer:\n" + out.content.strip())

    print("\nTop matches:")
    for i, d in enumerate(retriever.get_relevant_documents(q), 1):
        print(f"  {i}. {d.metadata.get('title', f'doc-{i}')} (chars: {len(d.page_content)})")

def repl():
    db = load_faiss()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    chain = build_chain(retriever)
    print("FAISS loaded. Ask questions (type 'exit').\n")

    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        out = chain.invoke(q)
        print("\nAnswer:\n" + out.content.strip())
        print("\nTop matches:")
        for i, d in enumerate(retriever.get_relevant_documents(q), 1):
            print(f"  {i}. {d.metadata.get('title', f'doc-{i}')} (chars: {len(d.page_content)})")
        print()

def main():
    ap = argparse.ArgumentParser(description="Console RAG with FAISS")
    ap.add_argument("--reindex", action="store_true", help="Rebuild FAISS from ./samples")
    ap.add_argument("--query", type=str, help="Run one query and exit")
    args = ap.parse_args()

    if args.reindex:
        ingest()

    if args.query:
        query_once(args.query)
    else:
        # if FAISS doesn’t exist yet, build it
        if not pathlib.Path(FAISS_DIR).exists():
            ingest()
        repl()

if __name__ == "__main__":
    main()