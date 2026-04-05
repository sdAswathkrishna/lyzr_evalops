"""
build_vectorstore.py — Embedding pipeline for the ACME support knowledge base.

WHAT THIS SCRIPT DOES (read before anything else):
====================================================

  Raw text files
       │
       ▼
  [1] LOAD documents          → each file becomes a Document(page_content, metadata)
       │
       ▼
  [2] SPLIT into chunks       → RecursiveCharacterTextSplitter breaks long docs into
       │                         overlapping windows so no single chunk is too long
       ▼
  [3] EMBED each chunk        → OpenAI text-embedding-3-small turns each chunk
       │                         into a 1536-dimensional float vector
       ▼
  [4] STORE in FAISS index    → FAISS builds an IVF index that allows sub-millisecond
       │                         nearest-neighbour search at query time
       ▼
  [5] PERSIST to disk         → index.faiss + index.pkl saved under vectorstore/
       │
       ▼
  At query time (inside the agent):
       query string
         │
         ▼
       Embed query → same model → 1536-d vector
         │
         ▼
       FAISS.similarity_search(query, k=4) → top-4 most similar chunks
         │
         ▼
       chunks returned as context to the LLM

WHY CHUNKING MATTERS:
  OpenAI embedding models have a token limit (~8000 for text-embedding-3-small).
  More importantly: a 5-page doc embedded as one vector loses fine-grained meaning.
  A question about "SSO setup" should match the SSO paragraph, not the entire manual.
  Chunking creates focused vectors. Overlap (200 chars) prevents answers from being
  cut mid-sentence at chunk boundaries.

WHY FAISS?
  FAISS (Facebook AI Similarity Search) is an open-source library for dense vector
  similarity search. It uses approximate nearest-neighbour (ANN) algorithms.
  For small datasets (<100k vectors) it's perfect: zero infrastructure, runs in-process,
  persists as two files. For production at scale, you'd swap to Pinecone/Weaviate/pgvector.
"""

import os, sys, json
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]          # lyzr_comparison/
DOCS_DIR = Path(__file__).parent / "docs"
VS_DIR   = Path(__file__).parent / "vectorstore"    # where index files land
VS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_documents(docs_dir: Path):
    """
    Load all .txt files from the docs directory.

    DirectoryLoader iterates the folder and creates one Document per file.
    Each Document has:
      .page_content  — the raw text of the file
      .metadata      — dict with 'source' key (file path)

    LangChain also has loaders for PDF (PyPDFLoader), Word (Docx2txtLoader),
    HTML (BSHTMLLoader), CSV (CSVLoader), Notion, Confluence, GitHub, etc.
    """
    print("\n📂 STEP 1: Loading documents...")
    loader = DirectoryLoader(
        str(docs_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"   Loaded {len(docs)} documents")
    for d in docs:
        fname = Path(d.metadata["source"]).name
        print(f"   • {fname}: {len(d.page_content):,} chars")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SPLIT (Chunking)
# ══════════════════════════════════════════════════════════════════════════════
def split_documents(docs):
    """
    Split documents into smaller overlapping chunks.

    RecursiveCharacterTextSplitter tries these separators in order:
      \\n\\n  (paragraph break — best natural boundary)
      \\n    (line break)
      ' '   (word boundary)
      ''    (character — last resort)

    chunk_size=800:   Each chunk ≤ 800 chars (~150-200 tokens). This fits comfortably
                      within the embedding model's context and keeps chunks focused.
    chunk_overlap=150: Last 150 chars of chunk N are repeated at the start of chunk N+1.
                      This prevents an answer from being split across two chunks.

    The result: each chunk is a stand-alone paragraph that can be embedded independently.
    """
    print("\n✂️  STEP 2: Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"   {len(docs)} documents → {len(chunks)} chunks")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    # Show a sample chunk so you can see what the embedder will receive
    print("\n   📌 Sample chunk (chunk #3):")
    print("   " + "─" * 60)
    print("   " + chunks[2].page_content[:400].replace("\n", "\n   "))
    print("   " + "─" * 60)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EMBED + STEP 4 — INDEX
# ══════════════════════════════════════════════════════════════════════════════
def build_and_save_index(chunks, vs_dir: Path):
    """
    Embed every chunk and build the FAISS vector index.

    OpenAIEmbeddings(model="text-embedding-3-small"):
      - Calls POST api.openai.com/v1/embeddings for each batch of chunks
      - Returns a 1536-dimension float32 vector per chunk
      - "3-small" costs $0.020 per million tokens — very cheap for POC
      - Alternatively: "text-embedding-ada-002" (older) or "3-large" (3072-d, costlier)

    FAISS.from_documents(chunks, embeddings):
      - Internally calls embeddings.embed_documents([c.page_content for c in chunks])
      - Builds a FlatL2 index (exact search, fine for <10k vectors)
      - Stores both the index (numerical) and the docstore (text + metadata)

    Why not embed at query time in real-time?
      Because embedding all documents every time the server starts is slow and expensive.
      We embed once, save to disk, and load the ready index on startup.
    """
    print("\n🔢 STEP 3+4: Embedding chunks and building FAISS index...")
    print("   (calling OpenAI Embeddings API — text-embedding-3-small)")
    print("   Each chunk → 1536-dimensional float vector")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # This single call: embeds all chunks + builds the FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    total_vectors = vectorstore.index.ntotal
    dims = vectorstore.index.d
    print(f"   ✅ Index built: {total_vectors} vectors, {dims} dimensions each")

    # ── STEP 5 — PERSIST ──────────────────────────────────────────────────────
    print(f"\n💾 STEP 5: Saving index to {vs_dir}/")
    vectorstore.save_local(str(vs_dir))
    print(f"   Saved: index.faiss ({(vs_dir / 'index.faiss').stat().st_size // 1024} KB)")
    print(f"   Saved: index.pkl   ({(vs_dir / 'index.pkl').stat().st_size // 1024} KB)")

    return vectorstore


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SMOKE TEST: verify the index works
# ══════════════════════════════════════════════════════════════════════════════
def smoke_test(vs_dir: Path):
    """
    Reload the saved index and run a few test queries.

    This simulates exactly what the agent will do at runtime:
      1. User message arrives
      2. Message embedded into a query vector
      3. FAISS finds k=4 most similar chunk vectors (cosine similarity via L2 on normalised vecs)
      4. Top-k chunks returned as context
      5. LLM receives: system_prompt + context + user_question

    The similarity score shown is L2 distance (lower = more similar).
    A score < 1.0 is generally considered a strong match.
    """
    print("\n🧪 STEP 6: Smoke test — reloading index and running queries...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    vs = FAISS.load_local(
        str(vs_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we wrote this index ourselves
    )

    test_queries = [
        "How do I configure SSO with Okta?",
        "What is the price of the Growth plan?",
        "My account is locked after too many failed login attempts",
        "How does data retention work for cancelled accounts?",
    ]

    for q in test_queries:
        results = vs.similarity_search_with_score(q, k=2)
        print(f"\n   Q: \"{q}\"")
        for doc, score in results:
            src = Path(doc.metadata["source"]).name
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"   ↳ [{score:.3f}] {src}: {snippet}...")

    print("\n✅ Vector store is working correctly!\n")
    return vs


# ══════════════════════════════════════════════════════════════════════════════
# SAVE BUILD MANIFEST (for the agent to know if index is fresh)
# ══════════════════════════════════════════════════════════════════════════════
def save_manifest(docs_dir: Path, vs_dir: Path, chunks):
    manifest = {
        "docs_dir": str(docs_dir),
        "num_chunks": len(chunks),
        "files": [f.name for f in sorted(docs_dir.glob("*.txt"))],
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 800,
        "chunk_overlap": 150,
    }
    with open(vs_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Manifest saved to {vs_dir}/manifest.json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  ACME Support Knowledge Base — Embedding Pipeline")
    print("=" * 65)

    docs   = load_documents(DOCS_DIR)
    chunks = split_documents(docs)
    vs     = build_and_save_index(chunks, VS_DIR)
    smoke_test(VS_DIR)
    save_manifest(DOCS_DIR, VS_DIR, chunks)

    print("\n" + "=" * 65)
    print("  Pipeline complete. Vector store is ready for the agent.")
    print("=" * 65)
