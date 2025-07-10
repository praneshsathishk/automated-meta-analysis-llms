import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNK_CSV = "data/fulltext_chunks.csv"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "chunk_metadata.csv")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def vectorize_chunks():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("🔍 Loading chunk data...")
    df = pd.read_csv(CHUNK_CSV)
    texts = df["chunk_text"].tolist()

    print(f"📏 Embedding {len(texts)} chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ FAISS index saved to {INDEX_FILE}")

    # Save metadata (needed to map search results to chunk info)
    metadata_df = df[["PMCID", "chunk_id", "chunk_text"]].copy()
    metadata_df.to_csv(METADATA_FILE, index=False)
    print(f"✅ Metadata saved to {METADATA_FILE}")
