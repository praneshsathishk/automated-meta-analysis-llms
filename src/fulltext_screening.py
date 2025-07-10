import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import call_llama3
from tqdm import tqdm
import os

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index/index.faiss"
CHUNK_METADATA_PATH = "faiss_index/chunk_metadata.csv"
FULLTEXT_PROMPT_PATH = "prompts/fulltext_prompt.txt"
SCREENING_RESULTS_CSV = "outputs/fulltext_screening_results.csv"
FULL_METADATA_PATH = "outputs/filtered_papers.csv"  # <-- Make sure this exists

# Load FAISS index + chunk metadata
def load_faiss_index_and_metadata():
    index = faiss.read_index(FAISS_INDEX_PATH)
    metadata = pd.read_csv(CHUNK_METADATA_PATH)
    return index, metadata

# Load embedding model
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# Load screening prompt template
def load_screening_prompt():
    with open(FULLTEXT_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

# Retrieve top-k chunks for a paper
def get_top_k_chunks_for_paper(pmcid, index, metadata, embedder, k=5):
    paper_chunks = metadata[metadata["PMCID"] == pmcid]
    chunk_texts = paper_chunks["chunk_text"].tolist()
    if len(chunk_texts) == 0:
        return []
    chunk_embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)
    return chunk_texts[:k]

# Run LLaMA and interpret decision
def screen_paper(pmcid, chunks, prompt_template):
    chunk_str = "\n\n---\n\n".join(chunks)
    prompt = prompt_template.replace("{{CHUNKS}}", chunk_str).replace("{{PMCID}}", str(pmcid))
    response = call_llama3(prompt)

    if "include" in response.lower():
        return "Include", response
    elif "exclude" in response.lower():
        return "Exclude", response
    else:
        return "Unclear", response

# Main screening function
def fulltext_screening(pmcids=None, top_k_chunks=5):
    print("\n[Step 7] FULL-TEXT SCREENING")

    index, chunk_metadata = load_faiss_index_and_metadata()
    embedder = load_embedding_model()
    prompt_template = load_screening_prompt()
    full_metadata = pd.read_csv(FULL_METADATA_PATH)

    if pmcids is None:
        pmcids = chunk_metadata["PMCID"].unique()

    results = []
    for pmcid in tqdm(pmcids, desc="Screening papers"):
        chunks = get_top_k_chunks_for_paper(pmcid, index, chunk_metadata, embedder, k=top_k_chunks)
        if not chunks:
            print(f"⚠️ No chunks found for PMCID {pmcid}, skipping.")
            continue

        decision, response = screen_paper(pmcid, chunks, prompt_template)

        results.append({
            "PMCID": pmcid,
            "decision": decision,
            "llama_response": response
        })

    df_results = pd.DataFrame(results)

    # Filter for only included papers
    df_included = df_results[df_results["decision"] == "Include"]

    # Merge with full metadata
    df_merged = pd.merge(df_included, full_metadata, on="PMCID", how="inner")

    # Reorder and select only the required columns
    desired_columns = [
        "PMCID", "PMID", "Title", "Abstract", "Journal", "PubDate",
        "Source", "Authors", "School/Company", "DOI", "Impact Factor"
    ]
    df_final = df_merged[desired_columns]

    os.makedirs("outputs", exist_ok=True)
    df_final.to_csv(SCREENING_RESULTS_CSV, index=False)

    print(f"\n🎉 Done! Only included papers saved to {SCREENING_RESULTS_CSV}")
    return df_final
