import os
import json
import time
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Bio import Entrez

# === CONFIGURATION ===
Entrez.email = "your-email@example.com"  # Use the same format as your working code
# Entrez.api_key = "YOUR_NCBI_API_KEY"  # Comment out the API key line

MAX_RESULTS = 1000
OUTPUT_DIR = "data/meta_analysis_raw"
SPLIT_DIR = "data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

def search_pubmed_ids(term, max_results):
    print(f"🔍 Searching PubMed for: {term}")
    handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def pmid_to_pmcid(pmid):
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc")
        record = Entrez.read(handle)
        handle.close()

        links = record[0]["LinkSetDb"]
        for linkset in links:
            if linkset["LinkName"] == "pubmed_pmc":
                return linkset["Link"][0]["Id"]
        return None
    except:
        return None

def fetch_and_parse_paper(pmcid):
    try:
        time.sleep(0.4)  # Match the rate limiting from your working code
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
        tree = ET.parse(handle)
        handle.close()
        root = tree.getroot()

        title_elem = root.find(".//article-title")
        title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""

        abstract_elem = root.find(".//abstract")
        abstract = ''.join(abstract_elem.itertext()).strip() if abstract_elem is not None else ""

        body_elem = root.find(".//body")
        body = ''.join(body_elem.itertext()).strip() if body_elem is not None else ""

        return {
            "PMCID": pmcid,
            "title": title,
            "abstract": abstract,
            "body": body
        }

    except Exception as e:
        print(f"⚠️ Failed to fetch PMCID {pmcid}: {e}")
        return None

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def collect_dataset(keyword):
    # Simplify the search term - remove the complex filters that might cause issues
    search_term = f"{keyword} AND meta-analysis"  # Start simple
    pmids = search_pubmed_ids(search_term, MAX_RESULTS)

    dataset = []
    print(f"🔗 Mapping {len(pmids)} PMIDs to PMCIDs (full text)...")

    for pmid in tqdm(pmids, desc="🧭 Mapping PMIDs → PMCIDs"):
        pmcid = pmid_to_pmcid(pmid)
        if pmcid:
            paper = fetch_and_parse_paper(pmcid)
            if paper and paper["body"].strip():
                dataset.append(paper)

    fname = keyword.replace(" ", "_").lower()
    raw_path = os.path.join(OUTPUT_DIR, f"{fname}_meta_analysis_full.jsonl")
    save_jsonl(dataset, raw_path)
    print(f"\n✅ Saved {len(dataset)} papers to {raw_path}")
    return dataset

def split_dataset(dataset, keyword):
    random.seed(42)
    random.shuffle(dataset)

    train, temp = train_test_split(dataset, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    fname = keyword.replace(" ", "_").lower()
    save_jsonl(train, os.path.join(SPLIT_DIR, f"{fname}_train.jsonl"))
    save_jsonl(val, os.path.join(SPLIT_DIR, f"{fname}_val.jsonl"))
    save_jsonl(test, os.path.join(SPLIT_DIR, f"{fname}_test.jsonl"))

    print(f"\n📂 Dataset split: train ({len(train)}), val ({len(val)}), test ({len(test)})")

def main():
    print("🧠 Meta-Analysis Dataset Collector")
    keyword = input("Enter a topic keyword to search meta-analyses for (e.g., 'cardiology', 'diabetes'): ").strip()
    if not keyword:
        print("❌ No keyword entered. Exiting.")
        return

    papers = collect_dataset(keyword)
    if not papers:
        print("❌ No papers collected. Exiting.")
        return

    split_dataset(papers, keyword)

if __name__ == "__main__":
    main()