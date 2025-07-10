import os
import csv
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

CHUNK_SIZE = 1000  # ~1000 characters per chunk
CHUNK_OVERLAP = 200
XML_DIR = "data/fulltext_xml"
OUTPUT_CSV = "data/fulltext_chunks.csv"

def extract_text_from_xml(xml_path):
    """
    Parses full-text XML and extracts body text from sections and paragraphs.
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "lxml-xml")

    body = soup.find("body")
    if body is None:
        return ""

    paragraphs = []
    for sec in body.find_all(["sec", "p"]):
        if sec.name == "sec":
            title = sec.find("title")
            if title:
                paragraphs.append(title.get_text(separator=" ", strip=True))
        paragraphs.extend(p.get_text(separator=" ", strip=True) for p in sec.find_all("p"))

    return "\n".join(paragraphs)


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits long text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_all_fulltexts():
    rows = []
    for filename in tqdm(os.listdir(XML_DIR)):
        if not filename.endswith(".xml"):
            continue

        pmcid = filename.replace(".xml", "")
        xml_path = os.path.join(XML_DIR, filename)

        try:
            full_text = extract_text_from_xml(xml_path)
            chunks = split_into_chunks(full_text)

            for i, chunk in enumerate(chunks):
                rows.append({
                    "PMCID": pmcid,
                    "chunk_id": f"{pmcid}_chunk_{i+1}",
                    "chunk_text": chunk
                })

        except Exception as e:
            print(f"❌ Failed to process {pmcid}: {e}")

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(rows)} chunks to {OUTPUT_CSV}")

