import os
import time
import pandas as pd
from tqdm import tqdm
from Bio import Entrez

Entrez.email = "your_email@example.com"  # <-- Replace with your email (required by NCBI)

OUTPUT_DIR = "data/fulltext_xml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_fulltext_xml(pmcid, save_path):
    try:
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
        xml_content = handle.read()  # ← This returns bytes

        # ✅ Decode from bytes to UTF-8 string
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode("utf-8")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        return True
    except Exception as e:
        print(f"❌ Failed to download PMCID {pmcid}: {e}")
        return False

def download_fulltexts(csv_path):
    df = pd.read_csv(csv_path)
    pmcids = df["PMCID"].dropna().astype(str)

    print(f"📥 Attempting download of {len(pmcids)} full-text articles...")

    downloaded = 0
    for pmcid in tqdm(pmcids):
        clean_pmcid = pmcid.replace("PMC", "")  # Entrez sometimes needs just the digits
        save_path = os.path.join(OUTPUT_DIR, f"{pmcid}.xml")
        if os.path.exists(save_path):
            continue  # Skip if already downloaded

        success = download_fulltext_xml(clean_pmcid, save_path)
        if success:
            downloaded += 1
        time.sleep(0.5)  # NCBI recommends no more than 3 requests/sec

    print(f"\n✅ Downloaded {downloaded} XML files to {OUTPUT_DIR}")

