from Bio import Entrez
import pandas as pd
import os
import time

Entrez.email = "your-email@example.com"  # ← Replace with your email for NCBI compliance


def search_pubmed(query, max_results=100):
    """Search PubMed and return a list of PMIDs."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_pubmed_metadata(pmids):
    """Fetch metadata for each PMID."""
    all_metadata = []

    for start in range(0, len(pmids), 10):
        time.sleep(0.4)  # Rate limit to avoid being blocked
        end = start + 10
        batch = pmids[start:end]
        ids_str = ",".join(batch)

        handle = Entrez.efetch(db="pubmed", id=ids_str, rettype="medline", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for article in records["PubmedArticle"]:
            info = {}

            medline = article["MedlineCitation"]
            article_data = medline.get("Article", {})
            journal_data = article_data.get("Journal", {})
            authors_list = article_data.get("AuthorList", [])

            info["PMID"] = medline.get("PMID", "")
            info["Title"] = article_data.get("ArticleTitle", "")
            info["Abstract"] = article_data.get("Abstract", {}).get("AbstractText", [""])[0]
            info["Journal"] = journal_data.get("Title", "")
            info["PubDate"] = journal_data.get("JournalIssue", {}).get("PubDate", {}).get("Year", "")
            info["Source"] = journal_data.get("ISOAbbreviation", "")
            info["Authors"] = "; ".join([
                f"{a.get('LastName', '')} {a.get('ForeName', '')}"
                for a in authors_list if "LastName" in a and "ForeName" in a
            ])

            # ✅ SAFE: Only extract affiliations if they exist and are valid
            info["School/Company"] = "; ".join([
                a["AffiliationInfo"][0]["Affiliation"]
                for a in authors_list
                if "AffiliationInfo" in a and len(a["AffiliationInfo"]) > 0 and "Affiliation" in a["AffiliationInfo"][0]
            ])

            info["DOI"] = ""
            info["PMCID"] = ""
            info["Impact Factor"] = "N/A"  # Placeholder — requires external source

            # Check for DOI and PMCID in PubmedData
            pubmed_data = article.get("PubmedData", {})
            article_ids = pubmed_data.get("ArticleIdList", [])

            for article_id in article_ids:
                if article_id.attributes["IdType"] == "doi":
                    info["DOI"] = str(article_id)
                elif article_id.attributes["IdType"] == "pmc":
                    info["PMCID"] = str(article_id)

            all_metadata.append(info)

    return all_metadata


def save_metadata_to_csv(metadata, filename="outputs/papers.csv"):
    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(metadata)
    df.to_csv(filename, index=False)
    print(f"[✅] Saved metadata for {len(df)} papers to {filename}")


# Optional test block
if __name__ == "__main__":
    query = "aspirin AND heart attack AND primary prevention AND elderly"
    pmids = search_pubmed(query, max_results=50)
    metadata = fetch_pubmed_metadata(pmids)
    save_metadata_to_csv(metadata)