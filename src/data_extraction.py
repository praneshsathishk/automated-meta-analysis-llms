import pandas as pd
import os
from tqdm import tqdm
from src.utils import call_llama3
import json  # Make sure this is at the top of your file

# Paths
FINAL_SCREENING_CSV = "outputs/fulltext_screening_results.csv"
CHUNK_METADATA_PATH = "faiss_index/chunk_metadata.csv"
EXTRACTION_OUTPUT_CSV = "outputs/extracted_data.csv"

# Prompt template
EXTRACTION_PROMPT_TEMPLATE = """
You are an expert in clinical meta-analysis.

Based on the following full-text content from a research paper, extract the following information:
- Population
- Intervention
- Comparator (if any)
- Outcome(s)
- Effect Size(s) (or summary of results)
- Study Design (e.g., RCT, observational, meta-analysis, etc.)
- Sample Size
- Conclusion Summary

Paper Full Text:
-----------------
{full_text}

Format your output in JSON with keys:
"population", "intervention", "comparator", "outcomes", "effect_size", "study_design", "sample_size", "conclusion".

Provide only the JSON output, no other text before or after. This includes no disclaimers or anything of the sort.
"""


def load_data():
    screened_df = pd.read_csv(FINAL_SCREENING_CSV)
    chunk_df = pd.read_csv(CHUNK_METADATA_PATH)
    return screened_df, chunk_df


def extract_full_text(pmcid, chunk_df):
    chunks = chunk_df[chunk_df["PMCID"] == pmcid]["chunk_text"].tolist()
    return "\n\n".join(chunks)


def extract_structured_data(pmcid, full_text):
    prompt = EXTRACTION_PROMPT_TEMPLATE.replace("{full_text}", full_text[:10000])  # Truncate if needed
    response = call_llama3(prompt).strip()

    # Robust extraction of first balanced JSON object in response
    try:
        start = response.index('{')
        brace_count = 0
        end = None
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("No balanced JSON object found")

        json_str = response[start:end]
        parsed = json.loads(json_str)
        parsed["PMCID"] = pmcid
        return parsed

    except Exception as e:
        print(f"⚠️ Failed to parse response for PMCID {pmcid}: {e}")
        return {"PMCID": pmcid, "error": response.strip()}


def extract_all():
    screened_df, chunk_df = load_data()
    pmcids = screened_df["PMCID"].tolist()

    results = []
    for pmcid in tqdm(pmcids, desc="Extracting data from papers"):
        full_text = extract_full_text(pmcid, chunk_df)
        if not full_text:
            print(f"⚠️ No text found for {pmcid}")
            continue

        result = extract_structured_data(pmcid, full_text)
        results.append(result)

    df_out = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    df_out.to_csv(EXTRACTION_OUTPUT_CSV, index=False)
    print(f"\n✅ Extracted data saved to {EXTRACTION_OUTPUT_CSV}")
    return df_out


if __name__ == "__main__":
    extract_all()
