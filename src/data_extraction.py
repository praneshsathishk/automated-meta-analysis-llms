import pandas as pd
import os
import json
import re
from src.utils import call_llama3
from collections import defaultdict
from tqdm import tqdm

def extract_from_chunk(title, chunk_text, pmcid):
    system_prompt = (
        "You are an expert at reading scientific article chunks and extracting detailed information "
        "for meta-analysis. Extract structured data in JSON format with fields:\n"
        "- population (demographics, health status, subgroups)\n"
        "- intervention (type, dosage, frequency, duration, method)\n"
        "- comparator (control or baseline groups)\n"
        "- outcomes (primary, secondary, measurement methods)\n"
        "- effect_size (statistics, risk ratios, CIs, p-values)\n"
        "- study_design (type, randomization, blinding, follow-up)\n"
        "- sample_size (total and group sizes)\n"
        "- conclusion (key findings, statistical & practical significance)\n"
        "Be as detailed and specific as possible."
    )
    user_prompt = (
        f"Title: {title}\n\nText Chunk:\n{chunk_text}\n\n"
        "Extract exactly in this JSON format:\n"
        "{\n"
        '  "population": "...",\n'
        '  "intervention": "...",\n'
        '  "comparator": "...",\n'
        '  "outcomes": ["...", "..."],\n'
        '  "effect_size": "...",\n'
        '  "study_design": "...",\n'
        '  "sample_size": "...",\n'
        '  "conclusion": "...",\n'
        f'  "PMCID": "{pmcid}"\n'
        "}"
    )
    prompt = system_prompt + "\n\n" + user_prompt
    try:
        return call_llama3(prompt, model="llama3")
    except Exception as e:
        print(f"Error calling LLaMA 3 on chunk: {e}")
        return None

def safe_json_parse(llm_output):
    llm_output = llm_output.strip()
    match = re.search(r"\{[\s\S]*?\}", llm_output)
    if not match:
        return None
    json_str = match.group(0).replace("'", '"')
    json_str = re.sub(r'(:\s*)([A-Za-z][\w\s\-/&()]+)([,\n}])', r'\1"\2"\3', json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None

def merge_fields(agg_list):
    filtered = [str(x).strip() for x in agg_list if x and x.lower() not in ['not specified', 'none', 'not mentioned', '']]
    unique = list(dict.fromkeys(filtered))
    return "; ".join(unique) if unique else ""

def merge_lists(agg_lists):
    combined = []
    for lst in agg_lists:
        if isinstance(lst, list):
            combined.extend(lst)
        elif isinstance(lst, str):
            combined.append(lst)
    unique = list(dict.fromkeys([item.strip() for item in combined if item and item.lower() != 'none']))
    return unique

def extract_all():
    SCREENING_RESULTS = "outputs/fulltext_screening_results.csv"
    CHUNK_METADATA = "faiss_index/chunk_metadata.csv"

    screening_df = pd.read_csv(SCREENING_RESULTS)
    screening_df["PMCID"] = screening_df["PMCID"].astype(str).str.strip()
    chunks_df = pd.read_csv(CHUNK_METADATA, quotechar='"', encoding='utf-8')
    chunks_df["PMCID"] = chunks_df["PMCID"].astype(str).str.strip()

    all_results = []

    for pmcid in tqdm(screening_df["PMCID"].unique()):
        matching_papers = screening_df[screening_df["PMCID"] == pmcid]
        if matching_papers.empty:
            continue
        paper_meta = matching_papers.iloc[0]

        paper_chunks = chunks_df[chunks_df["PMCID"] == pmcid].copy()
        if paper_chunks.empty:
            continue

        paper_chunks["chunk_number"] = paper_chunks["chunk_id"].apply(lambda x: int(x.split('_chunk_')[1]))
        paper_chunks = paper_chunks.sort_values("chunk_number")

        aggregated = defaultdict(list)

        for _, row in paper_chunks.iterrows():
            print(f"Extracting from PMCID {pmcid} chunk {row['chunk_number']}...")
            llm_output = extract_from_chunk(paper_meta['Title'], row['chunk_text'], pmcid)
            if not llm_output:
                continue
            parsed = safe_json_parse(llm_output)
            if not parsed:
                continue
            for key in ['population', 'intervention', 'comparator', 'effect_size', 'study_design', 'sample_size', 'conclusion']:
                if key in parsed and parsed[key]:
                    aggregated[key].append(parsed[key])
            if 'outcomes' in parsed and parsed['outcomes']:
                aggregated['outcomes'].append(parsed['outcomes'])

        final_result = {
            "population": merge_fields(aggregated['population']),
            "intervention": merge_fields(aggregated['intervention']),
            "comparator": merge_fields(aggregated['comparator']),
            "outcomes": merge_lists(aggregated['outcomes']),
            "effect_size": merge_fields(aggregated['effect_size']),
            "study_design": merge_fields(aggregated['study_design']),
            "sample_size": merge_fields(aggregated['sample_size']),
            "conclusion": merge_fields(aggregated['conclusion']),
            "PMCID": pmcid
        }
        all_results.append(final_result)

    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/extracted_data_all_papers.csv"
    pd.DataFrame(all_results).to_csv(output_file, index=False)
    print(f"✅ Saved extracted data for all papers to {output_file}")

if __name__ == "__main__":
    extract_all()
