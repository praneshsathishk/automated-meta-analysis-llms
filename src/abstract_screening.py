import os
import pandas as pd
from src.utils import call_llama3

def screen_abstracts(input_csv_path: str, output_csv_path: str, prompt_path: str = "prompts/abstract_prompt.txt"):
    """
    Screens abstracts from input CSV using the prompt template and LLaMA,
    filters relevant papers, and saves filtered results to output CSV.

    Assumes the input CSV has at least columns: 'PMID', 'Title', 'Abstract', etc.
    """

    # If prompt_path is relative, resolve absolute path relative to this file's parent
    if not os.path.isabs(prompt_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        prompt_path = os.path.join(base_dir, prompt_path)

    # Load papers metadata
    df = pd.read_csv(input_csv_path)

    # Load prompt template
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    keep_rows = []
    print(f"> Screening {len(df)} abstracts...")

    for idx, row in df.iterrows():
        abstract = row.get("Abstract", "")
        if not abstract or pd.isna(abstract):
            # Skip papers without abstract
            continue

        # Fill prompt with the abstract text
        prompt = prompt_template.replace("{abstract}", abstract)

        # Call LLaMA (or your LLM wrapper)
        response = call_llama3(prompt).strip().lower()

        # Decide inclusion by checking if response contains "include"
        if "include" in response:
            keep_rows.append(row)

        # Optional: print progress and decision
        print(f"[{idx+1}/{len(df)}] PMID {row.get('PMID', 'N/A')} - Decision: {response}")

    # Save filtered papers
    if keep_rows:
        filtered_df = pd.DataFrame(keep_rows)
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Saved {len(filtered_df)} included papers to {output_csv_path}")
    else:
        print("\n⚠️ No papers passed abstract screening.")
