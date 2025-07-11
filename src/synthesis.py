import pandas as pd
from src.utils import call_llama3
import os

SYNTHESIS_PROMPT_PATH = "prompts/synthesis_prompt.txt"
EXTRACTED_DATA_CSV = "outputs/extracted_data.csv"
SYNTHESIS_OUTPUT_PATH = "outputs/meta_analysis_summary.txt"

def load_synthesis_prompt():
    with open(SYNTHESIS_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def load_extracted_data():
    df = pd.read_csv(EXTRACTED_DATA_CSV)
    return df

def build_prompt(prompt_template: str, extracted_df: pd.DataFrame):
    # Format the extracted data into a readable input for the LLM
    study_blocks = []
    for _, row in extracted_df.iterrows():
        block = f"""
🧪 **Study: {row.get('PMCID', 'Unknown')}**
- Design: {row.get('study_design', 'N/A')}
- Population: {row.get('population', 'N/A')}
- Intervention: {row.get('intervention', 'N/A')}
- Comparator: {row.get('comparator', 'N/A')}
- Outcome: {row.get('outcomes', 'N/A')}
- Effect: {row.get('effect_size', 'N/A')}
- Sample Size: {row.get('sample_size', 'N/A')}
- Conclusion: {row.get('conclusion', 'N/A')}
        """.strip()
        study_blocks.append(block)

    studies_section = "\n\n".join(study_blocks)
    
    # 🔍 Debug print: number of studies and preview of first one
    print(f"\n📊 Number of studies: {len(study_blocks)}")
    if study_blocks:
        print(f"\n🔍 First study block:\n{study_blocks[0]}\n")

    # Replace the placeholder in the prompt template
    final_prompt = prompt_template.replace("{{EXTRACTED_STUDIES}}", studies_section)

    # 🔍 Debug print: preview the full prompt
    print("\n📄 Final Prompt Preview (first 1000 chars):\n")
    print(final_prompt[:1000])  # Print only first 1000 chars for brevity

    return final_prompt

def synthesize_meta_analysis():
    print("\n🧠 [Step 8] SYNTHESIZING META-ANALYSIS")

    prompt_template = load_synthesis_prompt()
    extracted_df = load_extracted_data()

    if extracted_df.empty:
        print("⚠️ Extracted data CSV is empty. Aborting synthesis.")
        return "ERROR: No data to synthesize."

    final_prompt = build_prompt(prompt_template, extracted_df)

    response = call_llama3(final_prompt)

    os.makedirs("outputs", exist_ok=True)
    with open(SYNTHESIS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(response.strip())

    print(f"\n✅ Meta-analysis synthesis saved to {SYNTHESIS_OUTPUT_PATH}")
    return response
