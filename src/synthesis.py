import pandas as pd
from src.utils import call_llama3
import os
import traceback

SYNTHESIS_PROMPT_PATH = "prompts/synthesis_prompt.txt"
EXTRACTED_DATA_CSV = "outputs/extracted_data_all_papers.csv"
SYNTHESIS_OUTPUT_PATH = "outputs/meta_analysis_summary2.txt"

WARNING_TOKEN_LIMIT = 200000  # Adjust if needed
MAX_CHAR_PER_FIELD = 300      # Truncate very long fields
MAX_ROWS = 10                 # Limit number of studies (adjust as needed)

def load_synthesis_prompt():
    print("📥 Loading synthesis prompt...")
    with open(SYNTHESIS_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()
    if "{{EXTRACTED_STUDIES}}" not in prompt:
        raise ValueError("❌ The synthesis prompt must contain the placeholder '{{EXTRACTED_STUDIES}}'")
    return prompt

def load_extracted_data():
    print("📥 Loading extracted data CSV...")
    df = pd.read_csv(EXTRACTED_DATA_CSV, na_values=["null", "N/A", "NaN", "None"])

    # Strip whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    core_fields = [
        "population", "intervention", "comparator",
        "outcomes", "effect_size", "study_design",
        "sample_size", "conclusion"
    ]
    df_filtered = df.dropna(subset=core_fields, how="all")

    if df_filtered.empty:
        print("⚠️ No valid data to include in meta-analysis.")
    else:
        print(f"✅ Loaded {len(df_filtered)} valid studies\n")

    return df_filtered.head(MAX_ROWS)

def truncate(text, limit=MAX_CHAR_PER_FIELD):
    return (text[:limit] + '...') if isinstance(text, str) and len(text) > limit else text

def build_prompt(prompt_template: str, extracted_df: pd.DataFrame):
    print("🧱 Building prompt...")
    study_blocks = []

    for _, row in extracted_df.iterrows():
        block = f"""
🧪 **Study: {row.get('PMCID', 'Unknown')}**
- Design: {truncate(row.get('study_design'))}
- Population: {truncate(row.get('population'))}
- Intervention: {truncate(row.get('intervention'))}
- Comparator: {truncate(row.get('comparator'))}
- Outcome: {truncate(row.get('outcomes'))}
- Effect: {truncate(row.get('effect_size'))}
- Sample Size: {truncate(row.get('sample_size'))}
- Conclusion: {truncate(row.get('conclusion'))}
        """.strip()
        study_blocks.append(block)

    studies_section = "\n\n".join(study_blocks)
    print(f"\n📊 Included {len(study_blocks)} study summaries.")
    if study_blocks:
        print(f"\n🔍 First study block preview:\n{study_blocks[0]}\n")

    final_prompt = prompt_template.replace("{{EXTRACTED_STUDIES}}", studies_section)

    if len(final_prompt) > WARNING_TOKEN_LIMIT:
        print(f"⚠️ Prompt size is large ({len(final_prompt)} chars). This may exceed model capacity.")

    word_count = len(final_prompt.split())
    print(f"📏 Prompt size: {word_count} words")

    print("\n📄 Final Prompt Preview (first 1000 chars):\n")
    print(final_prompt[:1000])

    return final_prompt

def synthesize_meta_analysis():
    print("\n🧠 [Step 8] SYNTHESIZING META-ANALYSIS")
    try:
        prompt_template = load_synthesis_prompt()
        extracted_df = load_extracted_data()
        if extracted_df.empty:
            print("❌ Aborting: No valid extracted data.")
            return "ERROR: No data to synthesize."

        final_prompt = build_prompt(prompt_template, extracted_df)

        print("\n📡 Sending prompt to LLM...")
        response = call_llama3(final_prompt)
        print("\n📥 Response received from LLM.")

        os.makedirs("outputs", exist_ok=True)
        with open(SYNTHESIS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(response.strip())

        print(f"\n✅ Meta-analysis synthesis saved to {SYNTHESIS_OUTPUT_PATH}")
        return response

    except Exception as e:
        print("❌ An error occurred during synthesis.")
        traceback.print_exc()
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    synthesize_meta_analysis()
