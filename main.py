from src.research_question_generator import generate_clarifying_questions, finalize_research_question, generate_and_save_abstract_prompt
from src.keyword_generator import generate_keywords
from src.pubmed_search import search_pubmed, fetch_pubmed_metadata, save_metadata_to_csv
from src.abstract_screening import screen_abstracts

print("\n[Step 1] USER INPUT AND LLaMA CLARIFICATION")

user_input = "aspirin and heart attacks"
print(f"> User input: {user_input}")

questions = generate_clarifying_questions(user_input)
print("\n> Clarifying Questions:")
print(questions)

refined_input = "Does daily aspirin reduce the risk of first heart attack in adults over 65 with no prior cardiovascular disease?"
print("\n> Refined Input:", refined_input)

final_question = finalize_research_question(refined_input, prompt_type="few-shot")
print("\n✅ Final Research Question:")
print(final_question)

# Generate and save abstract screening prompt
generate_and_save_abstract_prompt(final_question, prompt_type="few-shot")

print("\n[Step 1.5] GENERATING PUBMED SEARCH KEYWORDS")
keyword_query = generate_keywords(final_question)
print(f"> PubMed Search Query: {keyword_query}")

print("\n[Step 2] SEARCHING PUBMED AND SAVING METADATA")
pmids = search_pubmed(keyword_query, max_results=50)
print(f"> Found {len(pmids)} PMIDs")

metadata = fetch_pubmed_metadata(pmids)
save_metadata_to_csv(metadata)

print("\n🎉 Done! Metadata saved to outputs/papers.csv")

print("\n[Step 3] ABSTRACT SCREENING")
input_csv = "outputs/papers.csv"
filtered_csv = "outputs/filtered_papers.csv"
screen_abstracts(input_csv, filtered_csv)

print(f"\n🎉 Done! Filtered abstracts saved to {filtered_csv}")
