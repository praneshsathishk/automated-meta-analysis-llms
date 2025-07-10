from src.research_question_generator import (
    generate_clarifying_questions,
    finalize_research_question,
    generate_and_save_abstract_prompt,
    generate_and_save_fulltext_prompt
)
from src.keyword_generator import generate_keywords
from src.pubmed_search import search_pubmed, fetch_pubmed_metadata, save_metadata_to_csv
from src.abstract_screening import screen_abstracts
from src.fulltext_downloader import download_fulltexts
from src.fulltext_chunking import chunk_all_fulltexts
from src.chunk_vectorizer import vectorize_chunks
from src.fulltext_screening import fulltext_screening

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

# Generate and save prompts
generate_and_save_abstract_prompt(final_question, prompt_type="few-shot")
generate_and_save_fulltext_prompt(final_question, prompt_type="few-shot")

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

screen_abstracts(input_csv_path=input_csv, output_csv_path=filtered_csv)

print(f"\n🎉 Done! Filtered papers saved to {filtered_csv}")

print("\n[Step 4] DOWNLOADING FULL-TEXT XMLS")
download_fulltexts(filtered_csv)

print("\n🎉 Done! Full-text XMLs saved to data/fulltext_xml/")

print("\n[Step 5] CHUNKING FULL-TEXT XMLS")
chunk_all_fulltexts()

print("\n🎉 Done! Full-text chunks saved to data/fulltext_chunks.csv")

print("\n[Step 6] EMBEDDING CHUNKS AND BUILDING FAISS INDEX")
vectorize_chunks()

print("\n🎉 Done! FAISS index and metadata saved to faiss_index/")

print("\n[Step 7] FULL-TEXT SCREENING")
fulltext_screening()
