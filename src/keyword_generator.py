from src.research_question_generator import call_llama3

def generate_keywords(research_question):
    """
    Uses LLaMA to generate PubMed-friendly search terms from a research question.
    """
    prompt = f"""
You are an expert biomedical researcher. Based on the following research question, generate a concise PubMed search query using 2–5 basic keywords.

- DO NOT use field tags like [ti], [tiab], [mesh], or [yr].
- DO NOT use comparison operators like ≥ or quotes like "elderly".
- DO NOT use parentheses unless absolutely necessary.
- At the end of the query, always add this filter:
  NOT (meta-analysis[pt] OR review[pt])

Use simple Boolean logic like AND/OR, and ensure the search is broad enough to return at least 30 results.

Research Question:
"{research_question}"

Output only the search string that should be passed to PubMed (no explanations, no bullet points).

"""
    return call_llama3(prompt).strip()
