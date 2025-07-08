from src.research_question_generator import call_llama3

def generate_keywords(research_question):
    """
    Uses LLaMA to generate PubMed-friendly search terms from a research question.
    """
    prompt = f"""
You are an expert biomedical researcher. Based on the following research question, generate a concise, extremely brief PubMed search query using 1-3 core keywords.

Avoid long phrases, quotes, and too many field tags. Use Boolean operators (AND, OR) if needed, but keep the query broad and simple.

Research Question:
"{research_question}"

Output only the search string that should be passed to PubMed (no explanations, no bullet points).

Use Boolean operators like AND, OR if needed.
"""
    return call_llama3(prompt).strip()
