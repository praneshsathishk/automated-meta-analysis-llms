import os
from src.utils import call_llama3


def generate_clarifying_questions(user_input):
    """
    Uses LLaMA 3 to generate clarifying questions based on vague user input.
    """
    prompt = f"""
You are an expert research assistant. A user has given the following general research idea:

"{user_input}"

Please generate 3–5 clarifying questions to help the user refine this topic into a precise, answerable research question.
    
Your output should just be a numbered list of questions.
"""
    return call_llama3(prompt)


def finalize_research_question(detailed_input, prompt_type="zero-shot"):
    """
    Uses LLaMA 3 to generate a well-structured biomedical research question based on refined input.
    """
    if prompt_type == "zero-shot":
        prompt = f"""
You are an expert in biomedical research. Based on the following user input, write a clear, structured, and specific research question suitable for use in a systematic review or meta-analysis:

"{detailed_input}"

The output should be a single question, phrased clearly and professionally.
"""
    elif prompt_type == "few-shot":
        prompt = f"""
You are an expert in writing biomedical research questions. Given vague or slightly detailed input, rewrite it into a structured research question.

Examples:

Input: Whether omega-3 supplements help memory in older adults  
Output: Do omega-3 fatty acid supplements improve short-term memory in individuals aged 65+?

Input: Does exercise help teenagers with anxiety?  
Output: Does a structured aerobic exercise program reduce symptoms of anxiety in adolescents aged 13–18?

Now do this one:

Input: {detailed_input}  
Output:
"""
    else:
        raise ValueError("Unsupported prompt type. Use 'zero-shot' or 'few-shot'.")

    return call_llama3(prompt)


def generate_and_save_abstract_prompt(research_question, prompt_type="few-shot"):
    """
    Generates an abstract screening prompt based on the research question and prompt type,
    then saves it to prompts/abstract_prompt.txt (relative to project root).
    """

    if prompt_type == "zero-shot":
        prompt_text = f"""
You are an expert biomedical researcher performing abstract screening for a meta-analysis.

Research Question: "{research_question}"

For each abstract provided, respond with "Include" if the abstract is relevant to the research question and should be included in the meta-analysis; otherwise respond with "Exclude".

Abstract:
{{abstract}}

Decision:
"""
    elif prompt_type == "few-shot":
        prompt_text = f"""
You are an expert biomedical researcher performing abstract screening for a meta-analysis.

Research Question: "{research_question}"

Decide if the following abstracts are relevant to the research question and should be included in the meta-analysis. Respond ONLY with "Include" or "Exclude".

Examples:

Abstract: Omega-3 fatty acids have been shown to improve memory in elderly patients aged 65 and above...  
Decision: Include

Abstract: This study analyzes the impact of daily vitamin D supplementation on bone density in children...  
Decision: Exclude

Abstract:
{{abstract}}

Decision:
"""
    elif prompt_type == "chain-of-thought":
        prompt_text = f"""
You are an expert biomedical researcher screening abstracts for a meta-analysis.

Research Question: "{research_question}"

For each abstract, reason step-by-step about its relevance to the research question. Then conclude with "Include" or "Exclude".

Abstract:
{{abstract}}

Reasoning:
"""
    else:
        raise ValueError("Unsupported prompt type for abstract screening.")

    # Determine absolute path to prompts directory
    prompts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
    os.makedirs(prompts_dir, exist_ok=True)

    prompt_file = os.path.join(prompts_dir, "abstract_prompt.txt")

    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt_text.strip())

    print(f"✅ Abstract screening prompt saved to {prompt_file}")
