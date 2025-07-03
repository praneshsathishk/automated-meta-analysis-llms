import subprocess

def call_llama3(prompt, model="llama3"):
    """
    Calls LLaMA 3 using Ollama with the given prompt and returns the response.
    """
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=prompt.encode("utf-8"))

    if process.returncode != 0:
        raise RuntimeError(f"Error from LLaMA: {stderr.decode()}")
    
    return stdout.decode("utf-8")


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
