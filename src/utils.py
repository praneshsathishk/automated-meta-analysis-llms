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
