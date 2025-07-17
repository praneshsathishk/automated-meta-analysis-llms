import subprocess
import openai  # pip install openai
import os

# Set your OpenAI API key (you can use an env variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly: openai.api_key = "your-key"

def call_model(prompt, model_type="ollama", model_name="llama3"):
    """
    Calls either a local Ollama model or GPT-4 via OpenAI API based on model_type.
    model_type: "ollama" or "openai"
    model_name: For ollama, the local model (e.g., "llama3"). For openai, the model ID (e.g., "gpt-4").
    """
    if model_type == "ollama":
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=prompt.encode("utf-8"))

        if process.returncode != 0:
            raise RuntimeError(f"Error from Ollama: {stderr.decode()}")

        return stdout.decode("utf-8")

    elif model_type == "openai":
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error from OpenAI: {e}")

    else:
        raise ValueError("Invalid model_type. Use 'ollama' or 'openai'.")
