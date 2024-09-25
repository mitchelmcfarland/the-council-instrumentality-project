from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

client = InferenceClient(token=HF_API_TOKEN)

def get_response(user_input: str) -> str:
    try:
        response = client.text_generation(
            prompt=user_input,
            model="microsoft/DialoGPT-medium",
            max_new_tokens=100,  # Correct parameter
        )
        return response
    except Exception as e:
        return f"Sorry, I couldn't generate a response. Error: {e}"
