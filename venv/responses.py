from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_response(user_input: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        return (chat_completion.choices[0].message.content)
    except Exception as e:
        return f"Sorry, I couldn't generate a response. Error: {e}"
