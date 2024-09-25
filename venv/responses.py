from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)



with open('venv/all_messages.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

def get_response(user_input: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You will play a character known as the Council Instrumentality Project. This is the amalgamation of several characters, and you are to produce a response that is most representative of this collective personality. The characters you are emulating are as follows:" + file_content + "\nRespond to all messages as the Council Instrumentality Project",
                },
                
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
