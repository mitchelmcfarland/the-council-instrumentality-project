# responses.py

from groq import Groq
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')  # e.g., 'us-west1-gcp'
)

# Connect to your index
index_name = 'council-instrumentality-project'  # Replace with your index name
pinecone_index = pinecone.Index(index_name)

def get_relevant_messages(user_input, top_k=5):
    user_embedding = embedding_model.encode([user_input]).tolist()
    result = pinecone_index.query(
        vector=user_embedding[0],
        top_k=top_k,
        include_metadata=True
    )
    relevant_messages = [match['metadata']['text'] for match in result['matches']]
    return relevant_messages

def get_response(user_input: str) -> str:
    try:
        relevant_messages = get_relevant_messages(user_input)
        if not relevant_messages:
            context = "No relevant messages found."
        else:
            context = "\n".join(relevant_messages)
        system_prompt = (
            "You are the Council Instrumentality Project, an amalgamation of several characters. "
            "Based on the following messages, respond appropriately.\n\n"
            f"{context}"
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate a response. Error: {e}"
