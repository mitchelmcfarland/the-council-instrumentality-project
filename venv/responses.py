# responses.py

import os
import pinecone
from semantic_router.encoders import HuggingFaceEncoder
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Initialize Pinecone and Groq
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = "message-index"
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(index_name)

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

# Initialize Encoder
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# Function to query relevant messages based on user input
def get_relevant_messages(query, top_k=5):
    query_embedding = encoder([query])
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match['metadata']['content'] for match in results['matches']]

# Function to generate a response using Groq and Pinecone context
def get_response(query):
    # Retrieve relevant messages
    docs = get_relevant_messages(query)
    
    # Create a system message with context
    system_message = (
        "You are a helpful assistant that answers questions using the context provided below.\n\n"
        "CONTEXT:\n"
        "\n---\n".join(docs)
    )
    
    # Prepare messages for the API request
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    # Generate response using Groq API
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content
