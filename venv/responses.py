import os
from pinecone import Pinecone, ServerlessSpec
from semantic_router.encoders import HuggingFaceEncoder
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Pinecone API key and environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = "message-index"

# Initialize Pinecone using the new Pinecone class
pc = Pinecone(api_key=pinecone_api_key)

# Connect to the existing index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

# Initialize Encoder for text embeddings
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# Function to query relevant messages based on user input
def get_relevant_messages(query, top_k=5):
    # Convert user query to an embedding vector
    query_embedding = encoder([query])
    
    # Search the Pinecone index for the most relevant messages
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract and return the content of the matched messages
    return [match['metadata']['content'] for match in results['matches']]

# Function to generate a response using Groq and Pinecone context
def get_response(query):
    # Retrieve relevant messages from the Pinecone index
    docs = get_relevant_messages(query)
    
    # Create a system message using the retrieved context
    system_message = (
        "You are an amalgamation of several characters, a project know as the 'Council Instrumentality Project'. "
        "You respond as if you ARE a character. This character is the combination of all provided characters thoughts, feelings, and actions."
        "You are not to deviate from playing this character, and should try to mimic the contexts messages as closely as possible while responding to the new message (exluding the date and username, just the message content)."
        "CONTEXT:\n"
        "\n---\n".join(docs)
    )
    
    # Prepare the message payload for the Groq API
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    # Generate a response using the Groq API with the given model and context
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    
    # Return the AI's response
    return chat_response.choices[0].message.content