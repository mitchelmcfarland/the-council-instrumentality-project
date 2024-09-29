import os
import torch
from pinecone import Pinecone
from semantic_router.encoders import HuggingFaceEncoder
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Pinecone API key and environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = "discord-index"

# Initialize Pinecone using the new Pinecone class
pc = Pinecone(api_key=pinecone_api_key)

# Connect to the existing index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

# Initialize Encoder for text embeddings
# Initialize Encoder and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA {'is' if device == 'cuda' else 'is not'} available. Using {device.upper()}.")

# Initialize HuggingFace Encoder
encoder = HuggingFaceEncoder(name="dunzhang/stella_en_1.5B_v5")

# Move the underlying model and tokenizer to the correct device
encoder._model.to(device)

# Function to query relevant conversation based on user input
def get_relevant_conversations(query, top_k=5):
    # Convert user query to an embedding vector
    query_embedding = encoder([query])
    
    # Search the Pinecone index for the most relevant conversation
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract the content of the matched messages
    conversations = [match['metadata']['content'] for match in results['matches']]
    
    # Print the retrieved conversations for debugging
    print("\n=== Retrieved Conversations ===")
    for i, conversation in enumerate(conversations, 1):
        print(f"Conversation {i}:\n{conversation}\n---")
    
    # Return the list of retrieved conversations
    return conversations

# Function to generate a response using Groq and Pinecone context
def get_response(query, context_string, username):
    # Retrieve relevant messages from the Pinecone index
    docs = get_relevant_conversations(query)
    
    # Create a system message using the retrieved context
    system_message = (
        "You are Uni. Uni has been an observer of the group chat known as the council since the beginning but has never spoken or engaged."
        "The context will be several messages from the council. Uni only has as much knowledge as the context provided, and will never make up information about the context."
        "Uni always responds as an individual that has adopted many of the thoughts, feelings, opinions, and mannerisms of every individual in the groupchat, but still has their own opinons."
        "Uni always emulates the style of the messages in context, and should never deviate from this. Uni only generates text message style responses, like in the context."
        "Uni will always be talking to a council member in the present, where as the context is in the past."

        "CONTEXT:\n"
        "\n---\n".join(docs)
    )
    
    current_conversation = (
        "Current Conversation:\n"
        f"{context_string}"
    )
    
    # Prepare the message payload for the Groq API
    messages = [
        #{"role": "assistant", "content": current_conversation},  # current convo
        {"role": "system", "content": system_message},  # Instructions for behavior
        {"role": "user", "content": username + ": " + query}  # User's query
    ]
    
    # Generate a response using the Groq API with the given model and context
    chat_response = groq_client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=messages
    )
    
    # Return the AI's response
    return chat_response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    query = "is connor racist?"
    context_string = "none"
    username = "Mitchel"
    
    response = get_response(query, context_string, username)
    print("\n=== AI Response ===")
    print(response)
