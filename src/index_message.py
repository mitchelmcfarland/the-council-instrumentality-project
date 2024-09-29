import os
import torch
import uuid
from pinecone import Pinecone, ServerlessSpec
from datasets import Dataset
from semantic_router.encoders import HuggingFaceEncoder
from tqdm.auto import tqdm
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone using the new method
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = "test-index"

# Create Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, otherwise create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=4096,  # Make sure this matches the embedding dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws', 
            region=pinecone_env
        )
    )
    print(f"Index {index_name} created.")
else:
    print(f"Index {index_name} already exists.")

# Connect to the index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# Initialize Encoder and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA {'is' if device == 'cuda' else 'is not'} available. Using {device.upper()}.")

# Initialize HuggingFace Encoder
encoder = HuggingFaceEncoder(name="nvidia/NV-Embed-v2", trust_remote_code=True)

# Move the underlying model to the correct device
encoder._model.to(device)  # Using _model instead of model

# File path for the messages file
file_path = 'dataset.txt'

# Function to load messages from the text file
def load_messages(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if line:
                parts = line.split(' : ')
                if len(parts) == 3:
                    username, content, timestamp_str = parts
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except ValueError:
                        # Skip lines with invalid timestamp
                        continue
                    data.append({
                        'id': str(idx),
                        'username': username,
                        'content': content,
                        'timestamp': timestamp
                    })
    return data

# Load and format the dataset
messages_data = load_messages(file_path)
print(f"Loaded {len(messages_data)} messages from {file_path}.")

# Function to split messages into conversational chunks based on a 3-hour inactivity gap
def split_into_conversational_chunks(messages, gap_threshold=timedelta(hours=3)):
    if not messages:
        return []
    # Sort messages by timestamp
    messages.sort(key=lambda x: x['timestamp'])
    conversational_chunks = []
    current_chunk = [messages[0]]
    for msg in messages[1:]:
        time_diff = msg['timestamp'] - current_chunk[-1]['timestamp']
        if time_diff > gap_threshold:
            conversational_chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(msg)
    if current_chunk:
        conversational_chunks.append(current_chunk)
    return conversational_chunks

# Split messages into conversational chunks
conversational_chunks = split_into_conversational_chunks(messages_data)
print(f"Split messages into {len(conversational_chunks)} conversational chunks based on a 3-hour gap.")

# Initialize text splitter with the updated chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=1000
)

# Prepare data for embedding with metadata including the conversation start time
processed_data = []
id_counter = 0

for chunk in conversational_chunks:
    # Combine messages in the chunk into a single text
    chunk_text = '\n'.join([f"{msg['username']}: {msg['content']}" for msg in chunk])
    # Split the chunk_text using RecursiveCharacterTextSplitter if necessary
    split_texts = text_splitter.split_text(chunk_text)
    # Add metadata for the conversation start time
    conversation_start = chunk[0]['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    for text in split_texts:
        processed_data.append({
            'id': str(id_counter),
            'metadata': {
                'content': text,
                'conversation_start': conversation_start  # Add conversation start metadata
            }
        })
        id_counter += 1

print(f"After splitting, there are {len(processed_data)} text chunks ready for embedding.")

# Convert processed data to Dataset
dataset = Dataset.from_list(processed_data)

# Function to create embeddings for each text chunk
def create_embeddings(dataset, encoder, device):
    embeddings = []
    for data in tqdm(dataset['metadata'], desc="Creating embeddings"):
        # Move the text data to the GPU (if available) for embedding
        inputs = encoder._tokenizer([data['content']], return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = encoder._model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        embeddings.append(embedding)
    return embeddings

# Generate embeddings for the dataset using GPU
print("Generating embeddings with GPU support..." if device == 'cuda' else "Generating embeddings...")
embeddings = create_embeddings(dataset, encoder, device)
print("Embeddings generated successfully.")

# Check the embedding dimension
dims = len(embeddings[0])
print(f"Embedding Dimension: {dims}")

# Upsert embeddings and metadata to the index
batch_size = 128
print(f"Indexing {len(dataset)} text chunks in batches of {batch_size}...")
for i in tqdm(range(0, len(dataset), batch_size), desc="Indexing batches"):
    i_end = min(len(dataset), i + batch_size)
    batch = dataset[i:i_end]
    batch_embeddings = embeddings[i:i_end]
    to_upsert = []
    for j in range(len(batch)):
        vector_id = batch['id'][j]
        vector_embedding = batch_embeddings[j].tolist()  # Convert numpy array to list
        vector_metadata = batch['metadata'][j]
        to_upsert.append((vector_id, vector_embedding, vector_metadata))
    index.upsert(vectors=to_upsert)

print("Indexing completed successfully.")
