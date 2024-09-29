import os
import torch
from pinecone import Pinecone, ServerlessSpec
from datasets import Dataset
from semantic_router.encoders import HuggingFaceEncoder
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime, timedelta
import math
import logging

# Configure logging
logging.basicConfig(filename='embedding_process.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

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
        dimension=1536,  # Keeping 1536 as per your preference
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
encoder = HuggingFaceEncoder(
    name="dunzhang/stella_en_1.5B_v5"
)

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

# Function to split text by tokens
def split_text_by_tokens(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# Prepare data for embedding with token-based splitting
tokenizer = encoder._tokenizer
processed_data = []
id_counter = 0

for chunk in conversational_chunks:
    chunk_text = '\n'.join([f"{msg['username']}: {msg['content']}" for msg in chunk])
    # Split the chunk text by 512 tokens
    split_texts = split_text_by_tokens(chunk_text, tokenizer, max_length=512, overlap=50)
    conversation_start = chunk[0]['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    for text in split_texts:
        if text.strip():  # Ensure non-empty
            processed_data.append({
                'id': str(id_counter),
                'metadata': {
                    'content': text,
                    'conversation_start': conversation_start
                }
            })
            id_counter += 1

print(f"After token splitting, there are {len(processed_data)} text chunks ready for embedding.")

# Convert processed data to Dataset
dataset = Dataset.from_list(processed_data)

# Updated create_embeddings function with batching and logging
def create_embeddings(dataset, encoder, device, batch_size=1):
    embeddings = []
    all_contents = [data['content'] for data in dataset['metadata']]
    
    # Calculate the total number of batches
    total_batches = math.ceil(len(all_contents) / batch_size)
    
    # Initialize tqdm with correct total
    with tqdm(total=total_batches, desc="Creating embeddings") as pbar:
        for i in range(0, len(all_contents), batch_size):
            batch_contents = all_contents[i:i + batch_size]
            
            try:
                # Tokenize the batch
                inputs = encoder._tokenizer(
                    batch_contents,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                with torch.no_grad():
                    outputs = encoder._model(**inputs)
                    # Compute mean pooling for each sequence in the batch
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                embeddings.extend(batch_embeddings)
                
                # Update the progress bar
                pbar.update(1)
                
                # Optional: Clear CUDA cache to free memory
                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"Error processing batch starting at index {i}: {e}")
                pbar.update(1)  # Skip this batch and continue
    
    return embeddings

# Generate embeddings for the dataset using GPU with batching
batch_size = 1  # Adjust based on your GPU's memory capacity
print(f"Generating embeddings with GPU support using batch size {batch_size}...")

embeddings = create_embeddings(dataset, encoder, device, batch_size=batch_size)
print("Embeddings generated successfully.")

# Check the embedding dimension
dims = len(embeddings[0])
print(f"Embedding Dimension: {dims}")

# Upsert embeddings and metadata to the index
upsert_batch_size = 128
print(f"Indexing {len(dataset)} text chunks in batches of {upsert_batch_size}...")

for i in tqdm(range(0, len(dataset), upsert_batch_size), desc="Indexing batches"):
    i_end = min(len(dataset), i + upsert_batch_size)
    batch = dataset[i:i_end]
    batch_embeddings = embeddings[i:i_end]
    to_upsert = list(zip(batch['id'], batch_embeddings, batch['metadata']))
    index.upsert(vectors=to_upsert)

print("Indexing completed successfully.")
