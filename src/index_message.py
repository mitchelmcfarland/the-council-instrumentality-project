import os
import torch
import uuid
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from datasets import Dataset
from semantic_router.encoders import HuggingFaceEncoder
from tqdm.auto import tqdm
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = "discord-index"

# Create Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, otherwise create it with dotproduct metric for hybrid search
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Ensure this matches your dense embedding dimension
        metric='dotproduct',  # Required for hybrid search
        spec=ServerlessSpec(
            cloud='aws',  # Adjust based on your environment
            region=pinecone_env
        )
    )
    print(f"Index '{index_name}' created with 'dotproduct' metric.")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# Initialize Encoder and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA {'is' if device == 'cuda' else 'is not'} available. Using {device.upper()}.")

# Initialize HuggingFace Encoder for dense vectors
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")
encoder._model.to(device)  # Move model to the correct device

# Initialize BM25Encoder for sparse vectors
bm25 = BM25Encoder()
print("Preparing to fit BM25 encoder...")

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
print(f"Loaded {len(messages_data)} messages from '{file_path}'.")

# Function to split messages into conversational chunks based on an 8-hour gap
def split_into_conversational_chunks(messages, gap_threshold=timedelta(hours=8)):
    if not messages:
        return []
    # Sort messages by timestamp
    messages.sort(key=lambda x: x['timestamp'])
    chunks = []
    current_chunk = []
    last_timestamp = None

    for msg in messages:
        if last_timestamp and (msg['timestamp'] - last_timestamp) > gap_threshold:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(msg)
        last_timestamp = msg['timestamp']

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Split messages into conversational chunks using an 8-hour gap
conversational_chunks = split_into_conversational_chunks(messages_data)
print(f"Split messages into {len(conversational_chunks)} conversational chunks.")

# Initialize text splitter with optimized chunk size and overlap
# Based on model's token limit and assuming ~4 characters per token
# For 2064 tokens per conversation: 2064 * 4 = 8256 characters
# To stay within, use chunk_size=7500 and chunk_overlap=1000
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=7500,  # Adjusted based on token limits
    chunk_overlap=1000  # Ensures context preservation
)

# Prepare data for embedding with metadata including the day of interaction
processed_data = []
split_texts_list = []  # To fit BM25Encoder on split texts
id_counter = 0

for chunk in conversational_chunks:
    # Combine messages in the chunk into a single text
    chunk_text = '\n'.join([f"{msg['username']}: {msg['content']}" for msg in chunk])
    # Split the chunk_text using RecursiveCharacterTextSplitter if it is too long
    split_texts = text_splitter.split_text(chunk_text)
    split_texts_list.extend(split_texts)  # Collect all split texts for BM25 fitting
    # Add metadata for the day of interaction
    day_of_interaction = chunk[0]['timestamp'].strftime("%Y-%m-%d")
    for text in split_texts:
        processed_data.append({
            'id': str(uuid.uuid4()),  # Ensure unique IDs
            'metadata': {
                'content': text,
                'day_of_interaction': day_of_interaction  # Add day metadata
            }
        })
        id_counter += 1

print(f"After splitting, there are {len(processed_data)} text chunks ready for embedding.")

# Convert processed data to Dataset
dataset = Dataset.from_list(processed_data)

# Fit BM25Encoder on the split texts
print("Fitting BM25 encoder on split texts...")
bm25.fit(split_texts_list)
print("BM25 encoder fitted successfully.")

# Function to create dense and sparse embeddings for each text chunk
def create_embeddings(dataset, encoder, bm25_encoder, device):
    dense_embeddings = []
    sparse_embeddings = []
    for data in tqdm(dataset['metadata'], desc="Creating embeddings"):
        content = data['content']
        # Dense embedding
        inputs = encoder._tokenizer([content], return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = encoder._model(**inputs)
            dense_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        dense_embeddings.append(dense_embedding)
        
        # Sparse embedding using BM25
        sparse_embedding = bm25_encoder.encode_documents(content)
        # Convert sparse_embedding to Pinecone's expected format
        sparse_values = {
            'indices': sparse_embedding['indices'],
            'values': sparse_embedding['values']
        }
        sparse_embeddings.append(sparse_values)
        
    return dense_embeddings, sparse_embeddings

# Generate embeddings for the dataset
print("Generating dense and sparse embeddings...")
dense_embeddings, sparse_embeddings = create_embeddings(dataset, encoder, bm25, device)
print("Embeddings generated successfully.")

# Check the embedding dimension
dims = len(dense_embeddings[0])
print(f"Embedding Dimension: {dims}")

# Upsert embeddings and metadata to the index with sparse-dense vectors
batch_size = 128
total_chunks = len(dataset)
num_batches = (total_chunks + batch_size - 1) // batch_size
print(f"Indexing {total_chunks} text chunks in batches of {batch_size}...")

for i in tqdm(range(0, total_chunks, batch_size), desc="Indexing batches"):
    i_end = min(total_chunks, i + batch_size)
    batch = dataset[i:i_end]  # Retrieves a Batch object (dict of lists)
    batch_ids = batch['id']  # List of IDs
    batch_dense = dense_embeddings[i:i_end]
    batch_sparse = sparse_embeddings[i:i_end]
    batch_metadata = batch['metadata']  # List of metadata dicts
    
    # Prepare vectors with both dense and sparse values, including metadata
    vectors = []
    for idx in range(len(batch_ids)):
        vector = {
            'id': batch_ids[idx],
            'values': batch_dense[idx].tolist(),  # Convert numpy array to list
            'sparse_values': batch_sparse[idx],
            'metadata': batch_metadata[idx]      # Include metadata
        }
        vectors.append(vector)
    
    # Upsert the batch to Pinecone
    try:
        index.upsert(vectors=vectors)
    except Exception as e:
        print(f"Error upserting batch {i // batch_size + 1}: {e}")

print("Indexing completed successfully.")
