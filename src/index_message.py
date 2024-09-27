import os
import torch
from pinecone import Pinecone, ServerlessSpec
from datasets import Dataset
from semantic_router.encoders import HuggingFaceEncoder
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone using the new method
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = "message-index"

# Create Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, otherwise create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=768,  # Make sure this matches the embedding dimension
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
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# Move the underlying model and tokenizer to the correct device
encoder._model.to(device)  # Using _model instead of model
# No need to move the tokenizer to device as it doesn't perform tensor computations
# encoder._tokenizer is just used for text processing

# File path for the messages file
file_path = 'timestamped.txt'

# Function to load messages from the text file
def load_messages(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if line:
                data.append({
                    'id': str(idx),
                    'metadata': {'content': line}
                })
    return data

# Load and format the dataset
messages_data = load_messages(file_path)
dataset = Dataset.from_list(messages_data)
print(f"Loaded {len(messages_data)} messages from {file_path}.")

# Function to create embeddings for each message
def create_embeddings(dataset, encoder, device):
    embeddings = []
    for data in tqdm(dataset['metadata'], desc="Creating embeddings"):
        # Move the text data to the GPU (if available) for embedding
        inputs = encoder._tokenizer([data['content']], return_tensors='pt', padding=True).to(device)
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
print(f"Indexing {len(dataset)} messages in batches of {batch_size}...")
for i in tqdm(range(0, len(dataset), batch_size), desc="Indexing batches"):
    i_end = min(len(dataset), i + batch_size)
    batch = dataset[i:i_end]
    batch_embeddings = embeddings[i:i_end]
    to_upsert = list(zip(batch['id'], batch_embeddings, batch['metadata']))
    index.upsert(vectors=to_upsert)
    print(f"Batch {i//batch_size + 1} indexed.")

print("Indexing completed successfully.")
