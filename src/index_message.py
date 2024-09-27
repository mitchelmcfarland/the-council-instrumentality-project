import os
import uuid
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch

from dotenv import load_dotenv
from semantic_router.encoders import HuggingFaceEncoder
from langchain.docstore.document import Document
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Pinecone using the new method
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENV')  # e.g., 'us-west1-gcp'
index_name = 'discord-conversations'

# Create Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, otherwise create it
if index_name not in pc.list_indexes().names():
    # Create the index with ServerlessSpec
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_environment
        )
    )
    print(f"Index {index_name} created with dimension 768.")
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
encoder._model.to(device)  # Move model to the correct device

# Read and parse dataset.txt
messages = []
with open('dataset.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    parts = line.strip().split(' : ')
    if len(parts) == 3:
        username, content, timestamp_str = parts
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            messages.append({
                'id': str(idx),
                'username': username,
                'content': content,
                'timestamp': timestamp
            })
        except ValueError:
            continue  # Skip lines with invalid timestamp

print(f"Loaded {len(messages)} messages.")

# Generate embeddings for each message using the HuggingFaceEncoder method
def generate_embeddings(messages, encoder, device):
    message_embeddings = []
    for msg in tqdm(messages, desc='Generating embeddings'):
        # Move the text data to the GPU (if available) for embedding
        inputs = encoder._tokenizer([msg['content']], return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            outputs = encoder._model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        message_embeddings.append(embedding)
    return np.array(message_embeddings)

# Create embeddings for all messages
print("Generating embeddings with GPU support..." if device == 'cuda' else "Generating embeddings...")
message_embeddings = generate_embeddings(messages, encoder, device)
print("Embeddings generated successfully.")

# Use Agglomerative Clustering to cluster messages into conversations
# Adjust the distance_threshold as needed
print("Clustering messages into conversations...")
clustering = AgglomerativeClustering(
    n_clusters=None,
    affinity='cosine',
    linkage='average',
    distance_threshold=0.1  # Adjust this threshold based on your data
)

# Compute the cosine distance matrix
distance_matrix = cosine_distances(message_embeddings)

# Fit the clustering model
clustering.fit(distance_matrix)

# Assign messages to clusters
cluster_labels = clustering.labels_

# Group messages by cluster labels
conversations_dict = {}
for idx, label in enumerate(cluster_labels):
    if label not in conversations_dict:
        conversations_dict[label] = []
    conversations_dict[label].append(messages[idx])

print(f"Clustered into {len(conversations_dict)} conversations.")

# Create Document objects from conversations
docs = []
for convo in conversations_dict.values():
    # Sort messages by timestamp within each conversation
    convo = sorted(convo, key=lambda x: x['timestamp'])
    convo_text = '\n'.join([f"{msg['username']}: {msg['content']}" for msg in convo])
    metadata = {'timestamp': convo[0]['timestamp'].isoformat()}
    docs.append(Document(page_content=convo_text, metadata=metadata))

print(f"Created {len(docs)} documents from conversations.")

# Upsert documents into Pinecone
print("Preparing documents for upserting into Pinecone...")
vectors = []
for doc in tqdm(docs, desc='Preparing vectors'):
    # Generate embeddings for the conversation document
    inputs = encoder._tokenizer([doc.page_content], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = encoder._model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    metadata = doc.metadata
    vector_id = str(uuid.uuid4())
    vectors.append((vector_id, embedding, metadata))

# Batch upserts to Pinecone
batch_size = 100
print("Upserting documents into Pinecone...")
for i in tqdm(range(0, len(vectors), batch_size), desc='Upserting batches'):
    batch = vectors[i:i + batch_size]
    ids = [v[0] for v in batch]
    embeddings_batch = [v[1] for v in batch]
    metadata_batch = [v[2] for v in batch]
    # Pinecone upsert expects list of (id, embedding, metadata) tuples
    upsert_batch = list(zip(ids, embeddings_batch, metadata_batch))
    index.upsert(vectors=upsert_batch)

print("Documents upserted into Pinecone successfully.")
