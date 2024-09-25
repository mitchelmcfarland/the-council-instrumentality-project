import os
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel, AutoTokenizer
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
INDEX_NAME = 'council-instrumentality-project'  # Replace with your index name

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to your index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # This should match the dimension of your embedding model
        metric='cosine',  # Use cosine similarity for vector search
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Update as needed
    )

index = pc.Index(INDEX_NAME)

# Load the embedding model and tokenizer
model_name = 'jinaai/jina-embeddings-v2-base-en'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Function to convert text to vector
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the last hidden state (average pooling)
    vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()
    return vector

# Read the .txt file and prepare data for Pinecone upload
file_path = 'venv/all_messages.txt'  # Replace with the path to your file
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.readlines()

# Prepare data for Pinecone upload
vectors = []
for idx, line in enumerate(text_data):
    vector = text_to_vector(line)
    metadata = {'text': line}
    vectors.append((f'doc-{idx}', vector, metadata))

# Upload data to Pinecone
index.upsert(vectors)

print(f"Successfully indexed {len(vectors)} messages to Pinecone.")
