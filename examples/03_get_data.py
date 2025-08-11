import os
import torch
import librosa
from dotenv import load_dotenv
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest

load_dotenv()

# Load wav2vec2 base model (768-dim embeddings)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

def get_audio_embedding(audio_path):
    # Load audio (mono, 16kHz)
    audio, sr = librosa.load(audio_path, sr=16000)
    # Preprocess
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: (batch, time, 768)
    hidden = outputs.last_hidden_state.squeeze(0)
    # Mean-pool over time to get a single 768-dim vector
    emb = hidden.mean(dim=0).cpu().numpy()
    return emb

# Qdrant setup
# For a free tier Qdrant Cloud instance, you must provide both the URL and the API key:
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=20
)
COLLECTION_NAME = "audio_collection"

# Prepare query embedding from audio file
query_embedding = get_audio_embedding(f"data\lofi-sample.wav")  # Path to your query file in the data folder

# Perform search in Qdrant
search_result = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_embedding,
    limit=5  # number of results to return
)

for hit in search_result:
    print(f"Score: {hit.score}, Payload: {hit.payload}")

# To add new audio:
# embedding = get_audio_embedding("new_file.wav")
# client.upsert(collection_name=COLLECTION_NAME, points=[
#     PointStruct(id=..., vector=embedding, payload={"filename": "new_file.wav"})
# ])