# Update Data in Qdrant Vector Database
# Simple example for updating vector data and metadata

import os
import torch
import librosa
from dotenv import load_dotenv
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()

# Load wav2vec2 base model for generating embeddings
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

""" what is happening - 
#   Update metadata only
#   Update vector embedding as well as metadata
#   Update Multiple points based on some condition
#   Update points in batches
"""

def get_audio_embedding(audio_path):
    """Generate embedding from audio file"""
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
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=20
)
COLLECTION_NAME = "audio_collection"

# Example 1: Update payload (metadata) for a specific point
def update_metadata():
    """Update metadata for a point with ID 1"""
    point_id = 1
    
    # Update the payload
    updated_payload = {
        "filename": "updated_audio_part_2.wav",
        "session": "session_2",
        "zip_source": "updated-audio-source",
        "duration": 35,
        "file_size_mb": 0.44,
        "embedding_dim": 768,
        "processed_at": "2025-08-07T15:11:54.008172"
    }
    
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload=updated_payload,
        points=[point_id]
    )
    
    print(f"✅ Updated metadata for point {point_id}")

# Example 2: Update both vector and payload
def update_vector_and_metadata():
    """Update both vector and metadata for a point"""
    point_id = 2
    
    # Generate new embedding from updated audio file
    new_embedding = get_audio_embedding("data/lofi-sample.wav")
    
    # Update both vector and payload
    updated_point = PointStruct(
        id=point_id,
        vector=new_embedding.tolist(),
        payload={
            "filename": "lofi_sample_updated.wav",
            "session": "session_1",
            "zip_source": "lofi-samples",
            "duration": 30.5,
            "file_size_mb": 0.8,
            "embedding_dim": len(new_embedding),
            "processed_at": "2025-08-07T15:11:54.008172"
        }
    )
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[updated_point]
    )
    
    print(f"✅ Updated vector and metadata for point {point_id}")

# # Example 3: Update multiple points by filter
def update_multiple_points():
    """Update all points with session 'session_1'"""
    
    # Create filter for points with session 'session_1'
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="session",
                match=MatchValue(value="session_1")
            )
        ]
    )
    
    # Update payload for matching points
    updated_payload = {
        "zip_source": "updated-english-songs",
        "processed_at": "2025-08-07T15:11:54.008172"
    }
    
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload=updated_payload,
        points=filter_condition
    )
    
    print("✅ Updated all points from session_1")

# # Example 4: Batch update multiple points
def batch_update():
    """Update multiple points in batch"""
    
    # Get existing points to update
    existing_points = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=3
    )[0]
    
    updated_points = []
    for point in existing_points:
        # Keep existing vector, update payload
        updated_point = PointStruct(
            id=point.id,
            vector=point.vector,
            payload={
                **point.payload,  # Keep existing payload
                "zip_source": f"{point.payload.get('zip_source', 'unknown')}-updated",
                "processed_at": "2025-08-07T15:11:54.008172"
            }
        )
        updated_points.append(updated_point)
    
    # Batch update
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=updated_points
    )
    
    print(f"✅ Batch updated {len(updated_points)} points")


# Run examples
if __name__ == "__main__":
    print("🔄 Running update examples...")
    
    # Update metadata only
    update_metadata()
    
    # # Update vector and metadata
    # update_vector_and_metadata()
    
    # # Update multiple points by filter
    # update_multiple_points()
    
    # # Batch update
    # batch_update()
    
    print("🎉 Update examples completed!")
