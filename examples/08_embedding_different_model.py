# Audio embedding with different models such as PaSST and PaNNs from HuggingFace, also OpenL3 & YAMNet
# Compare the embeddings and their performance in Qdrant
from qdrant_client import QdrantClient
from src.client.connection import get_qdrant_client
from src.utils.embeddings import AudioEmbedding
from src.utils.embeddings import OpenL3Embedding
from src.utils.embeddings import YAMNetEmbedding
from src.utils.embeddings import PaNNsEmbedding
from src.utils.embeddings import PaSSTEmbedding

client = get_qdrant_client()
audio_file = "path/to/your/audio/file.wav"  # Replace with your audio file path
embedding_models = {
    # "OpenL3": OpenL3Embedding(),
    "YAMNet": YAMNetEmbedding(),
    "PaNNs": PaNNsEmbedding(),
    "PaSST": PaSSTEmbedding()
}

for model_name, model in embedding_models.items():
    embedding = model.get_embedding(audio_file)
    print(f"{model_name} embedding shape: {embedding.shape}")
    # Here you can add code to store the embedding in Qdrant and perform searches
    # For example:
    collection_name = f"audio_collection_{model_name.lower()}"
    client.recreate_collection(collection_name=collection_name, vector_size=len(embedding), distance="Cosine")
    client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": 1,
                "vector": embedding.tolist(),
                "payload": {"model": model_name}
            }
        ]
    )
    results = client.search(
        collection_name=collection_name,
        query_vector=embedding.tolist(),
        limit=5
    )
    print(f"Search results for {model_name}: {results}")

# Note: Ensure you have the required libraries installed and the audio file path is correct.
