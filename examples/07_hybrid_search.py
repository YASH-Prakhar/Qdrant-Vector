# Hybrid search is a comibnation of vector search and keyword search.
# It allows you to search for documents that are similar to a given query vector
# and also match specific keywords.
# This is useful when you want to find documents that are relevant to a topic but also contain specific terms.
# we have audio embeddings and we want to find audio files that are similar to a given audio file

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from src.client.connection import get_qdrant_client

client = get_qdrant_client()

# Example vector (e.g., from your embedding model)
query_vector = [0.1, 0.2, ...]  # Replace with actual embedding

# Example metadata filter: session == "session_1"
filter_condition = Filter(
    must=[
        FieldCondition(
            key="session",
            match=MatchValue(value="session_1")
        )
    ]
)

results = client.search(
    collection_name="audio_collection",
    query_vector=query_vector,
    filter=filter_condition,
    limit=5
)