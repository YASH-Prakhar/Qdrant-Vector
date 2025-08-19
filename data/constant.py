from qdrant_client.models import VectorParams, Distance

collections_to_create = [
            {
                "collection_name": "audio_collection",
                "vectors_config": VectorParams(size=768, distance=Distance.COSINE),
                "timeout": 20,
            },
            {
                "collection_name": "combined_audio_collection",
                "vectors_config": VectorParams(size=768, distance=Distance.COSINE),
                "timeout": 20,
            },
            {
                "collection_name": "text_collection",
                "vectors_config": VectorParams(size=768, distance=Distance.COSINE),
                "timeout":20,
            }
            # Add more collections here as needed
            # {
            #     "collection_name": "another_collection",
            #     "vectors_config": VectorParams(size=512, distance=Distance.DOT),
            #     "timeout": 20,
            # },
]