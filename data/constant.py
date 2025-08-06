from qdrant_client.models import VectorParams, Distance

collections_to_create = [
            {
                "collection_name": "test_collection",
                "vectors_config": VectorParams(size=1536, distance=Distance.COSINE),
                "timeout": 20,
            },
            {
                "collection_name": "test_collection_image",
                "vectors_config": VectorParams(size=1536, distance=Distance.COSINE),
                "timeout": 20,
            }
            # Add more collections here as needed
            # {
            #     "collection_name": "another_collection",
            #     "vectors_config": VectorParams(size=512, distance=Distance.DOT),
            #     "timeout": 20,
            # },
]