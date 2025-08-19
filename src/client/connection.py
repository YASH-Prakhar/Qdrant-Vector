import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from config.settings import QDRANT_URL, QDRANT_API_KEY
load_dotenv()

def get_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=20,
    )