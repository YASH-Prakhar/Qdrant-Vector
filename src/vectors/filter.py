# Vector Search functionality index
from src.client.connection import get_client
from qdrant_client.http.models import PayloadSchemaType, TextIndexParams

client = get_client()

client.create_payload_index(
    collection_name="audio_collection",
    field_name="session",
    field_schema=PayloadSchemaType.KEYWORD,
)
print("succesfully created index on field session")

client.create_payload_index(
    collection_name="audio_collection",
    field_name="zip_source",
    field_schema=PayloadSchemaType.KEYWORD,
)
print("succesfully created index on field zip_source")

client.create_payload_index(
    collection_name="audio_collection",
    field_name="duration",
    field_schema=PayloadSchemaType.FLOAT,
)
print("succesfully created index on field duration")

client.create_payload_index(
    collection_name="audio_collection",
    field_name="file_size_mb",
    field_schema=PayloadSchemaType.FLOAT,
)
print("succesfully created index on field file_size_mb")

# Only if you use MatchText on filename:
client.create_payload_index(
    collection_name="audio_collection",
    field_name="filename",
    field_schema=TextIndexParams(type="text", lowercase=True),
)
print("succesfully created index on field filename")