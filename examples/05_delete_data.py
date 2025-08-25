# Delete Data from Qdrant Vector Database
# Simple example for deleting vector data and collections

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

# Qdrant setup
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=20
)
COLLECTION_NAME = "audio_collection"

# Example 1: Delete a specific point by ID
def delete_point_by_id():
    """Delete a point with specific ID"""
    point_id = 2
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=[point_id]
    )
    
    print(f"✅ Deleted point {point_id}")

# Example 2: Delete multiple points by IDs
def delete_multiple_points():
    """Delete multiple points by their IDs"""
    point_ids = [2, 3, 4]
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=point_ids
    )
    
    print(f"✅ Deleted points: {point_ids}")

# Example 3: Delete points by filter
def delete_points_by_filter():
    """Delete all points from a specific session"""
    
    # Create filter for points from session_1
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="session",
                match=MatchValue(value="session_1")
            )
        ]
    )
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=filter_condition
    )
    
    print("✅ Deleted all points from session_1")

# Example 4: Delete points by payload condition
def delete_by_payload():
    """Delete points with specific zip_source"""
    
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="zip_source",
                match=MatchValue(value="english-song-snippet")
            )
        ]
    )
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=filter_condition
    )
    
    print("✅ Deleted points with zip_source 'english-song-snippet'")

# Example 4b: Delete points by duration range
def delete_by_duration():
    """Delete points with duration less than 10 seconds"""
    
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="duration",
                range={
                    "lt": 10.0  # Less than 10 seconds
                }
            )
        ]
    )
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=filter_condition
    )
    
    print("✅ Deleted points with duration < 10 seconds")

# Example 4c: Delete points by file size
def delete_by_file_size():
    """Delete points with file size less than 0.1 MB"""
    
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="file_size_mb",
                range={
                    "lt": 0.1  # Less than 0.1 MB
                }
            )
        ]
    )
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=filter_condition
    )
    
    print("✅ Deleted points with file size < 0.1 MB")

# Example 5: Delete entire collection
def delete_collection():
    """Delete the entire collection"""
    
    client.delete_collection(collection_name=COLLECTION_NAME)
    
    print(f"✅ Deleted collection: {COLLECTION_NAME}")

# Example 6: Check collection status
def check_collection():
    """Check if collection exists and count points"""
    
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Collection '{COLLECTION_NAME}' exists")
        print(f"Total points: {collection_info.points_count}")
        return True
    except:
        print(f"❌ Collection '{COLLECTION_NAME}' not found")
        return False

# Example 7: List all collections
def list_collections():
    """Show all available collections"""
    
    collections = client.get_collections()
    
    print("📚 Available collections:")
    for collection in collections.collections:
        print(f"- {collection.name}")

# Run examples
if __name__ == "__main__":
    print("🗑️  Running delete examples...")
    
    # Check current status
    if check_collection():
        print(f"\nCurrent collection status:")
        list_collections()
        
        # Delete specific points
        delete_point_by_id()
        # delete_multiple_points()
        
        # Delete by filter
        # delete_points_by_filter()
        # delete_by_payload()
        # delete_by_duration()
        # delete_by_file_size()
        
        # Check status after deletions
        check_collection()
        
        # Optionally delete entire collection (uncomment if needed)
        # delete_collection()
        
    print("\n🎉 Delete examples completed!")
