from src.client.connection import get_client
from data.constant import collections_to_create

def main():
    
    client = get_client()
    # print(client)
    
    # Test Connection
    try:
        collections = client.get_collections()
        print(f"Connection Successful: {len(collections.collections)} collections found")
        
    except Exception as e:
        print(f"Connection Failed: {e}")
    
    # Create a collection if it doesn't exist
    try:

        existing_collections = {c.name for c in client.get_collections().collections}

        for col in collections_to_create:
            if col["collection_name"] not in existing_collections:
                client.create_collection(**col)
                print(f"Collection '{col['collection_name']}' created successfully")
            else:
                print(f"Collection '{col['collection_name']}' already exists, not creating.")

        # print(f"Collection created successfully")

    except Exception as e:
        print(f"Collection creation failed: {e}")

if __name__ == "__main__":
    main()