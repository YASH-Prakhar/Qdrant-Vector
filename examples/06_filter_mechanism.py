# Qdrant Filter Mechanisms
# Comprehensive examples of all filtering capabilities

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, MatchText, MatchAny, MatchExcept, Range
)

load_dotenv()

# Qdrant setup
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=20
)
COLLECTION_NAME = "audio_collection"

# =============================================================================
# BASIC FILTERS
# =============================================================================

# Exact match filtering
def exact_match_filter():
    """Filter by exact value match"""
    print("🔍 Exact Match Filter:")
    
    # Find points with exact session name
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="session",
                match=MatchValue(value="session_1")
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=100
    )[0]
    
    print(f"Found {len(results)} points with session='session_1'")
    for point in results:
        print(f"  ID: {point.id}, Filename: {point.payload.get('filename')}")

# Text Search filter
def text_search_filter():
    """Filter by text search (partial match)"""
    print("\n🔍 Text Search Filter:")
    
    # Find points with filename containing "mp3"
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="filename",
                match=MatchText(text="mp3")
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=100
    )[0]
    
    print(f"Found {len(results)} points with filename containing 'mp3'")
    for point in results:
        print(f"  ID: {point.id}, Filename: {point.payload.get('filename')}")

# =============================================================================
# NUMERICAL RANGE FILTERS
# =============================================================================

def range_filter():
    """Filter by numerical ranges"""
    print("\n🔍 Range Filter:")
    
    # Find points with duration between 20-40 seconds
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="duration",
                range=Range(
                    gte=28.0,  # Greater than or equal to
                    lte=30.0    # Less than or equal to
                )
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points with duration 20-40 seconds")
    for point in results:
        print(f"  ID: {point.id}, Duration: {point.payload.get('duration')}s")

def file_size_filter():
    """Filter by file size ranges"""
    print("\n🔍 File Size Filter:")
    
    # Find points with file size less than 1 MB
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="file_size_mb",
                range=Range(lt=1.0)  # Less than 1.0 MB
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points with file size < 1 MB")
    for point in results:
        print(f"  ID: {point.id}, Size: {point.payload.get('file_size_mb')} MB")

# =============================================================================
# COMPLEX LOGICAL FILTERS
# =============================================================================

def logical_and_filter():
    """Filter with AND logic (must)"""
    print("\n🔍 AND Logic Filter:")
    
    # Find points that MUST have both conditions
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="session",
                match=MatchValue(value="session_1")
            ),
            FieldCondition(
                key="duration",
                range=Range(gte=25.0)
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points from session_1 AND duration >= 25s")
    for point in results:
        print(f"  ID: {point.id}, Session: {point.payload.get('session')}, Duration: {point.payload.get('duration')}s")

def logical_or_filter():
    """Filter with OR logic (should)"""
    print("\n🔍 OR Logic Filter:")
    
    # Find points that SHOULD have either condition
    filter_condition = Filter(
        should=[
            FieldCondition(
                key="zip_source",
                match=MatchValue(value="english-song-snippet")
            ),
            FieldCondition(
                key="zip_source",
                match=MatchValue(value="guitar-samples")
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points from english-song-snippet OR guitar-samples")
    for point in results:
        print(f"  ID: {point.id}, Source: {point.payload.get('zip_source')}")

def logical_not_filter():
    """Filter with NOT logic (must_not)"""
    print("\n🔍 NOT Logic Filter:")
    
    # Find points that MUST NOT have a condition
    filter_condition = Filter(
        must_not=[
            FieldCondition(
                key="zip_source",
                match=MatchValue(value="english-song-snippet")
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points NOT from english-song-snippet")
    for point in results:
        print(f"  ID: {point.id}, Source: {point.payload.get('zip_source')}")

# =============================================================================
# ADVANCED FILTERS
# =============================================================================

def multiple_conditions_filter():
    """Complex filter with multiple conditions"""
    print("\n🔍 Complex Multiple Conditions Filter:")
    
    # Find points with complex criteria
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="session",
                match=MatchValue(value="session_1")
            ),
            FieldCondition(
                key="filename",
                match=MatchText(text="wav")
            )
        ],
        should=[
            FieldCondition(
                key="duration",
                range=Range(gte=20.0, lte=60.0)
            )
        ],
        must_not=[
            FieldCondition(
                key="file_size_mb",
                range=Range(gt=2.0)  # Greater than 2 MB
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points matching complex criteria")
    for point in results:
        print(f"  ID: {point.id}, Session: {point.payload.get('session')}, Duration: {point.payload.get('duration')}s, Size: {point.payload.get('file_size_mb')} MB")

def any_value_filter():
    """Filter by multiple possible values"""
    print("\n🔍 Any Value Filter:")
    
    # Find points with any of these zip sources
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="zip_source",
                match=MatchAny(any=["english-song-snippet", "guitar-samples", "lofi-samples"])
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points from any of the specified sources")
    for point in results:
        print(f"  ID: {point.id}, Source: {point.payload.get('zip_source')}")

def except_value_filter():
    """Filter excluding specific values"""
    print("\n🔍 Except Value Filter:")
    
    # Find points excluding specific zip sources
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="zip_source",
                match=MatchExcept(except_=["english-song-snippet"])
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=5
    )[0]
    
    print(f"Found {len(results)} points excluding english-song-snippet")
    for point in results:
        print(f"  ID: {point.id}, Source: {point.payload.get('zip_source')}")

# =============================================================================
# SEARCH WITH FILTERS
# =============================================================================

def search_with_filter():
    """Perform similarity search with filters"""
    print("\n🔍 Search with Filter:")
    
    # Get a sample vector for search
    sample_points = client.scroll(collection_name=COLLECTION_NAME, limit=1)[0]
    if sample_points:
        query_vector = sample_points[0].vector
        
        # Search with filter: only from session_1
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="session",
                    match=MatchValue(value="session_1")
                )
            ]
        )
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=filter_condition,
            limit=5
        )
        
        print(f"Found {len(results)} similar points from session_1")
        for result in results:
            print(f"  ID: {result.id}, Score: {result.score:.3f}, Filename: {result.payload.get('filename')}")

# =============================================================================
# FILTER UTILITIES
# =============================================================================

def count_filtered_points():
    """Count points matching filter criteria"""
    print("\n🔍 Count Filtered Points:")
    
    # Count points from session_1 with duration > 30s
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="session",
                match=MatchValue(value="session_1")
            ),
            FieldCondition(
                key="duration",
                range=Range(gt=30.0)
            )
        ]
    )
    
    # Get total count
    total_count = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=filter_condition
    )
    
    print(f"Total points from session_1 with duration > 30s: {total_count.count}")

def filter_statistics():
    """Get statistics for filtered data"""
    print("\n🔍 Filter Statistics:")
    
    # Get points from session_1
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="session",
                match=MatchValue(value="session_1")
            )
        ]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=100  # Get more for statistics
    )[0]
    
    if results:
        durations = [point.payload.get('duration', 0) for point in results]
        file_sizes = [point.payload.get('file_size_mb', 0) for point in results]
        
        print(f"Session 1 Statistics:")
        print(f"  Total points: {len(results)}")
        print(f"  Average duration: {sum(durations)/len(durations):.2f}s")
        print(f"  Average file size: {sum(file_sizes)/len(file_sizes):.2f} MB")
        print(f"  Min duration: {min(durations):.2f}s")
        print(f"  Max duration: {max(durations):.2f}s")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all filter examples"""
    print("🚀 QDRANT FILTER MECHANISMS")
    print("=" * 50)
    
    # Check if collection exists
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' found with {collection_info.points_count} points")
    except Exception as e:
        print(f"❌ Collection '{COLLECTION_NAME}' not found or error: {e}")
        print("Please ensure the collection exists and contains data before running filters")
        return
    
    # Run filter examples
    # exact_match_filter()
    text_search_filter()
    # range_filter()
    # file_size_filter()
    # logical_and_filter()
    # logical_or_filter()
    # logical_not_filter()
    # multiple_conditions_filter()
    # any_value_filter()
    # except_value_filter()
    # search_with_filter()
    # count_filtered_points()
    # filter_statistics()
    
    print("\n🎉 Filter examples completed!")

if __name__ == "__main__":
    main()
