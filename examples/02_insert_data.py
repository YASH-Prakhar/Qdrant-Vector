# Local ZIP Upload to Qdrant - Multi-Session Audio Processing
# Designed for uploading 2 ZIP files in separate sessions to the same collection

# Paste the code in a cell in Google Colab
# Change the runtime to T4 GPU


# =============================================================================
# PART 1: SETUP AND INSTALLATIONS (Run this in every session)
# =============================================================================

# Install required packages
# pip install qdrant-client transformers librosa torch soundfile tqdm

# Check GPU
import torch
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
print(f"🎯 GPU device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU only'}")

# =============================================================================
# PART 2: CONFIGURATION (Set these in every session)
# =============================================================================

# 🚨 IMPORTANT: Set your Qdrant credentials here
QDRANT_CONFIG = {
    "url": "YOUR_QDRANT_CLUSTER_URL",      # e.g., "https://xyz-abc.us-east-0-1.aws.cloud.qdrant.io:6333"
    "api_key": "YOUR_API_KEY",             # Your API key (if required)
    "collection_name": "audio_collection"  # Same collection for both sessions
}

# Session configuration
SESSION_CONFIG = {
    "session_name": "session_1",           # Change this: "session_1", "session_2", etc.
    "zip_description": "youtube-royalty-free"  # Change this: describe your ZIP content
}

print(f"🏷️  Current session: {SESSION_CONFIG['session_name']}")
print(f"📦 Processing: {SESSION_CONFIG['zip_description']}")
print(f"🎯 Target collection: {QDRANT_CONFIG['collection_name']}")

# =============================================================================
# PART 3: AUDIO PROCESSING CLASSES
# =============================================================================

import os
import json
import zipfile
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class AudioEmbeddingGenerator:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """Fast and efficient audio embedding generator for Colab"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model on: {self.device}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model ready: {model_name}")
    
    def process_audio_file(self, file_path, max_duration=60):
        """Process single audio file and return embedding + metadata"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Truncate if too long
            if len(audio) > 16000 * max_duration:
                audio = audio[:16000 * max_duration]
            
            # Generate embedding
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
            # Metadata
            duration = len(audio) / sr
            file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
            
            return {
                "embedding": embedding.tolist(),
                "metadata": {
                    "filename": Path(file_path).name,
                    "session": SESSION_CONFIG['session_name'],
                    "zip_source": SESSION_CONFIG['zip_description'],
                    "duration": round(duration, 2),
                    "file_size_mb": round(file_size, 2),
                    "embedding_dim": len(embedding),
                    "processed_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing {Path(file_path).name}: {e}")
            return None

# =============================================================================
# PART 4: ZIP UPLOAD AND PROCESSING
# =============================================================================

# For Google Colab, uncomment the following line to enable file uploads
# from google.colab import files

def upload_and_extract_zip():
    """Upload ZIP file from local computer and extract"""
    print(f"📤 Upload your ZIP file for {SESSION_CONFIG['zip_description']}")
    print("👆 Click 'Choose Files' and select your ZIP file")
    
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ No file uploaded!")
        return None
    
    # Process the uploaded ZIP
    for filename, data in uploaded.items():
        if not filename.endswith('.zip'):
            print(f"⚠️  {filename} is not a ZIP file, skipping...")
            continue
        
        print(f"📦 Extracting: {filename}")
        
        # Save uploaded file
        with open(filename, 'wb') as f:
            f.write(data)
        
        # Extract ZIP
        extract_folder = f"extracted_{SESSION_CONFIG['session_name']}"
        
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        # Clean up ZIP file
        os.remove(filename)
        
        print(f"✅ Extracted to: {extract_folder}")
        
        # Count audio files
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(Path(extract_folder).rglob(f"*{ext}"))
        
        print(f"🎵 Found {len(audio_files)} audio files")
        
        return Path(extract_folder)
    
    return None

def process_audio_folder(folder_path):
    """Process all audio files in the extracted folder"""
    
    generator = AudioEmbeddingGenerator()
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(folder_path.rglob(f"*{ext}"))
    
    if not audio_files:
        print("❌ No audio files found!")
        return []
    
    print(f"🎵 Processing {len(audio_files)} audio files...")
    
    results = []
    failed = 0
    
    for file_path in tqdm(audio_files, desc="Processing audio"):
        result = generator.process_audio_file(file_path)
        if result:
            results.append(result)
        else:
            failed += 1
    
    print(f"✅ Successfully processed: {len(results)}")
    print(f"❌ Failed: {failed}")
    
    return results

# =============================================================================
# PART 5: QDRANT UPLOAD WITH SESSION SUPPORT
# =============================================================================

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

def setup_qdrant_collection(embedding_dim):
    """Set up Qdrant collection (creates if doesn't exist)"""
    
    client = QdrantClient(
        url=QDRANT_CONFIG["url"],
        api_key=QDRANT_CONFIG["api_key"],
        timeout=60
    )
    
    collection_name = QDRANT_CONFIG["collection_name"]
    
    try:
        # Try to get existing collection
        collection_info = client.get_collection(collection_name)
        print(f"✅ Using existing collection: {collection_name}")
        print(f"📊 Current points in collection: {collection_info.points_count}")
        return client
        
    except:
        # Create new collection
        print(f"🆕 Creating new collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection created with dimension: {embedding_dim}")
        return client

def upload_to_qdrant(embeddings_data):
    """Upload embeddings to Qdrant collection"""
    
    if not embeddings_data:
        print("❌ No embeddings to upload!")
        return
    
    # Get embedding dimension
    embedding_dim = len(embeddings_data[0]["embedding"])
    
    # Setup Qdrant
    client = setup_qdrant_collection(embedding_dim)
    collection_name = QDRANT_CONFIG["collection_name"]
    
    # Get current collection size to generate unique IDs
    try:
        collection_info = client.get_collection(collection_name)
        start_id = collection_info.points_count
        print(f"🔢 Starting ID from: {start_id}")
    except:
        start_id = 0
    
    # Prepare points
    points = []
    for i, item in enumerate(embeddings_data):
        point = PointStruct(
            id=start_id + i,  # Unique ID across sessions
            vector=item["embedding"],
            payload=item["metadata"]
        )
        points.append(point)
    
    # Upload in batches
    batch_size = 50
    total_batches = (len(points) + batch_size - 1) // batch_size
    
    print(f"📤 Uploading {len(points)} embeddings in {total_batches} batches...")
    
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading"):
        batch = points[i:i + batch_size]
        try:
            client.upsert(collection_name=collection_name, points=batch)
        except Exception as e:
            print(f"❌ Error in batch {i//batch_size + 1}: {e}")
            return False
    
    # Verify upload
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ Upload complete!")
        print(f"📊 Total points in collection: {collection_info.points_count}")
        return True
    except Exception as e:
        print(f"⚠️  Could not verify upload: {e}")
        return False

# =============================================================================
# PART 6: MAIN WORKFLOW
# =============================================================================

def run_session():
    """Complete workflow for one session"""
    
    print("🚀 AUDIO PROCESSING SESSION")
    print("=" * 50)
    print(f"Session: {SESSION_CONFIG['session_name']}")
    print(f"Processing: {SESSION_CONFIG['zip_description']}")
    print(f"Target: {QDRANT_CONFIG['collection_name']}")
    
    # Step 1: Upload and extract ZIP
    print(f"\n📤 Step 1: Upload ZIP file")
    folder_path = upload_and_extract_zip()
    
    if not folder_path:
        print("❌ Failed to upload/extract ZIP. Exiting.")
        return
    
    # Step 2: Process audio files
    print(f"\n🎵 Step 2: Process audio files")
    embeddings_data = process_audio_folder(folder_path)
    
    if not embeddings_data:
        print("❌ No embeddings generated. Exiting.")
        return
    
    # Step 3: Save backup
    print(f"\n💾 Step 3: Save backup")
    backup_filename = f"embeddings_{SESSION_CONFIG['session_name']}.json"
    with open(backup_filename, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    file_size = Path(backup_filename).stat().st_size / (1024*1024)
    print(f"✅ Saved {file_size:.1f} MB to {backup_filename}")
    
    # Step 4: Upload to Qdrant
    print(f"\n📤 Step 4: Upload to Qdrant")
    success = upload_to_qdrant(embeddings_data)
    
    if success:
        print(f"\n🎉 SESSION {SESSION_CONFIG['session_name']} COMPLETE!")
        print(f"✅ {len(embeddings_data)} embeddings uploaded")
        print(f"📁 Source: {SESSION_CONFIG['zip_description']}")
        
        # Download backup
        files.download(backup_filename)
        
    else:
        print(f"\n❌ Upload failed. Backup saved for retry.")
        files.download(backup_filename)
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(folder_path)
        print(f"🧹 Cleaned up extracted files")
    except:
        pass
    
    return embeddings_data

# =============================================================================
# PART 7: COLLECTION MANAGEMENT
# =============================================================================

def check_collection_status():
    """Check current status of the Qdrant collection"""
    
    try:
        client = QdrantClient(
            url=QDRANT_CONFIG["url"],
            api_key=QDRANT_CONFIG["api_key"]
        )
        
        collection_name = QDRANT_CONFIG["collection_name"]
        collection_info = client.get_collection(collection_name)
        
        print(f"📊 COLLECTION STATUS: {collection_name}")
        print(f"Total points: {collection_info.points_count}")
        print(f"Vector dimension: {collection_info.config.params.vectors.size}")
        
        # Sample some points to see sessions
        sample_points = client.scroll(collection_name, limit=10)[0]
        sessions = set()
        zip_sources = set()
        
        for point in sample_points:
            if 'session' in point.payload:
                sessions.add(point.payload['session'])
            if 'zip_source' in point.payload:
                zip_sources.add(point.payload['zip_source'])
        
        print(f"Sessions found: {list(sessions)}")
        print(f"ZIP sources: {list(zip_sources)}")
        
    except Exception as e:
        print(f"❌ Error checking collection: {e}")

def search_test():
    """Test similarity search in the collection"""
    
    try:
        client = QdrantClient(
            url=QDRANT_CONFIG["url"],
            api_key=QDRANT_CONFIG["api_key"]
        )
        
        collection_name = QDRANT_CONFIG["collection_name"]
        
        # Get a random vector for testing
        sample = client.scroll(collection_name, limit=1)[0]
        if sample:
            query_vector = sample[0].vector
            
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=5
            )
            
            print("🔍 SIMILARITY SEARCH TEST:")
            for i, result in enumerate(results, 1):
                meta = result.payload
                print(f"{i}. {meta['filename']} from {meta.get('zip_source', 'unknown')} (score: {result.score:.3f})")
        
    except Exception as e:
        print(f"❌ Error testing search: {e}")

# =============================================================================
# PART 8: INSTRUCTIONS AND EXECUTION
# =============================================================================

print("🎯 MULTI-SESSION AUDIO UPLOAD SETUP COMPLETE!")
print("\n📋 INSTRUCTIONS:")
print("1. Set your QDRANT_CONFIG credentials above")
print("2. Set SESSION_CONFIG for current session")
print("3. Run: run_session()")
print("\n🔧 UTILITY FUNCTIONS:")
print("- check_collection_status() - Check collection")
print("- search_test() - Test similarity search")

print(f"\n📌 CURRENT SESSION: {SESSION_CONFIG['session_name']}")
print(f"📦 PROCESSING: {SESSION_CONFIG['zip_description']}")

# Uncomment to run immediately:
run_session()