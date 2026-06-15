import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os

# Initialize Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate("service_account.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def migrate_videos():
    print("Starting migration from 'videos' to 'transcriptions'...")
    
    # Get all documents from videos collection
    videos_ref = db.collection("videos")
    docs = videos_ref.stream()
    
    transcriptions_ref = db.collection("transcriptions")
    
    batch = db.batch()
    count = 0
    total_migrated = 0
    
    for doc in docs:
        data = doc.to_dict()
        doc_id = doc.id
        
        # Add mediaType if not present
        if "mediaType" not in data:
            data["mediaType"] = "video"
            
        # Create new document reference with same ID
        new_doc_ref = transcriptions_ref.document(doc_id)
        
        # Add set operation to batch
        batch.set(new_doc_ref, data)
        count += 1
        
        # Commit batch every 400 operations (limit is 500)
        if count >= 400:
            batch.commit()
            total_migrated += count
            print(f"Migrated {total_migrated} documents...")
            batch = db.batch()
            count = 0
            
    # Commit remaining operations
    if count > 0:
        batch.commit()
        total_migrated += count
        
    print(f"Migration complete! Successfully migrated {total_migrated} documents.")

if __name__ == "__main__":
    migrate_videos()
