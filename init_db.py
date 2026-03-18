from db.database import Base, engine
from db import models
import time

print("Attempting to initialize the database (Drop & Recreate)...")

# Retry logic for handling database locks
for attempt in range(5):
    try:
        # Drop all tables first
        Base.metadata.drop_all(bind=engine)
        print("Dropped all existing tables.")
        
        # Create all tables with the new schema
        Base.metadata.create_all(bind=engine)
        print("Created tables with the 11-feature schema.")
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(2)

print("DB initialization complete!")
