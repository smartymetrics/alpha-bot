
#!/usr/bin/env python3
"""
supabase_utils.py
Utility functions for uploading/downloading files to/from Supabase Storage.
Includes overwrite support and safe lazy-loading of Supabase credentials.
"""

import os
from supabase import create_client, Client

BUCKET_NAME = "bot-data"
OVERLAP_FILE_NAME = "overlap_results.pkl"


def get_supabase_client() -> Client:
    """Create and return a Supabase client. Raises if credentials are missing."""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "❌ Missing SUPABASE_URL or SUPABASE_KEY in environment variables"
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_file(file_path: str, bucket: str = BUCKET_NAME):
    """Upload a file to Supabase, overwriting if it already exists."""
    supabase = get_supabase_client()
    file_name = os.path.basename(file_path)

    # Delete old file if it exists
    try:
        supabase.storage.from_(bucket).remove([file_name])
    except Exception:
        pass

    with open(file_path, "rb") as f:
        file_data = f.read()

    supabase.storage.from_(bucket).upload(file_name, file_data)
    print(f"✅ Uploaded {file_name} to Supabase bucket '{bucket}'")


def download_file(save_path: str, file_name: str, bucket: str = BUCKET_NAME):
    """Download a file from Supabase Storage to a given local path."""
    try:
        supabase = get_supabase_client()
        res = supabase.storage.from_(bucket).download(file_name)
        with open(save_path, "wb") as f:
            f.write(res)
        print(f"✅ Downloaded '{file_name}' from Supabase to '{save_path}'")
    except Exception as e:
        print(f"⚠️ Could not download '{file_name}': {e}")


def upload_overlap_results(file_path: str, bucket: str = BUCKET_NAME):
    """Upload overlap_results.pkl specifically."""
    upload_file(file_path, bucket)


def download_overlap_results(save_path: str, bucket: str = BUCKET_NAME):
    """Download overlap_results.pkl specifically."""
    download_file(save_path, OVERLAP_FILE_NAME, bucket)
