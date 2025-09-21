#!/usr/bin/env python3
"""
supabase_utils.py
Utility functions for uploading/downloading files to/from Supabase Storage.
Uploads both .pkl and .json versions of overlap results.
Overwrites existing files (no upsert).
"""

import os
import json
import pickle
from supabase import create_client, Client

BUCKET_NAME = "monitor-data"
OVERLAP_FILE_NAME = "overlap_results.pkl"
OVERLAP_JSON_NAME = "overlap_results.json"
MAX_SIZE_MB = 1.7
MAX_HISTORY_KEEP = 5  # keep only last 5 entries if trimming is needed


def get_supabase_client() -> Client:
    """Create and return a Supabase client. Raises if credentials are missing."""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("❌ Missing SUPABASE_URL or SUPABASE_KEY in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def prepare_json_from_pkl(pkl_path: str) -> bytes:
    """Load pickle, filter NONE grades, and drop oldest tokens until file <1.7 MB."""
    with open(pkl_path, "rb") as f:
        overlap_results = pickle.load(f)

    # ✅ Filter out NONE grade tokens
    filtered = {
        token_id: history
        for token_id, history in overlap_results.items()
        if history and history[-1]["result"]["grade"] != "NONE"
    }

    # ✅ Sort tokens by their last update (newest first)
    def get_last_time(history):
        return history[-1].get("result", {}).get("discovered_at") or \
               history[-1].get("timestamp") or "1970-01-01T00:00:00"

    sorted_tokens = sorted(
        filtered.items(),
        key=lambda kv: get_last_time(kv[1]),
        reverse=True
    )

    # ✅ Start with all tokens, then drop oldest until size < MAX_SIZE_MB
    pruned = dict(sorted_tokens)
    json_bytes = json.dumps(pruned, indent=2).encode("utf-8")

    while len(json_bytes) / (1024 * 1024) > MAX_SIZE_MB and len(pruned) > 0:
        # Drop the oldest token
        sorted_tokens = sorted_tokens[:-1]
        pruned = dict(sorted_tokens)
        json_bytes = json.dumps(pruned, indent=2).encode("utf-8")

    print(f"✅ Final JSON size: {len(json_bytes)/(1024*1024):.2f} MB, {len(pruned)} tokens kept")
    return json_bytes



def upload_file(file_path: str, bucket: str = BUCKET_NAME):
    """Upload a .pkl file and also generate/upload a .json version (cleaned)."""
    supabase = get_supabase_client()
    file_name = os.path.basename(file_path)

    # ✅ Delete old version of the file
    try:
        supabase.storage.from_(bucket).remove([file_name])
        if file_name == OVERLAP_FILE_NAME:
            supabase.storage.from_(bucket).remove([OVERLAP_JSON_NAME])
    except Exception:
        pass

    # ✅ Upload the raw file
    with open(file_path, "rb") as f:
        file_data = f.read()
        supabase.storage.from_(bucket).upload(file_name, file_data)
    print(f"✅ Uploaded {file_name} ({len(file_data)/1024:.2f} KB) to bucket '{bucket}'")

    # ✅ If it's overlap_results.pkl, also upload JSON
    if file_name == OVERLAP_FILE_NAME:
        try:
            json_bytes = prepare_json_from_pkl(file_path)
            supabase.storage.from_(bucket).upload(
                OVERLAP_JSON_NAME, json_bytes, {"content-type": "application/json"}
            )
            print(f"✅ Uploaded {OVERLAP_JSON_NAME} ({len(json_bytes)/1024:.2f} KB)")
        except Exception as e:
            print(f"⚠️ Could not create/upload {OVERLAP_JSON_NAME}: {e}")


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
    """Upload overlap_results.pkl (and its JSON)."""
    upload_file(file_path, bucket)


def download_overlap_results(save_path: str, bucket: str = BUCKET_NAME):
    """Download overlap_results.pkl specifically."""
    download_file(save_path, OVERLAP_FILE_NAME, bucket)


if __name__ == "__main__":
    # Example usage:
    test_pkl_path = r"C:\Users\HP USER\Documents\Data Analyst\degen smart\data\overlap_results.pkl"  
    upload_overlap_results(test_pkl_path)

    download_path = "downloaded_overlap_results.pkl"
    download_overlap_results(download_path)