#!/usr/bin/env python3
"""
supabase_utils.py
Utility functions for uploading/downloading files to/from Supabase Storage.
Uploads both .pkl and .json versions of overlap results.
Adds Dexscreener price enrichment (priceUsd) for each token.
Overwrites existing files (no upsert).
Supports dune cache file uploads under dune_cache/.
"""

import os
import json
import pickle
import requests
from supabase import create_client, Client
import tempfile

BUCKET_NAME = "monitor-data"
OVERLAP_FILE_NAME = "overlap_results.pkl"
OVERLAP_JSON_NAME = "overlap_results.json"
MAX_SIZE_MB = 1.7


# -------------------
# Supabase Client
# -------------------
def get_supabase_client() -> Client:
    """Create and return a Supabase client. Uses env vars with local fallback."""
    # SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ldraroaloinsesjoayxc.supabase.co")
    # SUPABASE_KEY = os.getenv("SUPABASE_KEY", "SUPABASE_KEY")

    SUPABASE_URL="https://ldraroaloinsesjoayxc.supabase.co"
    SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxkcmFyb2Fsb2luc2Vzam9heXhjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYzMzEyNDYsImV4cCI6MjA3MTkwNzI0Nn0._F1o7W9ttSGCEh70dEM6l2dtpG5lieo1nQ7Q9zA2VUs"

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("❌ Missing SUPABASE_URL or SUPABASE_KEY in environment variables")

    return create_client(SUPABASE_URL, SUPABASE_KEY)


# -------------------
# Dexscreener Helper
# -------------------
def fetch_dexscreener_price(token_id: str, debug: bool = True) -> float | None:
    """Fetch current USD price for a token from Dexscreener (pairs[0].priceUsd)."""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_id}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        price = data.get("pairs", [{}])[0].get("priceUsd")
        if debug:
            print(f"💰 Dexscreener price for {token_id}: {price}")
        return float(price) if price else None
    except Exception as e:
        if debug:
            print(f"⚠️ Dexscreener fetch failed for {token_id}: {e}")
        return None


# -------------------
# Safe extractors
# -------------------
def safe_get_grade(history_entry):
    """Safely extract grade from a history entry with multiple fallback paths."""
    if not isinstance(history_entry, dict):
        return "UNKNOWN"

    if isinstance(history_entry.get("result"), dict):
        grade = history_entry["result"].get("grade")
        if isinstance(grade, str):
            return grade

    if isinstance(history_entry.get("grade"), str):
        return history_entry["grade"]

    for path in [["overlap_result", "grade"], ["data", "grade"], ["analysis", "grade"]]:
        obj = history_entry
        for key in path:
            obj = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(obj, str):
            return obj

    return "UNKNOWN"


def safe_get_timestamp(history_entry):
    """Safely extract timestamp from a history entry."""
    if not isinstance(history_entry, dict):
        return "1970-01-01T00:00:00"

    for field in ["ts", "timestamp", "checked_at", "created_at", "updated_at"]:
        ts = history_entry.get(field)
        if isinstance(ts, str):
            return ts

    result = history_entry.get("result", {})
    if isinstance(result, dict):
        for field in ["discovered_at", "checked_at", "timestamp"]:
            ts = result.get(field)
            if isinstance(ts, str):
                return ts

    return "1970-01-01T00:00:00"


# -------------------
# JSON Preparation
# -------------------
def prepare_json_from_pkl(pkl_path: str, debug: bool = True) -> bytes:
    """Load pickle, enrich with Dexscreener prices, filter NONE grades, sort, size limit."""
    if not os.path.exists(pkl_path):
        if debug:
            print(f"❌ Missing file: {pkl_path}")
        return b"{}"

    try:
        with open(pkl_path, "rb") as f:
            overlap_results = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load pickle: {e}")
        return b"{}"

    if not isinstance(overlap_results, dict) or not overlap_results:
        if debug:
            print("⚠️ Pickle contained no valid dict data")
        return b"{}"

    filtered = {}
    for token_id, history in overlap_results.items():
        if not isinstance(history, list) or not history:
            continue
        grade = safe_get_grade(history[-1])
        if grade != "NONE":
            latest = history[-1]
            # Ensure dexscreener section exists
            if "dexscreener" not in latest or not isinstance(latest["dexscreener"], dict):
                latest["dexscreener"] = {}
            # Add current price if missing
            if "current_price_usd" not in latest["dexscreener"]:
                price = fetch_dexscreener_price(token_id, debug=debug)
                latest["dexscreener"]["current_price_usd"] = price
            filtered[token_id] = history

    if not filtered:
        if debug:
            print("🚫 All entries NONE, JSON empty")
        return b"{}"

    # Sort by last timestamp
    sorted_tokens = sorted(
        filtered.items(),
        key=lambda kv: safe_get_timestamp(kv[1][-1]),
        reverse=True,
    )
    pruned = dict(sorted_tokens)

    try:
        json_bytes = json.dumps(pruned, indent=2, default=str).encode()
    except Exception:
        json_bytes = json.dumps(pruned, default=str).encode()

    # Trim size if too big
    while len(json_bytes) / (1024 * 1024) > MAX_SIZE_MB and pruned:
        sorted_tokens = sorted_tokens[:-1]
        pruned = dict(sorted_tokens)
        json_bytes = json.dumps(pruned, default=str).encode()

    if debug:
        print(f"✅ JSON ready: {len(pruned)} tokens, {len(json_bytes)/1024:.2f} KB")

    # Save enriched data back to PKL
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(pruned, f)
        if debug:
            print(f"💾 Updated PKL with Dexscreener prices: {pkl_path}")
    except Exception as e:
        if debug:
            print(f"⚠️ Failed to update PKL with prices: {e}")

    return json_bytes


# -------------------
# Upload Functions
# -------------------
def upload_file(file_path: str, bucket: str = BUCKET_NAME, remote_path: str = None, debug: bool = True) -> bool:
    """Upload a raw file to Supabase Storage."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        if debug:
            print(f"❌ File missing or empty: {file_path}")
        return False

    supabase = get_supabase_client()
    file_name = remote_path or os.path.basename(file_path)

    try:
        supabase.storage.from_(bucket).remove([file_name])
    except Exception:
        pass

    try:
        with open(file_path, "rb") as f:
            data = f.read()
        supabase.storage.from_(bucket).upload(file_name, data)
        if debug:
            print(f"✅ Uploaded {file_name} ({len(data)/1024:.2f} KB)")
        return True
    except Exception as e:
        if debug:
            print(f"❌ Upload failed for {file_name}: {e}")
        return False

def upload_overlap_results(file_path: str, bucket: str = BUCKET_NAME, debug: bool = True) -> bool:
    """Upload overlap_results.pkl + JSON, with Dexscreener enrichment."""
    if not os.path.exists(file_path):
        if debug:
            print(f"❌ Missing {file_path}")
        return False

    # Generate enriched JSON and update PKL
    json_bytes = prepare_json_from_pkl(file_path, debug=debug)
    try:
        json_obj = json.loads(json_bytes)
    except Exception:
        json_obj = {}

    # if not json_obj:
    #     if debug:
    #         print("🚫 Filtered JSON empty, removing remote files")
    #         get_supabase_client().storage.from_(bucket).remove([OVERLAP_FILE_NAME, OVERLAP_JSON_NAME])
    #     return False

    # Upload PKL
    if not upload_file(file_path, bucket, OVERLAP_FILE_NAME, debug=debug):
        return False

    # Upload JSON
    try:
        supabase = get_supabase_client()
        supabase.storage.from_(bucket).remove([OVERLAP_JSON_NAME])  # 🔥 delete old JSON first
        supabase.storage.from_(bucket).upload(
            OVERLAP_JSON_NAME, json_bytes, {"content-type": "application/json"}
        )
        if debug:
            print(f"✅ Uploaded {OVERLAP_JSON_NAME} ({len(json_bytes)/1024:.2f} KB)")
    except Exception as e:
        if debug:
            print(f"❌ JSON upload failed: {e}")
        return False

    return True


# -------------------
# Download Functions
# -------------------
def download_file(save_path: str, file_name: str, bucket: str = BUCKET_NAME) -> bool:
    """Download any file from Supabase Storage."""
    try:
        supabase = get_supabase_client()
        data = supabase.storage.from_(bucket).download(file_name)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(data)
        print(f"✅ Downloaded {file_name} → {save_path}")
        return True
    except Exception as e:
        msg = str(e).lower()
        if "404" not in msg and "not found" not in msg:
            print(f"⚠️ Download failed for {file_name}: {e}")
        return False


def download_overlap_results(save_path: str, bucket: str = BUCKET_NAME) -> bool:
    """Download overlap_results.pkl specifically."""
    return download_file(save_path, OVERLAP_FILE_NAME, bucket)


# -------------------
# Dune Cache Helpers
# -------------------
def upload_dune_cache_file(file_path: str, bucket: str = BUCKET_NAME) -> bool:
    """Upload a dune cache file into dune_cache/ folder."""
    filename = os.path.basename(file_path)
    return upload_file(file_path, bucket, f"dune_cache/{filename}")


def download_dune_cache_file(save_path: str, filename: str, bucket: str = BUCKET_NAME) -> bool:
    """Download a dune cache file from dune_cache/ folder."""
    return download_file(save_path, f"dune_cache/{filename}", bucket)

# -------------------
# Wallet PnL Helpers
# -------------------
WALLET_FOLDER = "wallet_pnl"

def wallet_file_name(wallet: str) -> str:
    """Return remote file path for a wallet JSON file."""
    return f"{WALLET_FOLDER}/{wallet}.json"


def upload_wallet_data(wallet: str, data: dict, bucket: str = BUCKET_NAME, debug: bool = True) -> bool:
    """Upload wallet balances + trades JSON to Supabase."""
    try:
        # Save JSON locally first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(json.dumps(data, indent=2, default=str).encode())
            tmp_path = tmp.name

        success = upload_file(tmp_path, bucket, wallet_file_name(wallet), debug=debug)
        os.remove(tmp_path)
        return success
    except Exception as e:
        if debug:
            print(f"❌ Failed to upload wallet {wallet}: {e}")
        return False


def download_wallet_data(wallet: str, bucket: str = BUCKET_NAME, debug: bool = True) -> dict | None:
    """Download wallet JSON from Supabase and return as dict."""
    save_path = f"/tmp/{wallet}.json"
    if not download_file(save_path, wallet_file_name(wallet), bucket=bucket):
        return None
    try:
        with open(save_path, "r") as f:
            return json.load(f)
    except Exception as e:
        if debug:
            print(f"⚠️ Failed to parse wallet JSON {wallet}: {e}")
        return None


def wallet_data_exists(wallet: str, bucket: str = BUCKET_NAME) -> bool:
    """Check if wallet data exists in Supabase Storage."""
    try:
        supabase = get_supabase_client()
        res = supabase.storage.from_(bucket).list(WALLET_FOLDER)
        if not isinstance(res, list):
            return False
        return any(obj.get("name") == f"{wallet}.json" for obj in res)
    except Exception:
        return False

# -------------------
# Script Runner
# -------------------
if __name__ == "__main__":
    test_pkl_path = r"C:\Users\HP USER\Documents\Data Analyst\degen smart\overlap_results (3).pkl"
    upload_overlap_results(test_pkl_path)
    download_overlap_results("./downloaded_overlap_results.pkl")
