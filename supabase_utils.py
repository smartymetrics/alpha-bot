#!/usr/bin/env python3
"""
supabase_utils.py
Utility functions for uploading/downloading files to/from Supabase Storage.
Uploads both .pkl and .json versions of overlap results.
Adds Dexscreener price enrichment (priceUsd) for each token.
Overwrites existing files (no upsert).
Supports dune cache file uploads under dune_cache/.

NEW: Added support for 'alpha' overlap results with
     upload_alpha_overlap_results().

REVISION:
- Download functions now use signed URLs and 'If-Modified-Since' / 'If-None-Match' (ETag)
  headers to work with private buckets and avoid re-downloading unchanged files.
- An in-memory cache stores 'Last-Modified' and 'ETag' times for conditional GETs.
"""

import os
import json
import pickle
import requests
from supabase import create_client, Client
import tempfile
from typing import Optional, Dict, Any, Union

BUCKET_NAME = "monitor-data"

# Original overlap files
OVERLAP_FILE_NAME = "overlap_results.pkl"
OVERLAP_JSON_NAME = "overlap_results.json"

# Alpha overlap files
OVERLAP_ALPHA_FILE_NAME = "overlap_results_alpha.pkl"
OVERLAP_ALPHA_JSON_NAME = "overlap_results_alpha.json"

MAX_SIZE_MB = 1.7

# --- In-Memory Cache for Conditional Fetching ---
# This dictionary will store 'Last-Modified' and 'ETag' headers for each file path.
_file_cache_headers: Dict[str, Dict[str, str]] = {}


# -------------------
# Supabase Client
# -------------------
def get_supabase_client() -> Client:
    """Create and return a Supabase client. Uses env vars with local fallback."""
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ldraroaloinsesjoayxc.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

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
        # if debug:
        #     print(f"💰 Dexscreener price for {token_id}: {price}")
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

    # New 'alpha' structure
    if isinstance(history_entry.get("result"), dict):
        grade = history_entry["result"].get("grade")
        if isinstance(grade, str):
            return grade

    # Original structure
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

    # New 'alpha' structure
    ts_field = history_entry.get("ts")
    if isinstance(ts_field, str):
        return ts_field

    # Original structure
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
        
        # Get grade from the *latest* history entry
        grade = safe_get_grade(history[-1])
        
        if grade != "NONE":
            latest = history[-1]
            
            # --- Handle both original and alpha structures ---
            target_dict = None
            if "result" in latest and isinstance(latest["result"], dict):
                # This is 'alpha' structure: history[-1] = {"ts":..., "result":{...}}
                target_dict = latest["result"]
            elif "grade" in latest:
                # This is 'original' structure: history[-1] = {"grade":..., "token":...}
                target_dict = latest
            else:
                target_dict = latest # Fallback
            
            # Ensure dexscreener section exists
            if "dexscreener" not in target_dict or not isinstance(target_dict.get("dexscreener"), dict):
                target_dict["dexscreener"] = {}
                
            # Add current price if missing
            if "current_price_usd" not in target_dict["dexscreener"]:
                price = fetch_dexscreener_price(token_id, debug=debug)
                target_dict["dexscreener"]["current_price_usd"] = price
            
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
# Upload Functions (Private bucket compatible)
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
        pass # Fails if file doesn't exist, which is fine

    try:
        with open(file_path, "rb") as f:
            data = f.read()
        # Uploading with the service key (from get_supabase_client)
        # automatically works for private buckets.
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
# NEW: Alpha Upload
# -------------------
def upload_alpha_overlap_results(file_path: str, bucket: str = BUCKET_NAME, debug: bool = True) -> bool:
    """
    Upload overlap_results_alpha.pkl + JSON, with Dexscreener enrichment.
    (Filters NONE grades before upload)
    """
    if not os.path.exists(file_path):
        if debug:
            print(f"❌ Missing {file_path}")
        return False

    # Generate enriched JSON (which also filters NONE) and update PKL
    json_bytes = prepare_json_from_pkl(file_path, debug=debug)
    
    # Upload PKL
    if not upload_file(file_path, bucket, OVERLAP_ALPHA_FILE_NAME, debug=debug):
        if debug:
            print(f"❌ Alpha PKL upload failed for {OVERLAP_ALPHA_FILE_NAME}")
        return False

    # Upload JSON
    try:
        supabase = get_supabase_client()
        supabase.storage.from_(bucket).remove([OVERLAP_ALPHA_JSON_NAME])
        supabase.storage.from_(bucket).upload(
            OVERLAP_ALPHA_JSON_NAME, json_bytes, {"content-type": "application/json"}
        )
        if debug:
            print(f"✅ Uploaded {OVERLAP_ALPHA_JSON_NAME} ({len(json_bytes)/1024:.2f} KB)")
    except Exception as e:
        if debug:
            print(f"❌ Alpha JSON upload failed: {e}")
        return False

    return True


# -------------------
# MODIFIED: Download Functions (Private + Conditional)
# -------------------
def download_file(save_path: str, file_name: str, bucket: str = BUCKET_NAME) -> Optional[bytes]:
    """
    Download file from private Supabase Storage using signed URL and
    conditional GET with 'If-Modified-Since' and 'If-None-Match' (ETag).
    If not modified (304), loads from local `save_path`.
    If modified (200), downloads, saves to `save_path`, and returns content.
    Returns file content as bytes if successful, None otherwise.
    """
    global _file_cache_headers
    supabase = get_supabase_client()

    try:
        # 1. Generate a 60-second signed URL for the private file
        signed_url_response = supabase.storage.from_(bucket).create_signed_url(file_name, 60)
        signed_url = signed_url_response.get('signedURL')
        if not signed_url:
            print(f"Error: Could not generate signed URL for '{file_name}'. Response: {signed_url_response}")
            return None

        # 2. Prepare headers for conditional GET using ETag and Last-Modified
        headers = {}
        cached_headers = _file_cache_headers.get(file_name, {})
        if cached_headers.get('Last-Modified'):
            headers['If-Modified-Since'] = cached_headers['Last-Modified']
        if cached_headers.get('ETag'):
            headers['If-None-Match'] = cached_headers['ETag']

        # 3. Perform the HTTP request
        response = requests.get(signed_url, headers=headers, timeout=15)

        # 4. Handle the response
        if response.status_code == 304:
            # 304 Not Modified: File hasn't changed
            print(f"File '{file_name}': No change detected (304 Not Modified).")
            if os.path.exists(save_path):
                print(f"Loading from local cache: '{save_path}'")
                with open(save_path, "rb") as f:
                    return f.read()
            else:
                # File not modified, but local copy is missing. Force re-download.
                print(f"File '{file_name}' not modified, but local file '{save_path}' missing. Forcing re-download.")
                headers.pop('If-Modified-Since', None)
                headers.pop('If-None-Match', None)
                _file_cache_headers.pop(file_name, None)
                response = requests.get(signed_url, headers=headers, timeout=15)
                # Allow to fall through to 200 logic

        if response.status_code == 200:
            # 200 OK: File is new or has been updated
            print(f"File '{file_name}': File updated — new data loaded.")
            data = response.content

            # Update our cache with the new 'Last-Modified' and 'ETag' headers
            new_last_modified = response.headers.get('Last-Modified')
            new_etag = response.headers.get('ETag')
            
            new_headers_to_cache = {}
            if new_last_modified:
                new_headers_to_cache['Last-Modified'] = new_last_modified
            if new_etag:
                new_headers_to_cache['ETag'] = new_etag
                
            if new_headers_to_cache:
                _file_cache_headers[file_name] = new_headers_to_cache
                print(f"File '{file_name}': Updated cache headers (ETag: {new_etag}, Last-Modified: {new_last_modified}).")

            # Save the new file content locally
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(data)
            print(f"Downloaded and saved '{file_name}' -> '{save_path}'")
            return data

        else:
            # Handle other errors (404 Not Found, 403 Forbidden, 500, etc.)
            print(f"File '{file_name}': Error fetching file. Status: {response.status_code}, Response: {response.text[:100]}...")
            # Check if file exists locally as a last resort
            if os.path.exists(save_path):
                print(f"Loading from local cache as fallback: '{save_path}'")
                with open(save_path, "rb") as f:
                    return f.read()
            return None

    except requests.exceptions.RequestException as e:
        print(f"File '{file_name}': Network or request error. {e}")
    except Exception as e:
        print(f"File '{file_name}': An unexpected error occurred. {e}")

    # Fallback: Try to load from local cache if any error occurs
    if os.path.exists(save_path):
        print(f"Loading from local cache as fallback: '{save_path}'")
        try:
            with open(save_path, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read local fallback file: {e}")

    return None


def download_overlap_results(save_path: str, bucket: str = BUCKET_NAME) -> Optional[bytes]:
    """Download overlap_results.pkl specifically."""
    return download_file(save_path, OVERLAP_FILE_NAME, bucket)


# -------------------
# --- ADDED THIS FUNCTION ---
# -------------------
def download_alpha_overlap_results(save_path: str, bucket: str = BUCKET_NAME) -> Optional[bytes]:
    """Download overlap_results_alpha.pkl specifically."""
    return download_file(save_path, OVERLAP_ALPHA_FILE_NAME, bucket)
# -------------------
# --- END OF ADDITION ---
# -------------------


# -------------------
# Dune Cache Helpers
# -------------------
def upload_dune_cache_file(file_path: str, bucket: str = BUCKET_NAME) -> bool:
    """Upload a dune cache file into dune_cache/ folder."""
    filename = os.path.basename(file_path)
    return upload_file(file_path, bucket, f"dune_cache/{filename}")


def download_dune_cache_file(save_path: str, filename: str, bucket: str = BUCKET_NAME) -> Optional[bytes]:
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
    save_path = f"/tmp/{wallet}.json" # Use temp dir for local save
    
    data_bytes = download_file(save_path, wallet_file_name(wallet), bucket=bucket)
    
    if not data_bytes:
        if debug:
            print(f"Failed to download wallet data for {wallet}")
        return None
        
    try:
        # Decode bytes and load as JSON
        return json.loads(data_bytes.decode('utf-8'))
    except json.JSONDecodeError as e:
        if debug:
            print(f"⚠️ Failed to parse wallet JSON {wallet}: {e}")
        return None
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
    # Test original upload
    test_pkl_path = r"C:\Users\HP USER\Documents\Data Analyst\degen smart\overlap_results (3).pkl"
    if os.path.exists(test_pkl_path):
        upload_overlap_results(test_pkl_path)
        print("\n--- Testing Main Download (1st time) ---")
        download_overlap_results("./downloaded_overlap_results.pkl")
        print("\n--- Testing Main Download (2nd time, should be 304) ---")
        download_overlap_results("./downloaded_overlap_results.pkl")
    else:
        print(f"Test file not found: {test_pkl_path}")
        
    # Test alpha upload
    test_alpha_path = "./data/overlap_results_alpha.pkl"
    if os.path.exists(test_alpha_path):
        print("\nTesting Alpha Upload...")
        upload_alpha_overlap_results(test_alpha_path)
        print("\n--- Testing Alpha Download (1st time) ---")
        download_file("./downloaded_alpha.pkl", OVERLAP_ALPHA_FILE_NAME)
        print("\n--- Testing Alpha Download (2nd time, should be 304) ---")
        download_file("./downloaded_alpha.pkl", OVERLAP_ALPHA_FILE_NAME)
    else:
        print(f"\nTest file not found: {test_alpha_path}")
        # Create dummy file for testing
        try:
            os.makedirs("./data", exist_ok=True)
            dummy_data = {
                "tokenA": [{"ts": "2025-01-01T00:00:00Z", "result": {"grade": "HIGH"}}],
                "tokenB": [{"ts": "2025-01-01T01:00:00Z", "result": {"grade": "NONE"}}]
            }
            with open(test_alpha_path, "wb") as f:
                pickle.dump(dummy_data, f)
            print("Created dummy alpha file for testing.")
            upload_alpha_overlap_results(test_alpha_path)
            print("\n--- Testing Alpha Download (1st time) ---")
            download_file("./downloaded_alpha.pkl", OVERLAP_ALPHA_FILE_NAME)
            print("\n--- Testing Alpha Download (2nd time, should be 304) ---")
            download_file("./downloaded_alpha.pkl", OVERLAP_ALPHA_FILE_NAME)
        except Exception as e:
            print(f"Failed to create/upload dummy alpha file: {e}")