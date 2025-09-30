#!/usr/bin/env python3
"""
alpha.py - Solana Early Trader PNL Analysis (with API-key rotation + failover,
           corrected, optimized, Supabase upload)

Usage:
    python alpha.py <token_mint1> [token_mint2 token_mint3]
    Optional: --window (hours), --minbuy (USD), --minprof (num profitable tokens required)

Notes:
 - Up to 3 tokens allowed.
 - Supports multiple provider keys via environment variables:
     MORALIS_KEYS (comma-separated) or MORALIS_API_KEY (legacy single; also accepts comma-separated)
     HELIUS_KEYS  (comma-separated) or HELIUS_API_KEY  (legacy single; also accepts comma-separated)
 - Requires SUPABASE_URL, SUPABASE_KEY to upload/list analyses.
 - Caches Dexscreener current prices in sol_price_cache.pkl using joblib.
 - If a request returns 401 or 429 the code will rotate to the next key and retry.
"""

import os
import time
import argparse
import requests
import pandas as pd
import numpy as np
import joblib
import asyncio
import aiohttp
import json
import logging
import math
from datetime import timedelta, datetime, timezone
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# --------------------
# Config & Keys
# --------------------
CACHE_FILE = "sol_price_cache.pkl"

# Support both single-key env var and comma-separated multi-key env var.
# Also handle the case where the user accidentally placed comma-separated keys in the single var.
def _load_keys(env_name_single: str, env_name_multi: str) -> List[str]:
    """
    Attempt to load keys from env_name_multi first (preferred). If not present,
    fall back to env_name_single. If either contains commas, split them.
    Returns a list of stripped keys (no empty entries).
    """
    multi = os.environ.get(env_name_multi)
    single = os.environ.get(env_name_single)

    keys: List[str] = []
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
    elif single:
        # allow the single variable to contain a comma-separated list as well
        if "," in single:
            keys = [k.strip() for k in single.split(",") if k.strip()]
        else:
            keys = [single.strip()]
    return keys

MORALIS_KEYS = _load_keys("MORALIS_API_KEY", "MORALIS_KEYS")
HELIUS_KEYS = _load_keys("HELIUS_API_KEY", "HELIUS_KEYS")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
# set your bucket name here
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "monitor-data")

DEFAULT_EARLY_WINDOW_HOURS = 6
DEFAULT_MINIMUM_INITIAL_BUY_USD = 100.0
DEFAULT_MIN_PROFITABLE_TRADES = 1

# Base tokens (used to identify buys)
BASE_TOKENS = {
    "So11111111111111111111111111111111111111112",  # SOL (example)
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"   # USDC
}

# --------------------
# Logging
# --------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=os.environ.get("LOGLEVEL", "INFO"),
)
logger = logging.getLogger("alpha")

# threshold (seconds) to warn for long RPC/API calls
LONG_CALL_THRESHOLD = float(os.environ.get("LONG_CALL_THRESHOLD", 2.0))

# simple counters for round-robin rotation
_moralis_idx = 0
_helius_idx = 0

# sets to hold bad keys (soft blacklist) to avoid repeatedly trying known-401 keys
_bad_moralis_keys = set()
_bad_helius_keys = set()

# --------------------
# Helpers
# --------------------
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def ensure_keys_present(require_supabase: bool = False):
    if not MORALIS_KEYS:
        raise RuntimeError("MORALIS_API_KEY / MORALIS_KEYS not set in environment (no Moralis keys found)")
    if not HELIUS_KEYS:
        raise RuntimeError("HELIUS_API_KEY / HELIUS_KEYS not set in environment (no Helius keys found)")
    if require_supabase:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set in environment")

def _mask_key(s: str, keep: int = 6) -> str:
    """Return a masked version of a key (show last `keep` chars)."""
    if not s:
        return "None"
    s = str(s)
    if len(s) <= keep:
        return s
    return "..." + s[-keep:]

def _valid_moralis_keys() -> List[str]:
    """Return Moralis keys excluding soft-blacklisted ones."""
    return [k for k in MORALIS_KEYS if k not in _bad_moralis_keys]

def _valid_helius_keys() -> List[str]:
    """Return Helius keys excluding soft-blacklisted ones."""
    return [k for k in HELIUS_KEYS if k not in _bad_helius_keys]

def _next_moralis_key_round_robin() -> str:
    """Pick next Moralis key from non-blacklisted keys using a round-robin counter."""
    global _moralis_idx
    valid = _valid_moralis_keys()
    if not valid:
        # If nothing valid, consider all keys (reset blacklist) to allow retries
        logger.error("All Moralis keys are currently blacklisted (removed due to 401). Raise error.")
        raise RuntimeError("No valid Moralis keys available (all keys returned 401).")
    key = valid[_moralis_idx % len(valid)]
    used_idx = _moralis_idx
    _moralis_idx += 1
    logger.debug("[moralis] using key idx_round=%d masked=%s", used_idx, _mask_key(key))
    return key

def _next_helius_key_round_robin() -> str:
    """Pick next Helius key from non-blacklisted keys using a round-robin counter."""
    global _helius_idx
    valid = _valid_helius_keys()
    if not valid:
        logger.error("All Helius keys are currently blacklisted (removed due to 401). Raise error.")
        raise RuntimeError("No valid Helius keys available (all keys returned 401).")
    key = valid[_helius_idx % len(valid)]
    used_idx = _helius_idx
    _helius_idx += 1
    logger.debug("[helius] using key idx_round=%d masked=%s", used_idx, _mask_key(key))
    return key

# --------------------
# Supabase helpers
# --------------------
def get_supabase_client() -> Client:
    ensure_keys_present(require_supabase=True)
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def _supabase_path(folder: str, filename: str) -> str:
    return f"{folder.rstrip('/')}/{filename}"

def save_to_supabase(df: pd.DataFrame, tokens: List[str], params: Dict[str, Any], job_id: str="unknown"):
    ensure_keys_present(require_supabase=True)
    supabase = get_supabase_client()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = "recent_analyses"
    os.makedirs(folder, exist_ok=True)

    base_name = f"analysis_{ts}_{job_id}"
    pkl_file_local = os.path.join(folder, f"{base_name}.pkl")
    json_file_local = os.path.join(folder, f"{base_name}.json")

    joblib.dump(df, pkl_file_local)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokens": tokens,
        "params": params,
        "records": df.fillna(0).to_dict(orient="records")
    }
    with open(json_file_local, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved analysis locally: %s and %s", pkl_file_local, json_file_local)

    for local_path in [pkl_file_local, json_file_local]:
        filename = os.path.basename(local_path)
        object_path = _supabase_path(folder, filename)
        try:
            supabase.storage.from_(BUCKET_NAME).remove([object_path])
        except Exception:
            pass
        with open(local_path, "rb") as fh:
            data_bytes = fh.read()
            supabase.storage.from_(BUCKET_NAME).upload(object_path, data_bytes)
        logger.info("Uploaded to Supabase: %s", object_path)

def list_recent_analyses(limit: int = 100, folder: str = "recent_analyses") -> List[Dict[str, Any]]:
    ensure_keys_present(require_supabase=True)
    supabase = get_supabase_client()

    try:
        items = supabase.storage.from_(BUCKET_NAME).list(folder, limit=limit)
    except Exception as e:
        logger.error("Supabase list error: %s", e)
        return []

    results = []
    for it in items:
        name = it.get("name")
        object_path = _supabase_path(folder, name)
        try:
            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(object_path).get("publicURL")
        except Exception:
            public_url = None
        results.append({
            "name": name,
            "id": it.get("id"),
            "updated_at": it.get("updated_at"),
            "size": it.get("size"),
            "public_url": public_url
        })
    return results

# --------------------
# Request failover helpers
# --------------------
def requests_with_failover(url: str, method: str = "GET", params: dict = None, headers: dict = None,
                           json_payload: dict = None, max_attempts: int = None, provider: str = "moralis", timeout: int = 20):
    """
    Synchronous requests wrapper that rotates keys on 401/429 responses or connection errors.
    - provider: "moralis" supported for sync usage
    Behavior:
     - rotate across keys that are not soft-blacklisted
     - if a key returns 401, mark it as bad and skip it in the future
     - on 429 (rate limit) treat as transient and continue rotating
    """
    attempt = 0
    last_exception = None
    # default attempts: at least 6 or twice number of keys available
    if max_attempts is None:
        max_attempts = max(6, (len(MORALIS_KEYS) if MORALIS_KEYS else 1) * 2)

    while attempt < max_attempts:
        attempt += 1
        try:
            if provider == "moralis":
                # pick next valid moralis key
                try:
                    key = _next_moralis_key_round_robin()
                except RuntimeError as e:
                    # no valid keys left
                    raise
                use_headers = dict(headers or {})
                use_headers["Accept"] = "application/json"
                use_headers["X-API-Key"] = key
                resp = requests.request(method, url, params=params, headers=use_headers, json=json_payload, timeout=timeout)
            else:
                use_headers = dict(headers or {})
                resp = requests.request(method, url, params=params, headers=use_headers, json=json_payload, timeout=timeout)

            # handle status codes
            if resp.status_code == 401:
                # mark this key as bad (soft blacklist) to avoid trying it repeatedly
                if provider == "moralis":
                    _bad_moralis_keys.add(key)
                    logger.warning("[moralis] key masked=%s returned 401; blacklisting this key index.", _mask_key(key))
                logger.warning("[%s] request returned 401 (attempt %d). Rotating key and retrying...", provider, attempt)
                time.sleep(min(1 + attempt * 0.5, 5.0))
                continue

            if resp.status_code == 429:
                logger.warning("[%s] request returned 429 (rate limited) (attempt %d). Rotating key and retrying...", provider, attempt)
                time.sleep(min(0.8 + attempt * 0.5, 5.0))
                continue

            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"_raw_text": resp.text}
        except requests.RequestException as e:
            last_exception = e
            logger.warning("[%s] request exception (attempt %d): %s", provider, attempt, e)
            time.sleep(min(0.5 * attempt, 5.0))
            continue

    raise RuntimeError(f"All attempts failed for {provider}. Last exception: {last_exception}")

async def aiohttp_with_failover_post(payload: dict, max_attempts: int = None, timeout: int = 30) -> dict:
    """
    Async wrapper to POST to Helius RPC, rotating through HELIUS_KEYS on 429/401 & retrying.
    Returns parsed json.
    """
    attempt = 0
    last_exc = None
    if max_attempts is None:
        max_attempts = max(6, (len(HELIUS_KEYS) if HELIUS_KEYS else 1) * 2)

    while attempt < max_attempts:
        attempt += 1
        try:
            key = _next_helius_key_round_robin()
        except RuntimeError:
            raise
        helius_url = f"https://mainnet.helius-rpc.com/?api-key={key}"
        try:
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(helius_url, json=payload, timeout=timeout) as resp:
                    elapsed = time.time() - start
                    if elapsed > LONG_CALL_THRESHOLD:
                        logger.warning("Long Helius RPC call method=%s elapsed=%.2fs", payload.get("method"), elapsed)
                    if resp.status == 401:
                        # blacklist this key
                        _bad_helius_keys.add(key)
                        logger.warning("[helius] key masked=%s returned 401; blacklisting this key index.", _mask_key(key))
                        await asyncio.sleep(min(0.5 * attempt, 5.0))
                        continue
                    if resp.status == 429:
                        logger.warning("[helius] returned 429 (rate limited) on attempt %d, rotating key...", attempt)
                        await asyncio.sleep(min(0.5 * attempt, 5.0))
                        continue
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            last_exc = e
            logger.warning("[helius] exception on attempt %d: %s", attempt, e)
            await asyncio.sleep(min(0.5 * attempt, 5.0))
            continue

    raise RuntimeError(f"All helius attempts failed. Last exception: {last_exc}")

# --------------------
# Moralis trades downloader (with timing logs)
# --------------------
def _timed_get(url, **kwargs):
    start = time.time()
    resp = requests.get(url, **kwargs)
    elapsed = time.time() - start
    if elapsed > LONG_CALL_THRESHOLD:
        logger.warning("Long GET call: %s (%.2fs)", url, elapsed)
    return resp

def get_solana_dex_trades(token_addresses: List[str], limit: int = 100) -> pd.DataFrame:
    """
    Normalize Moralis swaps into rows where:
      - mint_address = the token mint of interest (the token side, not the base)
      - token_amount = amount of that mint (positive)
      - other_mint_address/other_amount = counterparty token & amount
      - is_buy = True when trader acquired the token (sold base token to get token)
    Uses explicit base-token rule: if token_sold_mint_address in BASE_TOKENS then it's a buy.
    """
    if not isinstance(token_addresses, list):
        raise TypeError("token_addresses must be a list")
    if len(token_addresses) == 0:
        return pd.DataFrame()
    if len(token_addresses) > 3:
        raise ValueError("A maximum of 3 tokens is allowed per request.")
    ensure_keys_present()

    all_rows = []

    for addr in token_addresses:
        url = f"https://solana-gateway.moralis.io/token/mainnet/{addr}/swaps"
        cursor = None
        logger.info("[Moralis] fetching swaps for %s", addr)
        while True:
            params = {"limit": limit, "order": "DESC", "transactionTypes": "buy,sell"}
            if cursor:
                params["cursor"] = cursor
            try:
                data = requests_with_failover(url, method="GET", params=params, provider="moralis",
                                              max_attempts=max(6, len(MORALIS_KEYS) * 2 if MORALIS_KEYS else 6))
            except Exception as e:
                logger.error("[Moralis] request error for %s: %s", addr, e)
                break

            trades = []
            if isinstance(data, dict):
                # Handle different Moralis API response formats
                result = data.get("result")

                if isinstance(result, list):
                    # Case 1: API returned trades directly as a list
                    trades = result
                elif isinstance(result, dict):
                    # Case 2: API returned a dict, trades might be in "data" field  
                    trades = result.get("data", [])
                else:
                    # Case 3: result is None or unexpected type, try fallback
                    trades = []
                    
                    # Try some alternative response structures as fallback
                    if "data" in data and isinstance(data["data"], list):
                        trades = data["data"]
                    elif "trades" in data and isinstance(data["trades"], list):
                        trades = data["trades"]

                    if not trades:
                        break
                    # Validate that trades are actually for the token we requested
                    valid_trades = []
                    for t in trades:
                        bought = t.get("bought") or {}
                        sold = t.get("sold") or {}
                        bought_addr = (bought.get("address") or "").strip()
                        sold_addr = (sold.get("address") or "").strip()
                        
                        # At least one address should match our target token
                        if addr in [bought_addr, sold_addr]:
                            valid_trades.append(t)
                        else:
                            logger.warning(f"Skipping invalid trade for {addr}: bought={bought_addr}, sold={sold_addr}")

                    trades = valid_trades

                # Ensure trades is always a list
                if not isinstance(trades, list):
                    trades = []
            elif isinstance(data, list):
                trades = data
            else:
                trades = []

            if not trades:
                break

            for t in trades:
                direction = (t.get("transactionType") or "").lower()
                bought = t.get("bought") or {}
                sold = t.get("sold") or {}

                # raw addresses
                bought_addr = (bought.get("address") or "").strip()
                sold_addr = (sold.get("address") or "").strip()

                # Determine is_buy using explicit base-token rule:
                # - If sold_addr is a base token, trader sold base => they bought the token (is_buy=True)
                # - Else if bought_addr is a base token, trader bought base => they sold the token (is_buy=False)
                # - Else fall back to Moralis transactionType
                if sold_addr and sold_addr in BASE_TOKENS and (not bought_addr or bought_addr not in BASE_TOKENS):
                    is_buy = True
                elif bought_addr and bought_addr in BASE_TOKENS and (not sold_addr or sold_addr not in BASE_TOKENS):
                    is_buy = False
                else:
                    is_buy = (direction == "buy")

                # Normalize so mint_address/token_amount always refer to the token of interest
                if is_buy:
                    mint_address = bought_addr or None
                    token_amount = _safe_float(bought.get("amount"))
                    other_mint_address = sold_addr or None
                    other_amount = _safe_float(sold.get("amount"))
                else:
                    mint_address = sold_addr or None
                    token_amount = _safe_float(sold.get("amount"))
                    other_mint_address = bought_addr or None
                    other_amount = _safe_float(bought.get("amount"))

                amount_usd = _safe_float(t.get("totalValueUsd"))
                price_usd_at_trade = None
                if amount_usd is not None and token_amount not in (None, 0):
                    price_usd_at_trade = amount_usd / token_amount

                bt = pd.to_datetime(t.get("blockTimestamp"))

                all_rows.append({
                    "block_time": bt,
                    "block_date": bt.normalize() if not pd.isna(bt) else None,
                    "transaction_hash": t.get("transactionHash"),
                    "transaction_type": direction,
                    "is_buy": is_buy,
                    "trader_id": t.get("walletAddress"),
                    "pair_address": t.get("pairAddress"),
                    "pair_label": t.get("pairLabel"),
                    "exchange_name": t.get("exchangeName"),
                    "mint_address": mint_address,
                    "token_amount": token_amount,
                    "other_mint_address": other_mint_address,
                    "other_amount": other_amount,
                    "amount_usd": amount_usd,
                    "price_usd_at_trade": price_usd_at_trade
                })

            if isinstance(data, dict):
                cursor = data.get("cursor") or data.get("next")
            else:
                cursor = None

            if not cursor:
                break

            time.sleep(0.15)

        logger.info("[Moralis] fetched %d total trades (so far).", len(all_rows))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # derived_price fallback logic uses token_amount now
    df["derived_price"] = df["price_usd_at_trade"]
    mask = df["derived_price"].isna() & df["token_amount"].notna() & df["amount_usd"].notna()
    if not df.empty and mask.any():
        df.loc[mask, "derived_price"] = df.loc[mask, "amount_usd"] / df.loc[mask, "token_amount"].replace({0: np.nan})
    df["block_time"] = pd.to_datetime(df["block_time"])
    df = df.sort_values("block_time").reset_index(drop=True)
    logger.info("Total trades fetched: %d", len(df))
    return df

# --------------------
# Price builder (per-minute OHLC + vwap)
# --------------------
def build_minute_prices_from_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build OHLCV + VWAP minute-level candlesticks for tokens from trade data.

    Args:
        trades_df: DataFrame with at least:
            - block_time (datetime)
            - mint_address (str)
            - price_usd_at_trade or derived_price
            - amount_usd (float)

    Returns:
        DataFrame with columns:
            mint_address, minute, open, high, low, close, vwap, volume_usd
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "mint_address",
                "minute",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume_usd",
            ]
        )

    df = trades_df.copy()

    # Ensure datetime
    df["block_time"] = pd.to_datetime(df["block_time"])

    # Prefer price_usd_at_trade, fallback to derived_price
    if "price_usd_at_trade" in df.columns:
        df["price_for_price_series"] = df["price_usd_at_trade"].fillna(
            df.get("derived_price")
        )
    else:
        df["price_for_price_series"] = df.get("derived_price")

    # Round down to the nearest minute
    df["minute"] = df["block_time"].dt.floor("min")

    # Group by token mint + minute
    grouped = df.groupby(["mint_address", "minute"], sort=True)

    rows = []
    for (mint, minute), g in grouped:
        prices = g["price_for_price_series"].dropna()
        if prices.empty:
            continue

        open_p = prices.iloc[0]
        high_p = prices.max()
        low_p = prices.min()
        close_p = prices.iloc[-1]

        # VWAP calculation with division-by-zero guard
        if "amount_usd" in g.columns and g["amount_usd"].notna().any():
            weights = g["amount_usd"].fillna(0.0)
            denom = weights.sum()
            if denom > 0:
                vwap = (prices * weights).sum() / denom
            else:
                vwap = prices.mean()
        else:
            vwap = prices.mean()

        # Total traded volume (in USD terms)
        volume_usd = g.get("amount_usd", pd.Series(dtype=float)).fillna(0.0).sum()

        rows.append(
            {
                "mint_address": mint,
                "minute": minute,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "vwap": vwap,
                "volume_usd": volume_usd,
            }
        )

    price_df = pd.DataFrame(rows)
    if not price_df.empty:
        price_df = price_df.sort_values(["mint_address", "minute"]).reset_index(
            drop=True
        )

    return price_df

# --------------------
# Vectorized price lookup helpers
# --------------------
def attach_minute_price(df_in: pd.DataFrame, minute_price_df: pd.DataFrame, mint_col: str, time_col: str, price_col_name: str = "minute_vwap", tolerance_minutes: int = 5) -> pd.DataFrame:
    if df_in.empty:
        df_in[price_col_name] = np.nan
        return df_in

    if minute_price_df.empty:
        df_in[price_col_name] = np.nan
        return df_in

    left = df_in.copy()
    left["minute"] = left[time_col].dt.floor("min")
    mp = minute_price_df[["mint_address", "minute", "vwap"]].rename(columns={"mint_address": mint_col, "vwap": price_col_name})
    merged = pd.merge(left, mp, on=[mint_col, "minute"], how="left", copy=False)

    mask_missing = merged[price_col_name].isna()
    if mask_missing.any():
        left_asof = merged[mask_missing].sort_values([mint_col, "minute"]).copy()
        right_asof = mp.sort_values([mint_col, "minute"]).copy()
        try:
            left_asof = pd.merge_asof(
                left_asof.sort_values("minute"),
                right_asof.sort_values("minute"),
                on="minute",
                by=mint_col,
                direction="nearest",
                tolerance=pd.Timedelta(minutes=tolerance_minutes)
            )
            merged.loc[left_asof.index, price_col_name] = left_asof[price_col_name].values
        except Exception as e:
            logger.debug("merge_asof fallback failed: %s", e)

    return merged.drop(columns=["minute"])

# --------------------
# Dexscreener caching
# --------------------
def load_price_cache():
    try:
        cache = joblib.load(CACHE_FILE)
        return cache if isinstance(cache, dict) else {}
    except Exception:
        return {}

def save_price_cache(cache: dict):
    try:
        joblib.dump(cache, CACHE_FILE)
    except Exception as e:
        logger.warning("Failed to save cache: %s", e)

async def dexscreener_fetch_price(mint: str, session: aiohttp.ClientSession) -> Optional[float]:
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
    start = time.time()
    try:
        async with session.get(url, timeout=10) as resp:
            elapsed = time.time() - start
            if elapsed > LONG_CALL_THRESHOLD:
                logger.warning("Long Dexscreener call for %s: %.2fs", mint, elapsed)
            if resp.status != 200:
                return None
            data = await resp.json()
            pairs = data.get("pairs") or []
            if not pairs:
                return None
            return _safe_float(pairs[0].get("priceUsd"))
    except Exception as e:
        logger.debug("Dexscreener fetch error for %s: %s", mint, e)
        return None

async def get_current_prices_for_mints_async(mints: List[str]) -> dict:
    cache = load_price_cache()
    result = {}
    to_fetch = []
    for m in mints:
        if m in cache and isinstance(cache[m], (int, float)):
            result[m] = cache[m]
        else:
            to_fetch.append(m)
    if to_fetch:
        async with aiohttp.ClientSession() as session:
            tasks = [dexscreener_fetch_price(m, session) for m in to_fetch]
            fetched = await asyncio.gather(*tasks)
            for m, p in zip(to_fetch, fetched):
                p = 0.0 if p is None else p
                cache[m] = p
                result[m] = p
        save_price_cache(cache)
    return result

# --------------------
# Helius RPC helpers (async)
# --------------------
async def fetch_solana_rpc(payload: dict) -> dict:
    return await aiohttp_with_failover_post(payload, max_attempts=max(6, len(HELIUS_KEYS) * 2 if HELIUS_KEYS else 6))

async def get_current_balances_and_prices(traders: List[str], token_mints: List[str]) -> pd.DataFrame:
    token_mints = list({m.strip() for m in token_mints})
    prices_map = await get_current_prices_for_mints_async(token_mints)
    all_rows = []

    for trader in traders:
        sol_payload = {"jsonrpc":"2.0","id":1,"method":"getBalance","params":[trader]}
        try:
            sol_res = await fetch_solana_rpc(sol_payload)
            lamports = sol_res.get("result",{}).get("value",0)
            sol_balance = lamports / 1e9 if lamports is not None else 0.0
        except Exception as e:
            logger.debug("SOL balance fetch failed for %s: %s", trader, e)
            sol_balance = 0.0

        tokens_payload = {
            "jsonrpc":"2.0","id":"1","method":"getTokenAccountsByOwner",
            "params":[trader, {"programId":"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}, {"encoding":"jsonParsed"}]
        }
        try:
            tok_res = await fetch_solana_rpc(tokens_payload)
            accounts = tok_res.get("result",{}).get("value",[]) or []
        except Exception as e:
            logger.debug("Token accounts fetch failed for %s: %s", trader, e)
            accounts = []

        balances_map = {m: 0.0 for m in token_mints}
        for acc in accounts:
            info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}) or {}
            mint = (info.get("mint") or "").strip()
            if not mint:
                continue
            if mint in balances_map:
                ta = info.get("tokenAmount", {}) or {}
                balances_map[mint] = float(ta.get("uiAmount") or 0.0)

        for mint, bal in balances_map.items():
            price = prices_map.get(mint, 0.0) or 0.0
            all_rows.append({
                "trader_id": trader,
                "sol_balance": sol_balance,
                "mint_address": mint,
                "token_balance": bal,
                "current_price": price,
                "current_value_usd": bal * price
            })

        await asyncio.sleep(0.05)

    return pd.DataFrame(all_rows)

# --------------------
# Main pipeline (vectorized)
# --------------------
def run_pipeline(tokens: List[str],
                 early_trading_window_hours: Optional[int] = None,
                 minimum_initial_buy_usd: Optional[float] = None,
                 min_profitable_trades: int = DEFAULT_MIN_PROFITABLE_TRADES, 
                 job_id: str = "unknown") -> Optional[pd.DataFrame]:
    ensure_keys_present()
    tokens = [t.strip() for t in tokens]
    if len(tokens) == 0:
        logger.error("No tokens provided.")
        return
    if len(tokens) > 3:
        raise ValueError("A maximum of 3 token addresses is allowed.")
    logger.info("[Run] tokens=%s window=%s minbuy=%s minprof=%s", tokens, early_trading_window_hours, minimum_initial_buy_usd, min_profitable_trades)

    # Fetch trades and build minute prices
    trades_df = get_solana_dex_trades(tokens)
    if trades_df.empty:
        logger.info("No trades returned from Moralis.")
        return

    # ADD THIS VALIDATION:
    unique_tokens = trades_df['mint_address'].unique()
    logger.info(f"[Pipeline] Unique tokens in trades_df: {unique_tokens}")
    logger.info(f"[Pipeline] Expected tokens: {tokens}")

    # Check if we got unexpected tokens
    unexpected_tokens = [t for t in unique_tokens if t not in tokens]
    if unexpected_tokens:
        logger.warning(f"[Pipeline] Found unexpected tokens: {unexpected_tokens}")
        # Filter to only requested tokens
        trades_df = trades_df[trades_df['mint_address'].isin(tokens)]
        logger.info(f"[Pipeline] After filtering: {len(trades_df)} trades remaining")


    minute_price_df = build_minute_prices_from_trades(trades_df)

    # token launch times per normalized mint_address
    token_first_trade_times = trades_df.groupby("mint_address", as_index=False).agg(token_launch_time=("block_time", "min"))

    # -------------------------
    # First personal buy (cohort)
    # -------------------------
    buys_mask = trades_df["is_buy"] & trades_df["token_amount"].notna() & (trades_df["token_amount"] > 0)
    buys_only = trades_df[buys_mask].copy()
    if not buys_only.empty:
        idx = buys_only.groupby(["trader_id", "mint_address"])["block_time"].idxmin()
        tfbd = buys_only.loc[idx].copy()
        tfbd = tfbd.rename(columns={"block_time": "first_personal_buy_time"})
        tfbd = tfbd[["trader_id", "mint_address", "first_personal_buy_time", "token_amount", "amount_usd", "derived_price", "block_date"]]
    else:
        tfbd = pd.DataFrame(columns=["trader_id", "mint_address", "first_personal_buy_time", "token_amount", "amount_usd", "derived_price", "block_date"])

    if not tfbd.empty:
        tfbd = attach_minute_price(tfbd, minute_price_df, mint_col="mint_address", time_col="first_personal_buy_time", price_col_name="minute_vwap", tolerance_minutes=5)
        cond_amount_usd = tfbd["amount_usd"].notna() & (tfbd["amount_usd"] > 0)
        cond_minute = tfbd["minute_vwap"].notna() & tfbd["token_amount"].notna()
        cond_derived = tfbd["derived_price"].notna() & tfbd["token_amount"].notna()

        tfbd["first_buy_usd_amount"] = 0.0
        tfbd.loc[cond_amount_usd, "first_buy_usd_amount"] = tfbd.loc[cond_amount_usd, "amount_usd"].astype(float)
        tfbd.loc[~cond_amount_usd & cond_minute, "first_buy_usd_amount"] = (
            tfbd.loc[~cond_amount_usd & cond_minute, "minute_vwap"].astype(float)
            * tfbd.loc[~cond_amount_usd & cond_minute, "token_amount"].astype(float)
        )
        tfbd.loc[~cond_amount_usd & ~cond_minute & cond_derived, "first_buy_usd_amount"] = (
            tfbd.loc[~cond_amount_usd & ~cond_minute & cond_derived, "derived_price"].astype(float)
            * tfbd.loc[~cond_amount_usd & ~cond_minute & cond_derived, "token_amount"].astype(float)
        )
    else:
        tfbd["first_buy_usd_amount"] = pd.Series(dtype=float)

    # merge to compute time since launch & cohort filtering
    merged_for_cohort = pd.merge(tfbd, token_first_trade_times, on="mint_address", how="left")
    merged_for_cohort["time_since_launch"] = merged_for_cohort["first_personal_buy_time"] - merged_for_cohort["token_launch_time"]

    cohort_df = merged_for_cohort.copy()
    if early_trading_window_hours is not None and early_trading_window_hours > 0:
        cohort_df = cohort_df[cohort_df["time_since_launch"] <= timedelta(hours=early_trading_window_hours)]
    if minimum_initial_buy_usd is not None and minimum_initial_buy_usd > 0:
        cohort_df = cohort_df[cohort_df["first_buy_usd_amount"] >= minimum_initial_buy_usd]

    early_traders_cohort = cohort_df["trader_id"].dropna().unique().tolist()
    logger.info("[Cohort] traders in cohort (after filters): %d", len(early_traders_cohort))

    if len(early_traders_cohort) == 0:
        logger.info("No traders meeting conditions. Exiting.")
        return

    cohort_trades_df = trades_df[trades_df["trader_id"].isin(early_traders_cohort)].copy()
    cohort_trades_df = cohort_trades_df[cohort_trades_df["mint_address"].isin(tokens)].copy()

    # -------------------------
    # BUYS: normalized path
    # -------------------------
    buys = cohort_trades_df[cohort_trades_df["is_buy"] & cohort_trades_df["token_amount"].notna() & (cohort_trades_df["token_amount"] > 0)].copy()
    if not buys.empty:
        buys = attach_minute_price(buys, minute_price_df, mint_col="mint_address", time_col="block_time", price_col_name="minute_vwap", tolerance_minutes=5)
        cond_amount_usd = buys["amount_usd"].notna() & (buys["amount_usd"] > 0)
        cond_minute = buys["minute_vwap"].notna() & buys["token_amount"].notna()
        cond_derived = buys["derived_price"].notna() & buys["token_amount"].notna()
        buys["buy_usd_spent"] = 0.0
        buys.loc[cond_amount_usd, "buy_usd_spent"] = buys.loc[cond_amount_usd, "amount_usd"].astype(float)
        buys.loc[~cond_amount_usd & cond_minute, "buy_usd_spent"] = (
            buys.loc[~cond_amount_usd & cond_minute, "minute_vwap"].astype(float)
            * buys.loc[~cond_amount_usd & cond_minute, "token_amount"].astype(float)
        )
        buys.loc[~cond_amount_usd & ~cond_minute & cond_derived, "buy_usd_spent"] = (
            buys.loc[~cond_amount_usd & ~cond_minute & cond_derived, "derived_price"].astype(float)
            * buys.loc[~cond_amount_usd & ~cond_minute & cond_derived, "token_amount"].astype(float)
        )
    else:
        buys["buy_usd_spent"] = pd.Series(dtype=float)

    acquisition = buys.groupby(["trader_id", "mint_address"], as_index=False).agg(
        total_usd_spent=("buy_usd_spent", "sum"),
        total_tokens_bought=("token_amount", "sum")
    )
    acquisition["avg_buy_price_usd"] = acquisition["total_usd_spent"] / acquisition["total_tokens_bought"].replace({0: np.nan})

    # -------------------------
    # SELLS: normalized path
    # -------------------------
    sells = cohort_trades_df[~cohort_trades_df["is_buy"] & cohort_trades_df["token_amount"].notna() & (cohort_trades_df["token_amount"] > 0)].copy()
    if not sells.empty:
        sells = attach_minute_price(sells, minute_price_df, mint_col="mint_address", time_col="block_time", price_col_name="minute_vwap", tolerance_minutes=5)
        cond_amount_usd = sells["amount_usd"].notna() & (sells["amount_usd"] > 0)
        cond_minute = sells["minute_vwap"].notna() & sells["token_amount"].notna()
        cond_derived = sells["derived_price"].notna() & sells["token_amount"].notna()
        sells["sell_revenue_usd"] = 0.0
        sells.loc[cond_amount_usd, "sell_revenue_usd"] = sells.loc[cond_amount_usd, "amount_usd"].astype(float)
        sells.loc[~cond_amount_usd & cond_minute, "sell_revenue_usd"] = (
            sells.loc[~cond_amount_usd & cond_minute, "minute_vwap"].astype(float)
            * sells.loc[~cond_amount_usd & cond_minute, "token_amount"].astype(float)
        )
        sells.loc[~cond_amount_usd & ~cond_minute & cond_derived, "sell_revenue_usd"] = (
            sells.loc[~cond_amount_usd & ~cond_minute & cond_derived, "derived_price"].astype(float)
            * sells.loc[~cond_amount_usd & ~cond_minute & cond_derived, "token_amount"].astype(float)
        )
    else:
        sells["sell_revenue_usd"] = pd.Series(dtype=float)

    # Merge sells with acquisition on normalized mint key
    sells = pd.merge(
        sells,
        acquisition[["trader_id", "mint_address", "avg_buy_price_usd"]],
        on=["trader_id", "mint_address"],
        how="left"
    )
    sells["cost_of_goods_sold"] = sells["token_amount"] * sells["avg_buy_price_usd"].fillna(0.0)
    sells["realized_pnl_usd"] = sells["sell_revenue_usd"].fillna(0.0) - sells["cost_of_goods_sold"].fillna(0.0)

    sales_profit = sells.groupby(["trader_id", "mint_address"], as_index=False).agg(
        total_sales_revenue_usd=("sell_revenue_usd", "sum"),
        realized_profit_usd=("realized_pnl_usd", "sum")
    )

    # -------------------------
    # Combine metrics and fetch balances
    # -------------------------
    trader_token_metrics = pd.merge(
        acquisition.rename(columns={"mint_address": "mint_address"}),
        sales_profit,
        on=["trader_id", "mint_address"],
        how="left"
    ).fillna({"total_sales_revenue_usd": 0.0, "realized_profit_usd": 0.0})

    # token_mints_to_fetch = trader_token_metrics["mint_address"].dropna().unique().tolist()
    token_mints_to_fetch = tokens  # Only fetch balances for specified tokens
    balances_df = asyncio.run(get_current_balances_and_prices(list(early_traders_cohort), token_mints_to_fetch))

    if balances_df.empty:
        rows = []
        for t in early_traders_cohort:
            for m in token_mints_to_fetch:
                rows.append({"trader_id": t, "mint_address": m, "token_balance": 0.0, "current_price": 0.0, "current_value_usd": 0.0, "sol_balance": 0.0})
        balances_df = pd.DataFrame(rows)

    trader_token_metrics = pd.merge(
        trader_token_metrics,
        balances_df[["trader_id", "mint_address", "token_balance", "current_price", "current_value_usd"]],
        on=["trader_id", "mint_address"],
        how="left"
    ).fillna({"token_balance": 0.0, "current_price": 0.0, "current_value_usd": 0.0})

    # -------------------------
    # Unrealized / total PnL
    # -------------------------
    trader_token_metrics["avg_buy_price_usd"] = trader_token_metrics["avg_buy_price_usd"].fillna(0.0)
    trader_token_metrics["unrealized_cost_basis"] = trader_token_metrics["token_balance"] * trader_token_metrics["avg_buy_price_usd"]
    trader_token_metrics["estimated_unrealised_pnl"] = trader_token_metrics["current_value_usd"] - trader_token_metrics["unrealized_cost_basis"]
    trader_token_metrics["token_total_pnl"] = trader_token_metrics["realized_profit_usd"].fillna(0.0) + trader_token_metrics["estimated_unrealised_pnl"].fillna(0.0)
    trader_token_metrics["is_token_profitable"] = (trader_token_metrics["token_total_pnl"] > 0).astype(int)

    # -------------------------
    # Final summarization per trader
    # -------------------------
    final_summary = trader_token_metrics.groupby("trader_id", as_index=False).agg(
        overall_total_usd_spent=("total_usd_spent", "sum"),
        overall_total_sales_revenue=("total_sales_revenue_usd", "sum"),
        overall_realized_profit_usd=("realized_profit_usd", "sum"),
        overall_estimated_unrealised_pnl=("estimated_unrealised_pnl", "sum"),
        overall_current_value_of_holdings_usd=("current_value_usd", "sum"),
        overall_total_pnl=("token_total_pnl", "sum"),
        num_unique_tokens_traded=("mint_address", "nunique"),
        num_tokens_in_profit=("is_token_profitable", "sum")
    )

    final_summary["ROI"] = (final_summary["overall_total_pnl"] / final_summary["overall_total_usd_spent"]).replace([np.inf, np.nan], 0).fillna(0)
    final_summary["win_rate"] = (final_summary["num_tokens_in_profit"] / final_summary["num_unique_tokens_traded"]).fillna(0)

    trade_counts = cohort_trades_df.groupby("trader_id").size().reset_index(name="number_of_trades")
    final_summary = pd.merge(final_summary, trade_counts, on="trader_id", how="left").fillna(0)

    # final_summary = final_summary[final_summary["num_tokens_in_profit"] >= min_profitable_trades]
    if min_profitable_trades > 0:
        # Count profitable tokens only among the tokens we're analyzing
        specified_token_profits = trader_token_metrics[trader_token_metrics["mint_address"].isin(tokens)]
        profitable_by_trader = specified_token_profits.groupby("trader_id")["is_token_profitable"].sum()
        qualified_traders = profitable_by_trader[profitable_by_trader >= min_profitable_trades].index
        final_summary = final_summary[final_summary["trader_id"].isin(qualified_traders)]
    else:
        # If min_profitable_trades is 0, include all traders
        pass
    final_summary = final_summary.sort_values(by=["ROI", "overall_total_usd_spent"], ascending=[False, False]).reset_index(drop=True)

    display_cols = [
        "trader_id", "overall_current_value_of_holdings_usd", "overall_total_pnl", "overall_realized_profit_usd",
        "overall_estimated_unrealised_pnl", "ROI", "overall_total_usd_spent", "overall_total_sales_revenue",
        "num_unique_tokens_traded", "num_tokens_in_profit", "number_of_trades"
    ]
    for c in display_cols:
        if c not in final_summary.columns:
            final_summary[c] = 0.0

    logger.info("--- 🏆 Final Early Trader Profitability Report (top 50) ---")
    if final_summary.empty:
        logger.info("Final summary empty after filtering.")
    else:
        print(final_summary[display_cols].head(50).to_string(index=False))

    params = {
        "early_trading_window_hours": early_trading_window_hours,
        "minimum_initial_buy_usd": minimum_initial_buy_usd,
        "min_profitable_trades": min_profitable_trades
    }
    try:
        save_to_supabase(final_summary, tokens, params, job_id)
    except Exception as e:
        logger.warning("Failed to save/upload to Supabase: %s", e)

    return final_summary

# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solana Early Trader PNL Analysis (up to 3 tokens)")
    parser.add_argument("tokens", nargs="+", help="List of 1-3 token mint addresses (positional)")
    parser.add_argument("--window", type=int, default=None, help="Early trading window (hours) - optional")
    parser.add_argument("--minbuy", type=float, default=DEFAULT_MINIMUM_INITIAL_BUY_USD, help="Minimum initial buy in USD - optional")
    parser.add_argument("--minprof", type=int, default=DEFAULT_MIN_PROFITABLE_TRADES, help="Minimum number of profitable tokens")
    args = parser.parse_args()

    if len(args.tokens) > 3:
        parser.error("A maximum of 3 token addresses is allowed.")

    try:
        ensure_keys_present(require_supabase=False)
    except Exception as e:
        logger.error("Key validation failed: %s", e)
        raise

    run_pipeline(tokens=args.tokens,
                 early_trading_window_hours=args.window,
                 minimum_initial_buy_usd=args.minbuy,
                 min_profitable_trades=args.minprof)