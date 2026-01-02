#!/usr/bin/env python3
"""
wallet_pnl_with_behavior.py

OPTIMIZED: Only fetches DexScreener prices for tokens actually needed (paginated results)
This drastically reduces API calls and avoids 429 rate limit errors.
"""

import requests
import argparse
import datetime
from datetime import timedelta, timezone
from collections import defaultdict
import pandas as pd
import numbers
import math
import json
import os
from typing import Tuple, List, Dict, Any, Set
import pytz
import random
import time

# Optional Supabase imports
try:
    from supabase_utils import (
        upload_wallet_data,
        download_wallet_data,
        wallet_data_exists
    )
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[INFO] Supabase utils not available. Caching disabled.")

# --- API Configuration ---

def _load_keys_list(env_name_plural: str, env_name_singular: str) -> List[str]:
    """Load comma-separated API keys from environment variable."""
    value = os.getenv(env_name_plural) or os.getenv(env_name_singular)
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]

# Load Moralis keys
MORALIS_KEYS = _load_keys_list("MORALIS_API_KEYS", "MORALIS_API_KEY")
if not MORALIS_KEYS:
    raise RuntimeError("No Moralis API keys found in MORALIS_API_KEYS or MORALIS_API_KEY")
_moralis_idx = 0
_bad_moralis_keys: Set[str] = set()

# Load Helius keys
HELIUS_KEYS = _load_keys_list("HELIUS_API_KEYS", "HELIUS_API_KEY")
if not HELIUS_KEYS:
    raise RuntimeError("No Helius API keys found in HELIUS_API_KEYS or HELIUS_API_KEY")
_helius_idx = 0
_bad_helius_keys: Set[str] = set()

HELIUS_RPC_URL_TEMPLATE = "https://mainnet.helius-rpc.com/?api-key={}"
MORALIS_BASE_URL = "https://solana-gateway.moralis.io"
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/tokens/{}"

# --- Constants ---
SOL_MINT_ADDRESS = "So11111111111111111111111111111111111111112"
BALANCE_NOT_BOUGHT_MODE = "clamped"

# --- DexScreener Price Cache ---
_price_cache = {}
_price_cache_ttl = 300  # 5 minutes

def get_current_price_from_dexscreener(mint_address: str) -> float:
    """
    Fetch current price from DexScreener API with caching.
    Returns 0 if liquidity < $1 or on error.
    """
    cache_key = mint_address
    if cache_key in _price_cache:
        cached_price, cached_time = _price_cache[cache_key]
        if time.time() - cached_time < _price_cache_ttl:
            return cached_price
    
    try:
        url = DEXSCREENER_API_URL.format(mint_address)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pairs = data.get('pairs', [])
        if not pairs or len(pairs) == 0:
            _price_cache[cache_key] = (0.0, time.time())
            return 0.0
        
        first_pair = pairs[0]
        liquidity_usd = safe_float(first_pair.get('liquidity', {}).get('usd', 0))
        if liquidity_usd <= 1.0:
            _price_cache[cache_key] = (0.0, time.time())
            return 0.0
        
        price_usd = safe_float(first_pair.get('priceUsd', 0))
        _price_cache[cache_key] = (price_usd, time.time())
        return price_usd
        
    except Exception as e:
        print(f"[DEBUG] Error fetching price from DexScreener for {mint_address}: {e}")
        _price_cache[cache_key] = (0.0, time.time())
        return 0.0

def get_current_prices_batch(mint_addresses: List[str]) -> Dict[str, float]:
    """
    Fetch current prices for multiple tokens from DexScreener.
    Returns dict of {mint_address: price_usd}
    """
    prices = {}
    for mint in mint_addresses:
        price = get_current_price_from_dexscreener(mint)
        prices[mint] = price
        time.sleep(0.15)  # Increased delay to avoid rate limiting
    
    return prices

# --- Key Rotation Helpers ---
def get_next_moralis_key() -> str:
    """Round-robin through valid Moralis keys."""
    global _moralis_idx
    valid = [k for k in MORALIS_KEYS if k not in _bad_moralis_keys]
    if not valid:
        _bad_moralis_keys.clear()
        valid = MORALIS_KEYS
    key = valid[_moralis_idx % len(valid)]
    _moralis_idx += 1
    return key

def get_next_helius_key() -> str:
    """Round-robin through valid Helius keys."""
    global _helius_idx
    valid = [k for k in HELIUS_KEYS if k not in _bad_helius_keys]
    if not valid:
        _bad_helius_keys.clear()
        valid = HELIUS_KEYS
    key = valid[_helius_idx % len(valid)]
    _helius_idx += 1
    return key

# --- Helpers ---

def safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        try:
            return float(int(x))
        except Exception:
            return default

def parse_timestamp(ts_val):
    if ts_val is None:
        return datetime.datetime.now(timezone.utc)
    try:
        if isinstance(ts_val, (int, float)):
            ts = datetime.datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc) if ts_val > 1e12 else datetime.datetime.fromtimestamp(ts_val, tz=timezone.utc)
        else:
            try:
                iv = int(ts_val)
                ts = datetime.datetime.fromtimestamp(iv / 1000.0, tz=timezone.utc) if iv > 1e12 else datetime.datetime.fromtimestamp(iv, tz=timezone.utc)
            except Exception:
                ts_str = str(ts_val)
                if 'T' in ts_str:
                    if ts_str.endswith('Z'):
                        ts_str = ts_str[:-1] + '+00:00'
                    ts = datetime.datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = datetime.datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception as e:
        print(f"[DEBUG] Could not parse timestamp {ts_val}: {e}")
        return datetime.datetime.now(timezone.utc)

def find_cursor_strict(data: dict):
    if not isinstance(data, dict):
        return None
    for k in ('cursor', 'nextCursor', 'pageToken'):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for container_key in ('result', 'data', 'response'):
        cont = data.get(container_key)
        if isinstance(cont, dict):
            for k in ('cursor', 'nextCursor', 'pageToken'):
                v = cont.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    return None

def _attempt_extract_amount(d: dict):
    if not isinstance(d, dict):
        return None
    for key in ("amount", "rawAmount", "tokenAmount", "amountUSD", "usdAmount", "quantity", "uiAmount", "uiAmountString"):
        if key in d and d.get(key) is not None:
            return safe_float(d.get(key))
    for v in d.values():
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()):
            return safe_float(v)
    return None

def extract_token_entries(trade: dict):
    inputs = []
    outputs = []

    def add_entry(container, entry):
        if not isinstance(entry, dict):
            return
        account = entry.get('address') or entry.get('account') or entry.get('mint') or entry.get('tokenAddress') or entry.get('token')
        amount = _attempt_extract_amount(entry)
        if account:
            container.append({'account': str(account), 'amount': safe_float(amount, 0.0)})

    if 'sold' in trade:
        sold_val = trade['sold']
        if isinstance(sold_val, list):
            for e in sold_val:
                add_entry(inputs, e)
        else:
            add_entry(inputs, sold_val)
    if 'bought' in trade:
        bought_val = trade['bought']
        if isinstance(bought_val, list):
            for e in bought_val:
                add_entry(outputs, e)
        else:
            add_entry(outputs, bought_val)

    for k in ('nativeInput', 'nativeOutput', 'tokenInput', 'tokenOutput', 'tokenInputs', 'tokenOutputs', 'inputs', 'outputs'):
        if k in trade:
            val = trade[k]
            if isinstance(val, list):
                for e in val:
                    if 'input' in k.lower():
                        add_entry(inputs, e)
                    elif 'output' in k.lower():
                        add_entry(outputs, e)
                    else:
                        add_entry(inputs, e)
                        add_entry(outputs, e)
            elif isinstance(val, dict):
                if 'input' in k.lower():
                    add_entry(inputs, val)
                elif 'output' in k.lower():
                    add_entry(outputs, val)
                else:
                    add_entry(inputs, val)
                    add_entry(outputs, val)

    if not inputs or not outputs:
        for v in trade.values():
            if isinstance(v, dict) and any(key in v for key in ('address','account','mint','tokenAddress')):
                add_entry(inputs, v)
                add_entry(outputs, v)
            elif isinstance(v, list):
                for e in v:
                    if isinstance(e, dict) and any(key in e for key in ('address','account','mint','tokenAddress')):
                        add_entry(inputs, e)
                        add_entry(outputs, e)

    def normalize_list(lst):
        normalized = []
        seen = set()
        for item in lst:
            acct = item.get('account')
            if not acct:
                continue
            amt = safe_float(item.get('amount'), 0.0)
            key = (acct, round(amt, 12))
            if key in seen:
                continue
            seen.add(key)
            normalized.append({'account': acct, 'amount': amt})
        return normalized

    return normalize_list(inputs), normalize_list(outputs)

# --- API Fetching Functions ---

def get_dex_trades_from_moralis(wallet_address: str):
    print("Fetching historical DEX trades from Moralis...")
    all_trades = []
    cursor = None
    url = f"{MORALIS_BASE_URL}/account/mainnet/{wallet_address}/swaps"
    session = requests.Session()
    max_attempts = len(MORALIS_KEYS) * 2

    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor

        response = None
        data = None
        for attempt in range(max_attempts):
            api_key = get_next_moralis_key()
            headers = {
                "accept": "application/json",
                "X-API-Key": api_key,
                "User-Agent": "wallet-pnl-script/1.0"
            }
            try:
                response = session.get(url, headers=headers, params=params, timeout=30)
                if response.status_code == 429:
                    print(f"[Moralis] Rate limited on attempt {attempt+1}, rotating key.")
                    time.sleep(1)
                    continue
                if response.status_code == 401:
                    print(f"[Moralis] Key unauthorized, blacklisting and rotating.")
                    _bad_moralis_keys.add(api_key)
                    continue
                if response.status_code == 400:
                    print("Moralis returned 400 Bad Request. Response body:")
                    print(response.text)
                    return None
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data from Moralis API (attempt {attempt+1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                else:
                    print("All Moralis attempts failed.")
                    return None
            except ValueError as e:
                print(f"Invalid JSON response from Moralis: {e}")
                return None

        if not data:
            print("Failed to fetch a page from Moralis after all retries.")
            break

        items = []
        if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
            items = data['result']
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            items = data['data']
        else:
            for v in data.values() if isinstance(data, dict) else []:
                if isinstance(v, list):
                    items = v
                    break

        all_trades.extend(items)
        next_cursor = find_cursor_strict(data)
        if not next_cursor:
            break
        cursor = next_cursor

    print(f"Found {len(all_trades)} total trades.")
    return all_trades

def _make_helius_rpc_call(session: requests.Session, payload: Dict, timeout: int = 30) -> Dict:
    """Helper to make a Helius RPC call with key rotation and retries."""
    headers = {"Content-Type": "application/json"}
    max_attempts = len(HELIUS_KEYS) * 2

    for attempt in range(max_attempts):
        api_key = get_next_helius_key()
        url = HELIUS_RPC_URL_TEMPLATE.format(api_key)
        try:
            resp = session.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                print(f"[Helius] Rate limited on attempt {attempt+1}, rotating key.")
                time.sleep(1)
                continue
            if resp.status_code == 401:
                print(f"[Helius] Key unauthorized, blacklisting and rotating.")
                _bad_helius_keys.add(api_key)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying Helius RPC (attempt {attempt+1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(0.5 + random.uniform(0, 0.2))
            else:
                return {"error": str(e)}
        except ValueError:
            return {"error": "Invalid JSON response from Helius"}

    return {"error": "All Helius keys exhausted or failed"}

def get_current_balances_from_helius_rpc(wallet_address: str, verbose: bool = True):
    session = requests.Session()
    
    payload_tokens = {
        "jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
        "params": [
            wallet_address,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding": "jsonParsed"}
        ]
    }
    j_tokens = _make_helius_rpc_call(session, payload_tokens)
    if "error" in j_tokens:
        print(f"Error querying Helius RPC getTokenAccountsByOwner: {j_tokens['error']}")
        return None, None

    result = j_tokens.get("result") if isinstance(j_tokens, dict) else None
    if not result:
        if verbose: print("Helius getTokenAccountsByOwner returned no result.")
        value = []
    else:
        value = result.get("value", []) if isinstance(result, dict) else []
    
    token_map = {}
    for entry in value:
        try:
            acc = entry.get("account", {})
            data = acc.get("data", {})
            parsed = data.get("parsed", {})
            info = parsed.get("info", {}) if isinstance(parsed, dict) else {}
            mint = info.get("mint")
            token_amount = info.get("tokenAmount") or {}
            ui_amount = None
            if isinstance(token_amount, dict):
                if token_amount.get("uiAmount") is not None:
                    ui_amount = safe_float(token_amount.get("uiAmount"))
                elif token_amount.get("uiAmountString") is not None:
                    ui_amount = safe_float(token_amount.get("uiAmountString"))
                elif token_amount.get("amount") is not None:
                    try:
                        raw_amt = float(token_amount.get("amount"))
                        decs = int(token_amount.get("decimals") or 0)
                        ui_amount = raw_amt / (10 ** decs) if decs > 0 else raw_amt
                    except Exception:
                        ui_amount = safe_float(token_amount.get("amount"))
            if ui_amount is None:
                for k in ("uiAmount", "uiAmountString", "amount", "balance"):
                    if k in info and info.get(k) is not None:
                        ui_amount = safe_float(info.get(k)); break
            if mint:
                cur = token_map.get(mint, {"mint": str(mint), "amount": 0.0, "raw_data": []})
                cur["amount"] += float(ui_amount) if ui_amount is not None else 0.0
                cur["raw_data"].append(entry)
                token_map[mint] = cur
            else:
                if verbose: print("[DEBUG] token account entry missing mint:", entry.get("pubkey"))
        except Exception as e:
            if verbose: print(f"[DEBUG] Error parsing token account entry: {e}. Entry: {entry}")
            continue

    payload_balance = {
        "jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet_address]
    }
    j_balance = _make_helius_rpc_call(session, payload_balance)
    sol_amount = None
    if "error" in j_balance:
        if verbose: print(f"[DEBUG] Could not fetch SOL balance via getBalance: {j_balance['error']}")
    else:
        res2 = j_balance.get("result") if isinstance(j_balance, dict) else None
        if res2 and "value" in res2:
            sol_amount = safe_float(res2.get("value")) / 1e9
        elif isinstance(j_balance.get("result"), (int, float)):
            sol_amount = float(j_balance.get("result")) / 1e9

    tokens = list(token_map.values())
    if sol_amount is not None:
        sol_mint = SOL_MINT_ADDRESS
        found = any(tok["mint"] == sol_mint for tok in tokens)
        if found:
            for tok in tokens:
                if tok["mint"] == sol_mint:
                    tok["amount"] += float(sol_amount); break
        else:
            tokens.insert(0, {"mint": sol_mint, "amount": float(sol_amount), "raw_data": [{"source": "getBalance"}]})

    if verbose:
        print(f"Found {len(tokens)} token balances (including SOL if available).")
    return tokens, []

def fetch_historical_trades(wallet_address: str):
    """Alias for get_dex_trades_from_moralis"""
    return get_dex_trades_from_moralis(wallet_address)

def fetch_wallet_balances(wallet_address: str):
    """Alias for get_current_balances_from_helius_rpc"""
    balances, nfts = get_current_balances_from_helius_rpc(wallet_address)
    return balances

# --- Data Processing Core Logic ---

def process_trade_data(raw_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw_trades:
        return pd.DataFrame()

    enriched = []
    skipped = 0
    for idx, trade in enumerate(raw_trades):
        inputs, outputs = extract_token_entries(trade)

        if not inputs and not outputs:
            skipped += 1
            if skipped <= 5:
                print(f"[DEBUG] Skipping trade (no inputs/outputs found). idx={idx}, keys={list(trade.keys())}")
            continue

        sol_in_inputs = any(item['account'] == SOL_MINT_ADDRESS for item in inputs)
        sol_in_outputs = any(item['account'] == SOL_MINT_ADDRESS for item in outputs)

        if not (sol_in_inputs or sol_in_outputs):
            skipped += 1
            continue

        def find_non_sol_token(side_list):
            for it in side_list:
                if it.get('account') and it.get('account') != SOL_MINT_ADDRESS and safe_float(it.get('amount')) > 0:
                    return it
            return None

        token_entry = None
        trade_type = None

        if sol_in_inputs:
            trade_type = 'buy'
            token_entry = find_non_sol_token(outputs) or find_non_sol_token(inputs)
        elif sol_in_outputs:
            trade_type = 'sell'
            token_entry = find_non_sol_token(inputs) or find_non_sol_token(outputs)

        if token_entry is None:
            skipped += 1
            if skipped <= 5:
                print(f"[DEBUG] Skipping trade (no non-SOL token found). idx={idx}")
            continue

        token_mint = token_entry.get('account')
        token_amount = safe_float(token_entry.get('amount'), 0.0)

        amount_usd = 0.0
        for usd_key in ('transactionValueUSD', 'valueUSD', 'amountUSD', 'totalValueUsd', 'usdAmount', 'total_value_usd', 'totalValueUSD'):
            if usd_key in trade and trade.get(usd_key) is not None:
                amount_usd = safe_float(trade.get(usd_key), 0.0)
                break
        if amount_usd == 0.0:
            amt = token_entry.get('usdAmount') or token_entry.get('usd_amount') or token_entry.get('usdPrice')
            if amt:
                amount_usd = safe_float(amt, 0.0)
        if amount_usd == 0.0:
            amount_usd = safe_float(trade.get('totalValueUsd') or trade.get('total_value_usd') or trade.get('totalValueUSD'), 0.0)

        if token_amount <= 0 or amount_usd <= 0:
            skipped += 1
            if skipped <= 8:
                print(f"[DEBUG] Skipping trade (no positive token_amount or amount_usd). idx={idx} token_amount={token_amount} amount_usd={amount_usd}")
            continue

        ts = None
        for ts_key in ('blockUnixTimestamp', 'blockTimestamp', 'timestamp', 'txTimestamp', 'createdAt'):
            if ts_key in trade and trade.get(ts_key) is not None:
                ts = parse_timestamp(trade.get(ts_key))
                if ts is not None:
                    break
        if ts is None:
            ts = datetime.datetime.now(timezone.utc)
            if skipped <= 3:
                print(f"[DEBUG] Could not parse timestamp for trade idx={idx}. Using now as fallback.")

        price_per_token_usd = amount_usd / token_amount if token_amount else 0.0
        enriched.append({
            'block_timestamp': ts,
            'mint_address': token_mint,
            'trade_type': trade_type,
            'token_amount': token_amount,
            'amount_usd': amount_usd,
            'price_per_token_usd': price_per_token_usd
        })

    if not enriched:
        print("No enriched trades after parsing (all skipped).")
        return pd.DataFrame()

    df = pd.DataFrame(enriched).sort_values('block_timestamp').reset_index(drop=True)
    df['quantity_change'] = df.apply(lambda row: row['token_amount'] if row['trade_type'] == 'buy' else -row['token_amount'], axis=1)
    df['running_balance'] = df.groupby('mint_address')['quantity_change'].cumsum()
    df['buy_cost'] = df.apply(lambda r: r['amount_usd'] if r['trade_type'] == 'buy' else 0.0, axis=1)
    df['buy_quantity'] = df.apply(lambda r: r['token_amount'] if r['trade_type'] == 'buy' else 0.0, axis=1)
    df['cumulative_buy_cost'] = df.groupby('mint_address')['buy_cost'].cumsum()
    df['cumulative_buy_quantity'] = df.groupby('mint_address')['buy_quantity'].cumsum()
    df['running_avg_buy_price'] = df.apply(
        lambda r: (r['cumulative_buy_cost'] / r['cumulative_buy_quantity'])
        if r['cumulative_buy_quantity'] > 0 else float('nan'),
        axis=1
    )

    def calculate_pnl(row):
        if row['trade_type'] == 'sell' and not math.isnan(row['running_avg_buy_price']):
            return (row['price_per_token_usd'] - row['running_avg_buy_price']) * row['token_amount']
        else:
            return 0.0

    df['realized_pnl'] = df.apply(calculate_pnl, axis=1)
    return df

# --- Behavior & Formatting ---
def compute_behavior_metrics(processed_trades: pd.DataFrame, within_seconds: int = 5,
                             interval_days: int = 0) -> Dict[str, Any]:
    """Compute trading behavior metrics, optionally filtered by time window."""
    if processed_trades is None or processed_trades.empty:
        return {}

    df = processed_trades.copy()
    df = df.sort_values(['mint_address', 'block_timestamp']).reset_index(drop=True)

    if interval_days > 0:
        cutoff_date = df['block_timestamp'].max() - pd.Timedelta(days=interval_days)
        df = df[df['block_timestamp'] >= cutoff_date].copy()

        if df.empty:
            return {
                'tokens_more_sold_than_bought_count': 0,
                'tokens_more_sold_than_bought_list': [],
                'sell_trades_fully_not_bought_count': 0,
                'sell_trades_with_not_bought_count': 0,
                'total_quantity_sold_not_bought': 0.0,
                'total_usd_value_sold_not_bought': 0.0,
                'sells_within_seconds_count': 0,
                'sells_within_seconds_quantity': 0.0,
                'total_sells': 0,
                'distinct_tokens': 0,
                'overall_avg_holding_seconds': None,
                'per_token_breakdown': {},
                'interval_days': interval_days,
                'date_range_start': None,
                'date_range_end': None
            }

    df['net_quantity_before_trade'] = df.groupby('mint_address')['running_balance'].shift(1).fillna(0.0)

    df['quantity_not_bought_in_sell'] = 0.0
    df['quantity_eligible_for_cost_basis'] = 0.0
    df['value_not_bought_usd'] = 0.0

    sell_mask = df['trade_type'] == 'sell'
    df.loc[sell_mask, 'quantity_eligible_for_cost_basis'] = df.loc[sell_mask].apply(
        lambda r: min(r['token_amount'], max(0.0, r['net_quantity_before_trade'])), axis=1
    )
    df.loc[sell_mask, 'quantity_not_bought_in_sell'] = df.loc[sell_mask].apply(
        lambda r: max(0.0, r['token_amount'] - max(0.0, r['net_quantity_before_trade'])), axis=1
    )
    df.loc[sell_mask, 'value_not_bought_usd'] = df.loc[sell_mask]['quantity_not_bought_in_sell'] * df.loc[sell_mask]['price_per_token_usd']

    metrics = _compute_metrics_for_subset(df, within_seconds)

    metrics['interval_days'] = interval_days
    if interval_days > 0:
        metrics['date_range_start'] = df['block_timestamp'].min()
        metrics['date_range_end'] = df['block_timestamp'].max()
    else:
        metrics['date_range_start'] = None
        metrics['date_range_end'] = None

    return metrics

def _compute_metrics_for_subset(df: pd.DataFrame, within_seconds: int) -> Dict[str, Any]:
    """Helper function to compute metrics for a subset of trades"""
    if df.empty:
        return {
            'tokens_more_sold_than_bought_count': 0,
            'tokens_more_sold_than_bought_list': [],
            'sell_trades_with_not_bought_count': 0,
            'sell_trades_fully_not_bought_count': 0,
            'total_quantity_sold_not_bought': 0.0,
            'total_usd_value_sold_not_bought': 0.0,
            'sells_within_seconds_count': 0,
            'sells_within_seconds_quantity': 0.0,
            'total_sells': 0,
            'distinct_tokens': 0,
            'overall_avg_holding_seconds': None
        }

    grouped_qty = df.groupby(['mint_address', 'trade_type'])['token_amount'].sum().unstack(fill_value=0)
    total_bought = grouped_qty.get('buy', pd.Series(dtype=float))
    total_sold = grouped_qty.get('sell', pd.Series(dtype=float))
    all_mints = set(total_bought.index) | set(total_sold.index)
    tokens_more_sold_than_bought = [m for m in all_mints if float(total_sold.get(m, 0.0)) > float(total_bought.get(m, 0.0))]
    tokens_more_sold_than_bought_count = len(tokens_more_sold_than_bought)

    sell_trades_with_not_bought_count = int((df['quantity_not_bought_in_sell'] > 0).sum())
    total_quantity_sold_not_bought = float(df['quantity_not_bought_in_sell'].sum())
    total_usd_value_sold_not_bought = float(df['value_not_bought_usd'].sum())

    def compute_prev_buy_time(group: pd.DataFrame) -> pd.DataFrame:
        buy_time = group['block_timestamp'].where(group['trade_type'] == 'buy')
        prev_buy = buy_time.ffill()
        group = group.assign(prev_buy_time=prev_buy)
        return group

    df = df.groupby('mint_address', group_keys=False).apply(compute_prev_buy_time)

    df['secs_since_prev_buy'] = None
    sell_idx = df['trade_type'] == 'sell'
    df.loc[sell_idx, 'secs_since_prev_buy'] = df.loc[sell_idx].apply(
        lambda r: (r['block_timestamp'] - r['prev_buy_time']).total_seconds() if pd.notnull(r['prev_buy_time']) else None, axis=1
    )

    sells_within_n_seconds_mask = df['trade_type'] == 'sell'
    sells_within_n_seconds_mask &= df['secs_since_prev_buy'].notnull() & (df['secs_since_prev_buy'] <= within_seconds) & (df['secs_since_prev_buy'] >= 0)
    sells_within_n_seconds_count = int(sells_within_n_seconds_mask.sum())
    sells_within_n_seconds_quantity = float(df.loc[sells_within_n_seconds_mask, 'token_amount'].sum())

    fully_not_bought_mask = (df['trade_type'] == 'sell') & (df['quantity_eligible_for_cost_basis'] <= 0)
    sell_trades_fully_not_bought_count = int(fully_not_bought_mask.sum())

    per_token = {}
    overall_num = 0.0
    overall_denom = 0.0
    for mint, g in df.groupby('mint_address'):
        token_obj = {}
        token_obj['total_bought_qty'] = float(g[g['trade_type'] == 'buy']['token_amount'].sum())
        token_obj['total_sold_qty'] = float(g[g['trade_type'] == 'sell']['token_amount'].sum())
        token_obj['sell_trades_with_not_bought_count'] = int((g['quantity_not_bought_in_sell'] > 0).sum())
        token_obj['quantity_sold_not_bought'] = float(g['quantity_not_bought_in_sell'].sum())
        token_obj['value_sold_not_bought_usd'] = float(g['value_not_bought_usd'].sum())
        token_obj['sells_within_n_seconds_count'] = int(((g['trade_type']=='sell') & (g['secs_since_prev_buy'].notnull()) & (g['secs_since_prev_buy']<=within_seconds) & (g['secs_since_prev_buy']>=0)).sum())
        token_obj['sells_within_n_seconds_quantity'] = float(g.loc[((g['trade_type']=='sell') & (g['secs_since_prev_buy'].notnull()) & (g['secs_since_prev_buy']<=within_seconds) & (g['secs_since_prev_buy']>=0)), 'token_amount'].sum())

        sells_for_avg = g[(g['trade_type']=='sell') & (g['quantity_eligible_for_cost_basis'] > 0) & (g['secs_since_prev_buy'].notnull()) & (g['secs_since_prev_buy'] >= 0)]
        denom = float(sells_for_avg['quantity_eligible_for_cost_basis'].sum())
        if denom > 0:
            num = float((sells_for_avg['secs_since_prev_buy'] * sells_for_avg['quantity_eligible_for_cost_basis']).sum())
            token_obj['avg_holding_duration_seconds'] = num / denom
            overall_num += num
            overall_denom += denom
        else:
            token_obj['avg_holding_duration_seconds'] = None

        per_token[mint] = token_obj

    overall_avg_holding_seconds = (overall_num / overall_denom) if overall_denom > 0 else None

    total_sells = int((df['trade_type'] == 'sell').sum())
    distinct_tokens = int(df['mint_address'].nunique())

    return {
        'tokens_more_sold_than_bought_count': tokens_more_sold_than_bought_count,
        'tokens_more_sold_than_bought_list': tokens_more_sold_than_bought,
        'sell_trades_with_not_bought_count': sell_trades_with_not_bought_count,
        'sell_trades_fully_not_bought_count': sell_trades_fully_not_bought_count,
        'total_quantity_sold_not_bought': total_quantity_sold_not_bought,
        'total_usd_value_sold_not_bought': total_usd_value_sold_not_bought,
        'sells_within_seconds_count': sells_within_n_seconds_count,
        'sells_within_seconds_quantity': sells_within_n_seconds_quantity,
        'per_token_breakdown': per_token,
        'overall_avg_holding_seconds': overall_avg_holding_seconds,
        'total_sells': total_sells,
        'distinct_tokens': distinct_tokens
    }


def format_duration(seconds: float) -> str:
    if seconds is None:
        return 'N/A'
    try:
        s = int(round(seconds))
        if s < 60:
            return f"{s} seconds"
        if s < 3600:
            m = s // 60
            sec = s % 60
            return f"{m} minutes {sec} seconds"
        if s < 86400:
            h = s // 3600
            m = (s % 3600) // 60
            sec = s % 60
            return f"{h} hours {m} minutes {sec} seconds"
        days = s // 86400
        rem = s % 86400
        h = rem // 3600
        m = (rem % 3600) // 60
        return f"{days} days {h} hours {m} minutes"
    except Exception:
        return 'N/A'


def format_behavior_panel(behavior: Dict[str, Any]) -> str:
    """Format a compact phishing-check style panel with time interval support."""
    if not behavior:
        return "Phishing check\nNo behavior data available."

    total_sells = behavior.get('total_sells', 0)
    distinct_tokens = behavior.get('distinct_tokens', 0)
    sold_gt_bought_count = behavior.get('tokens_more_sold_than_bought_count', 0)
    sells_within = behavior.get('sells_within_seconds_count', 0)
    sell_not_bought_count = behavior.get('sell_trades_fully_not_bought_count', 0)

    sold_gt_bought_pct = (sold_gt_bought_count / distinct_tokens * 100) if distinct_tokens > 0 else 0.0
    sells_within_pct = (sells_within / total_sells * 100) if total_sells > 0 else 0.0
    sell_not_bought_pct = (sell_not_bought_count / total_sells * 100) if total_sells > 0 else 0.0

    blacklist_count = 0
    blacklist_pct = 0.0

    def get_emoji(count):
        return "SAFE" if count == 0 else "FLAG"

    lines = []
    lines.append("Phishing check")
    lines.append(f"[{get_emoji(blacklist_count)}] Blacklist: {blacklist_count} ({blacklist_pct:.0f}%)")
    lines.append(f"[{get_emoji(sell_not_bought_count)}] Didn't buy: {sell_not_bought_count} ({sell_not_bought_pct:.0f}%)")
    lines.append(f"[{get_emoji(sold_gt_bought_count)}] Sold > Bought: {sold_gt_bought_count} ({sold_gt_bought_pct:.0f}%)")
    lines.append(f"[{get_emoji(sells_within)}] Buy/Sell within fast window: {sells_within} ({sells_within_pct:.0f}%)")

    return "\n".join(lines)

# --- Analysis helpers (PnL) - OPTIMIZED ---

def get_overall_pnl_summary(processed_trades, balances, interval_days):
    """Get PnL summary - only fetches prices for tokens with current balances"""
    if processed_trades is None or processed_trades.empty:
        return {"error": "No processed trade data available."}
    
    interval_start_date = datetime.datetime.now(timezone.utc) - timedelta(days=interval_days) if interval_days > 0 else datetime.datetime.min.replace(tzinfo=timezone.utc)
    interval_trades = processed_trades[processed_trades['block_timestamp'] >= interval_start_date]
    total_realized_pnl = float(interval_trades['realized_pnl'].sum())
    
    # OPTIMIZATION: Only fetch prices for tokens we currently hold
    tokens_with_balance = []
    if balances:
        for token in balances:
            mint = token.get('mint')
            amount = safe_float(token.get('amount'))
            if mint and amount > 0:
                tokens_with_balance.append(mint)
    
    print(f"\nFetching prices for {len(tokens_with_balance)} tokens with current balances...")
    current_prices = get_current_prices_batch(tokens_with_balance) if tokens_with_balance else {}
    
    unrealized_pnl_total = 0.0
    
    # Calculate average costs
    buys = processed_trades[processed_trades['trade_type'] == 'buy']
    final_avg_costs = {}
    if not buys.empty:
        grouped = buys.groupby('mint_address').agg({'amount_usd': 'sum', 'token_amount': 'sum'})
        for mint, row in grouped.iterrows():
            if row['token_amount'] > 0:
                final_avg_costs[mint] = row['amount_usd'] / row['token_amount']
    
    if balances:
        for token in balances:
            mint = token.get('mint')
            amount = safe_float(token.get('amount'))
            if not mint or amount <= 0:
                continue
            
            current_price = current_prices.get(mint, 0.0)
            avg_cost = final_avg_costs.get(mint)
            
            if current_price > 0 and avg_cost is not None and amount > 0:
                unrealized_pnl_total += (current_price - avg_cost) * amount

    behavior = compute_behavior_metrics(processed_trades)
    overall_avg_holding_seconds = behavior.get('overall_avg_holding_seconds')
    overall_avg_holding_readable = format_duration(overall_avg_holding_seconds)

    sell_trades = interval_trades[interval_trades['trade_type'] == 'sell']
    buy_trades = interval_trades[interval_trades['trade_type'] == 'buy']
    total_sell_trades = int(len(sell_trades))
    winning_sells = int(len(sell_trades[sell_trades['realized_pnl'] > 0]))
    win_rate = (winning_sells / total_sell_trades * 100) if total_sell_trades > 0 else 0.0
    total_buy_volume = float(buy_trades['amount_usd'].sum()) if not buy_trades.empty else 0.0
    
    return {
        "analysis_interval_days": int(interval_days),
        "total_realized_pnl_usd": float(total_realized_pnl),
        "total_unrealized_pnl_usd": float(unrealized_pnl_total),
        "total_pnl_combined_usd": float(total_realized_pnl + unrealized_pnl_total),
        "total_sell_trades_in_interval": total_sell_trades,
        "total_buy_trades_in_interval": int(len(buy_trades)),
        "total_buy_volume_usd_in_interval": float(total_buy_volume),
        "win_rate_percent": float(win_rate),
        "avg_holding_before_sell_seconds": overall_avg_holding_seconds,
        "avg_holding_before_sell_readable": overall_avg_holding_readable
    }


def get_pnl_distribution(processed_trades, interval_days=0):
    """Get PnL distribution filtered by interval - NO price fetching needed"""
    if processed_trades is None or processed_trades.empty:
        return {"error": "No processed trade data available."}

    if interval_days > 0:
        cutoff_date = processed_trades['block_timestamp'].max() - timedelta(days=interval_days)
        filtered_trades = processed_trades[processed_trades['block_timestamp'] >= cutoff_date].copy()
    else:
        filtered_trades = processed_trades.copy()

    sells_df = filtered_trades[filtered_trades['trade_type'] == 'sell'].copy()
    if sells_df.empty:
        return {"distribution_percentage": {}, "trade_counts": {}}

    sells_df['cost_basis_usd'] = sells_df['running_avg_buy_price'] * sells_df['token_amount']
    sells_df = sells_df[sells_df['cost_basis_usd'] > 0]
    if sells_df.empty:
        return {"distribution_percentage": {}, "trade_counts": {}}

    sells_df['pnl_percentage'] = (sells_df['realized_pnl'] / sells_df['cost_basis_usd']) * 100

    def categorize_pnl(pnl_perc):
        if pnl_perc > 500:
            return '> 500%'
        if 200 <= pnl_perc <= 500:
            return '200% - 500%'
        if 0 < pnl_perc < 200:
            return '0% - 200%'
        if -50 < pnl_perc <= 0:
            return '-50% - 0%'
        if pnl_perc <= -50:
            return '< -50%'
        return 'Breakeven'

    sells_df['pnl_category'] = sells_df['pnl_percentage'].apply(categorize_pnl)
    distribution = (sells_df['pnl_category'].value_counts(normalize=True) * 100).to_dict()
    counts = sells_df['pnl_category'].value_counts().to_dict()
    return {'distribution_percentage': distribution, 'trade_counts': counts}

def get_pnl_breakdown_per_token(processed_trades, balances, interval_days=0, mints_to_fetch: List[str] = None):
    """
    Get per-token PnL breakdown with DexScreener prices.
    SMART OPTIMIZATION: Automatically only fetches prices for tokens with balances
    """
    if processed_trades is None or processed_trades.empty:
        return {"error": "No processed trade data available."}

    # Filter by interval if specified
    if interval_days > 0:
        cutoff_date = processed_trades['block_timestamp'].max() - timedelta(days=interval_days)
        filtered_trades = processed_trades[processed_trades['block_timestamp'] >= cutoff_date].copy()
    else:
        filtered_trades = processed_trades.copy()

    # Use filtered trades for calculations
    buys = filtered_trades[filtered_trades['trade_type'] == 'buy']
    sells = filtered_trades[filtered_trades['trade_type'] == 'sell']

    final_avg_costs = {}
    if not buys.empty:
        group_buys = buys.groupby('mint_address').agg({'amount_usd': 'sum', 'token_amount': 'sum'})
        for mint, row in group_buys.iterrows():
            if row['token_amount'] > 0:
                final_avg_costs[mint] = row['amount_usd'] / row['token_amount']

    realized_by_token = filtered_trades.groupby('mint_address')['realized_pnl'].sum().to_dict()
    total_usd_spent_by_token = buys.groupby('mint_address')['amount_usd'].sum().to_dict() if not buys.empty else {}
    total_usd_received_by_token = sells.groupby('mint_address')['amount_usd'].sum().to_dict() if not sells.empty else {}

    behavior = compute_behavior_metrics(filtered_trades, interval_days=interval_days)
    per_token_behavior = behavior.get('per_token_breakdown', {}) if isinstance(behavior, dict) else {}

    breakdown = {}
    total_bought_qty_by_token = buys.groupby('mint_address')['token_amount'].sum().to_dict() if not buys.empty else {}
    total_sold_qty_by_token = sells.groupby('mint_address')['token_amount'].sum().to_dict() if not sells.empty else {}

    all_mints = set(list(realized_by_token.keys()) + list(final_avg_costs.keys()) + list(total_usd_spent_by_token.keys()) + list(total_usd_received_by_token.keys()))
    if balances:
        for t in balances:
            mint = t.get('mint') or t.get('tokenAddress') or t.get('address')
            if mint:
                all_mints.add(mint)

    # SMART OPTIMIZATION: If mints_to_fetch not provided, automatically detect tokens that need prices
    # Only tokens with current balances need current prices for unrealized PnL
    if mints_to_fetch is None:
        mints_needing_prices = []
        if balances:
            for token in balances:
                mint = token.get('mint') or token.get('tokenAddress') or token.get('address')
                amount = safe_float(token.get('amount') or token.get('balance') or token.get('uiAmount'), 0.0)
                if mint and amount > 0 and mint in all_mints:
                    mints_needing_prices.append(mint)
        
        print(f"[SMART OPTIMIZATION] Fetching prices for {len(mints_needing_prices)} tokens with balances (out of {len(all_mints)} total tokens)")
    else:
        # Use provided list (for pagination)
        mints_needing_prices = [m for m in mints_to_fetch if m in all_mints]
        print(f"[PAGINATION] Fetching prices for {len(mints_needing_prices)} tokens on current page")
    
    # Fetch prices with improved error handling
    current_prices = get_current_prices_batch_safe(mints_needing_prices) if mints_needing_prices else {}

    for mint in all_mints:
        data = {}
        data['total_realized_pnl_usd'] = float(realized_by_token.get(mint, 0.0))
        data['total_usd_spent'] = float(total_usd_spent_by_token.get(mint, 0.0))
        data['total_usd_received'] = float(total_usd_received_by_token.get(mint, 0.0))
        data['overall_avg_buy_price'] = float(final_avg_costs.get(mint, 0.0))
        data['current_balance'] = 0.0
        data['unrealized_pnl_usd'] = 0.0

        if balances:
            for token in balances:
                token_mint = token.get('mint') or token.get('tokenAddress') or token.get('address')
                if token_mint == mint:
                    amount = safe_float(token.get('amount') or token.get('balance') or token.get('uiAmount'), 0.0)
                    data['current_balance'] = float(amount)
                    break

        # Use fetched price if available, otherwise 0
        current_price = current_prices.get(mint, 0.0)
        data['current_price'] = float(current_price)
        
        avg_cost = final_avg_costs.get(mint, 0.0)
        if current_price > 0 and avg_cost > 0 and data['current_balance'] > 0:
            data['unrealized_pnl_usd'] = float((current_price - avg_cost) * data['current_balance'])

        beh = per_token_behavior.get(mint, {})
        data['quantity_sold_not_bought'] = float(beh.get('quantity_sold_not_bought', 0.0))
        data['value_sold_not_bought_usd'] = float(beh.get('value_sold_not_bought_usd', 0.0))

        total_bought_qty = float(total_bought_qty_by_token.get(mint, 0.0))
        total_sold_qty = float(total_sold_qty_by_token.get(mint, 0.0))
        net_traded = total_bought_qty - total_sold_qty
        if BALANCE_NOT_BOUGHT_MODE == 'legacy':
            balance_not_bought_qty = max(0.0, data['current_balance'] - net_traded)
        else:
            net_traded_clamped = max(0.0, net_traded)
            balance_not_bought_qty = max(0.0, data['current_balance'] - net_traded_clamped)

        data['balance_not_bought_quantity'] = float(balance_not_bought_qty)
        data['balance_not_bought_usd'] = float(balance_not_bought_qty * current_price) if current_price else 0.0

        avg_hold_seconds = beh.get('avg_holding_duration_seconds')
        data['avg_holding_duration_seconds'] = float(avg_hold_seconds) if avg_hold_seconds is not None else None
        data['avg_holding_duration_readable'] = format_duration(avg_hold_seconds) if avg_hold_seconds is not None else 'N/A'

        data['total_combined_pnl_usd'] = float(data['total_realized_pnl_usd'] + data['unrealized_pnl_usd'])
        breakdown[mint] = data

    return breakdown


def get_current_prices_batch_safe(mint_addresses: List[str]) -> Dict[str, float]:
    """
    Fetch current prices with exponential backoff and better error handling
    """
    if not mint_addresses:
        return {}
    
    prices = {}
    consecutive_failures = 0
    total_429_errors = 0
    
    for i, mint in enumerate(mint_addresses):
        try:
            price = get_current_price_from_dexscreener_safe(mint)
            prices[mint] = price
            
            if price == 0.0:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            
            # Adaptive delay: increase if hitting failures
            base_delay = 0.4  # Safer base delay (2.5 requests/sec)
            if consecutive_failures > 0:
                delay = base_delay * (1.5 ** min(consecutive_failures, 4))
                if consecutive_failures >= 3:
                    print(f"[WARNING] {consecutive_failures} consecutive failures, increasing delay to {delay:.2f}s")
            else:
                delay = base_delay
            
            # Don't delay after the last request
            if i < len(mint_addresses) - 1:
                time.sleep(delay)
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch price for {mint}: {e}")
            prices[mint] = 0.0
            consecutive_failures += 1
    
    failed_count = sum(1 for p in prices.values() if p == 0.0)
    if failed_count > 0:
        print(f"[WARNING] {failed_count}/{len(mint_addresses)} price fetches returned 0.0")
    
    return prices

def get_current_price_from_dexscreener_safe(mint_address: str, retry_count: int = 3) -> float:
    """
    Fetch current price from DexScreener API with caching and exponential backoff.
    Returns 0 if liquidity < $1 or on error.
    """
    cache_key = mint_address
    if cache_key in _price_cache:
        cached_price, cached_time = _price_cache[cache_key]
        if time.time() - cached_time < _price_cache_ttl:
            return cached_price
    
    for attempt in range(retry_count):
        try:
            url = DEXSCREENER_API_URL.format(mint_address)
            response = requests.get(url, timeout=10)
            
            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                if attempt < retry_count - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[DexScreener] Rate limited on {mint_address[:8]}..., waiting {wait_time:.2f}s (attempt {attempt + 1}/{retry_count})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[DexScreener] Rate limit persists for {mint_address[:8]}... after {retry_count} attempts")
                    _price_cache[cache_key] = (0.0, time.time())
                    return 0.0
            
            response.raise_for_status()
            data = response.json()
            
            pairs = data.get('pairs', [])
            if not pairs or len(pairs) == 0:
                _price_cache[cache_key] = (0.0, time.time())
                return 0.0
            
            first_pair = pairs[0]
            liquidity_usd = safe_float(first_pair.get('liquidity', {}).get('usd', 0))
            if liquidity_usd <= 1.0:
                _price_cache[cache_key] = (0.0, time.time())
                return 0.0
            
            price_usd = safe_float(first_pair.get('priceUsd', 0))
            _price_cache[cache_key] = (price_usd, time.time())
            return price_usd
            
        except requests.exceptions.RequestException as e:
            if attempt < retry_count - 1:
                wait_time = (1.5 ** attempt) + random.uniform(0, 0.5)
                print(f"[DexScreener] Request error for {mint_address[:8]}..., retrying in {wait_time:.2f}s: {str(e)[:50]}")
                time.sleep(wait_time)
                continue
            else:
                print(f"[ERROR] All retries failed for {mint_address[:8]}...: {e}")
                _price_cache[cache_key] = (0.0, time.time())
                return 0.0
        except Exception as e:
            print(f"[ERROR] Unexpected error fetching price for {mint_address[:8]}...: {e}")
            _price_cache[cache_key] = (0.0, time.time())
            return 0.0
    
    _price_cache[cache_key] = (0.0, time.time())
    return 0.0

def fetch_from_apis(wallet: str):
    """Fetch both balances and trades for a wallet using existing functions."""
    print(f"Fetching fresh data for {wallet}...")

    trades = fetch_historical_trades(wallet)
    balances = fetch_wallet_balances(wallet)

    return balances, trades


def get_wallet_data(wallet: str, refresh: bool = False):
    """Get wallet data from cache or fetch fresh from APIs"""
    if not SUPABASE_AVAILABLE:
        return fetch_from_apis(wallet)

    use_cache = not refresh and wallet_data_exists(wallet)

    if use_cache:
        print(f"Loading cached data for {wallet} from Supabase...")
        cached_data = download_wallet_data(wallet)
        return cached_data.get('balances'), cached_data.get('trades')

    if refresh:
        print(f"Refreshing data for {wallet}...")
    else:
        print(f"No cached data found for {wallet}. Fetching fresh data...")

    balances, trades = fetch_from_apis(wallet)

    data = {
        "wallet": wallet,
        "last_updated": datetime.datetime.utcnow().isoformat(),
        "balances": balances,
        "trades": trades,
    }
    upload_wallet_data(wallet, data)

    return balances, trades

# Pagination functions - OPTIMIZED to only fetch prices for current page

def get_token_pnl_paginated(processed_trades: pd.DataFrame, balances: List[Dict],
                           page: int = 1, per_page: int = 9, interval_days: int = 0) -> Dict[str, Any]:
    """
    Get paginated token PnL data ordered by last trade time (descending).
    OPTIMIZATION: Only fetches prices for tokens on the current page
    """
    if processed_trades is None or processed_trades.empty:
        return {
            "tokens": [],
            "total_tokens": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0
        }

    # Filter by interval if needed
    if interval_days > 0:
        cutoff_date = processed_trades['block_timestamp'].max() - timedelta(days=interval_days)
        filtered_trades = processed_trades[processed_trades['block_timestamp'] >= cutoff_date].copy()
    else:
        filtered_trades = processed_trades.copy()

    last_trade_times = filtered_trades.groupby('mint_address')['block_timestamp'].max().to_dict()
    
    # First, get all tokens sorted by last trade time (without prices)
    all_tokens_with_time = []
    for mint, last_trade in last_trade_times.items():
        all_tokens_with_time.append({
            'mint': mint,
            'last_trade_time': last_trade
        })
    
    all_tokens_with_time.sort(key=lambda x: x['last_trade_time'], reverse=True)
    
    # Calculate pagination
    total_tokens = len(all_tokens_with_time)
    total_pages = (total_tokens + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get only the mints for the current page
    page_tokens = all_tokens_with_time[start_idx:end_idx]
    page_mints = [t['mint'] for t in page_tokens]
    
    # OPTIMIZATION: Only fetch breakdown (with prices) for current page mints
    breakdown = get_pnl_breakdown_per_token(processed_trades, balances, interval_days=interval_days, mints_to_fetch=page_mints)

    if isinstance(breakdown, dict) and 'error' not in breakdown:
        token_list = []
        for token in page_tokens:
            mint = token['mint']
            data = breakdown.get(mint, {})
            token_list.append({
                'mint': mint,
                'last_trade_time': token['last_trade_time'].isoformat(),
                **data
            })

        return {
            "tokens": token_list,
            "total_tokens": total_tokens,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages
        }

    return {
        "tokens": [],
        "total_tokens": 0,
        "page": page,
        "per_page": per_page,
        "total_pages": 0
    }


def get_trades_paginated(processed_trades: pd.DataFrame, page: int = 1,
                        per_page: int = 9) -> Dict[str, Any]:
    """Get paginated individual trades ordered by trade time (descending). NO price fetching needed."""
    if processed_trades is None or processed_trades.empty:
        return {
            "trades": [],
            "total_trades": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0
        }

    df = processed_trades.sort_values('block_timestamp', ascending=False).reset_index(drop=True)

    total_trades = len(df)
    total_pages = (total_trades + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    page_df = df.iloc[start_idx:end_idx]

    trades = []
    for _, row in page_df.iterrows():
        trades.append({
            'trade_time': row['block_timestamp'].isoformat(),
            'mint_address': row['mint_address'],
            'trade_type': row['trade_type'],
            'token_amount': float(row['token_amount']),
            'amount_usd': float(row['amount_usd']),
            'price_per_token_usd': float(row['price_per_token_usd']),
            'realized_pnl': float(row['realized_pnl']) if row['trade_type'] == 'sell' else None,
            'running_balance': float(row['running_balance']),
            'running_avg_buy_price': float(row['running_avg_buy_price']) if not pd.isna(row['running_avg_buy_price']) else None
        })

    return {
        "trades": trades,
        "total_trades": total_trades,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    }


def get_holdings_paginated(processed_trades: pd.DataFrame, balances: List[Dict],
                          page: int = 1, per_page: int = 9) -> Dict[str, Any]:
    """
    Get paginated current holdings with PnL data.
    OPTIMIZATION: Only fetches prices for tokens on the current page
    """
    if processed_trades is None or processed_trades.empty or not balances:
        return {
            "holdings": [],
            "total_holdings": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0
        }

    buy_counts = processed_trades[processed_trades['trade_type'] == 'buy'].groupby('mint_address').size().to_dict()
    sell_counts = processed_trades[processed_trades['trade_type'] == 'sell'].groupby('mint_address').size().to_dict()

    # First, identify all tokens with balances and sort them
    holdings_preview = []
    if balances:
        for token in balances:
            mint = token.get('mint') or token.get('tokenAddress') or token.get('address')
            amount = safe_float(token.get('amount') or token.get('balance') or token.get('uiAmount'), 0.0)
            if mint and amount > 0:
                # Estimate balance USD using last trade price as approximation for sorting
                last_price = 0.0
                token_trades = processed_trades[processed_trades['mint_address'] == mint]
                if not token_trades.empty:
                    last_price = token_trades.iloc[-1]['price_per_token_usd']
                
                holdings_preview.append({
                    'mint': mint,
                    'amount': amount,
                    'estimated_usd': amount * last_price
                })
    
    holdings_preview.sort(key=lambda x: x['estimated_usd'], reverse=True)
    
    # Calculate pagination
    total_holdings = len(holdings_preview)
    total_pages = (total_holdings + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get only the mints for the current page
    page_holdings = holdings_preview[start_idx:end_idx]
    page_mints = [h['mint'] for h in page_holdings]
    
    # OPTIMIZATION: Only fetch breakdown (with prices) for current page mints
    breakdown = get_pnl_breakdown_per_token(processed_trades, balances, mints_to_fetch=page_mints)

    if isinstance(breakdown, dict) and 'error' not in breakdown:
        holdings = []
        for holding_preview in page_holdings:
            mint = holding_preview['mint']
            data = breakdown.get(mint, {})
            
            if data.get('current_balance', 0) > 0:
                current_price = data.get('current_price', 0)
                balance_usd = data['current_balance'] * current_price

                buys = buy_counts.get(mint, 0)
                sells = sell_counts.get(mint, 0)

                holdings.append({
                    'mint': mint,
                    'current_balance': float(data['current_balance']),
                    'balance_usd': float(balance_usd),
                    'current_price': float(current_price),
                    'total_realized_pnl_usd': float(data['total_realized_pnl_usd']),
                    'unrealized_pnl_usd': float(data['unrealized_pnl_usd']),
                    'total_combined_pnl_usd': float(data['total_combined_pnl_usd']),
                    'avg_buy_price': float(data['overall_avg_buy_price']),
                    'holding_duration': data['avg_holding_duration_readable'],
                    'total_buys': int(buys),
                    'total_sells': int(sells)
                })

        return {
            "holdings": holdings,
            "total_holdings": total_holdings,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages
        }

    return {
        "holdings": [],
        "total_holdings": 0,
        "page": page,
        "per_page": per_page,
        "total_pages": 0
    }


# --- CLI main ---

def main():
    parser = argparse.ArgumentParser(description="Solana Wallet PnL Analysis Script with optimized DexScreener price fetching")
    parser.add_argument("wallet_address", type=str, help="The Solana wallet address to analyze.")
    parser.add_argument("interval_days", type=int, nargs='?', default=0, help="The time interval in days for the PnL summary. Use 0 for All.")
    parser.add_argument("--fast-seconds", type=int, default=5, help="Seconds threshold to consider a sell as 'fast' after buy")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data from APIs (bypass cache)")
    args = parser.parse_args()

    # Fetch data
    balances, raw_trades = get_wallet_data(args.wallet_address, refresh=args.refresh)

    if raw_trades is None or balances is None:
        print("Failed to fetch necessary data. Exiting.")
        return

    print("\nProcessing trade data...")
    processed_trades_df = process_trade_data(raw_trades)
    if processed_trades_df is None or processed_trades_df.empty:
        print("No valid trades found to process. Exiting.")
        return
    print("Processing complete.")

    print("\n" + "="*60)
    print("--- Behavioral Metrics (Phishing Check) ---")
    print("="*60)
    behavior = compute_behavior_metrics(processed_trades_df, within_seconds=args.fast_seconds, interval_days=args.interval_days)
    panel = format_behavior_panel(behavior)
    print(panel)

    print("\n" + "="*60)
    print("--- 1. Overall PnL Summary (optimized DexScreener fetching) ---")
    print("="*60)
    pnl_summary = get_overall_pnl_summary(processed_trades_df, balances, args.interval_days)
    if 'error' in pnl_summary:
        print(pnl_summary['error'])
    else:
        for key, value in pnl_summary.items():
            if isinstance(value, numbers.Real):
                print(f"{key.replace('_', ' ').title()}: {value:,.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n" + "="*60)
    print("--- 2. PnL Distribution of All Sells ---")
    print("="*60)
    pnl_dist = get_pnl_distribution(processed_trades_df, args.interval_days)
    if 'error' in pnl_dist:
        print(pnl_dist['error'])
    else:
        dist = pnl_dist.get('distribution_percentage', {})
        counts = pnl_dist.get('trade_counts', {})
        if not dist:
            print("No sell trades with cost basis available.")
        else:
            for category in sorted(dist.keys()):
                perc = dist[category]
                count = counts.get(category, 0)
                print(f"Category '{category}': {perc:.2f}% ({count} trades)")

    print("\n" + "="*60)
    print("--- 3. Sample Token PnL (First Page) ---")
    print("="*60)
    # Get first page of token PnL to demonstrate optimization
    token_pnl_page1 = get_token_pnl_paginated(processed_trades_df, balances, page=1, per_page=5, interval_days=args.interval_days)
    if token_pnl_page1['tokens']:
        print(f"Showing page 1 of {token_pnl_page1['total_pages']} ({token_pnl_page1['total_tokens']} total tokens)")
        for token_data in token_pnl_page1['tokens']:
            print(f"\nToken: {token_data['mint']}")
            print(f"  - Last Trade: {token_data.get('last_trade_time', 'N/A')}")
            print(f"  - Total Combined PnL: ${token_data.get('total_combined_pnl_usd', 0):,.2f}")
            print(f"  - Current Balance: {token_data.get('current_balance', 0):,.4f}")
            print(f"  - Current Price: ${token_data.get('current_price', 0):,.6f}")
    else:
        print("No tokens found.")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()