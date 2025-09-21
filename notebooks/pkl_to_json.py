#!/usr/bin/env python3
"""
alpha.py - Solana Early Trader PNL Analysis (with Supabase upload for free plan)

Usage:
    python alpha.py <token_mint1> [token_mint2 token_mint3]
    Optional: --window (hours), --minbuy (USD), --minprof (num profitable tokens required)

Notes:
 - Up to 3 tokens allowed.
 - Requires MORALIS_API_KEY, HELIUS_API_KEY, SUPABASE_URL, SUPABASE_KEY in env or .env
 - Caches Dexscreener current prices in sol_price_cache.pkl using joblib.
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
from datetime import timedelta, datetime, timezone
from typing import List
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# =========================
# Config & Keys
# =========================
CACHE_FILE = "sol_price_cache.pkl"
MORALIS_API_KEY = os.environ.get("MORALIS_API_KEY")
HELIUS_API_KEY = os.environ.get("HELIUS_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "your-bucket-name"

HELIUS_RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

DEFAULT_EARLY_WINDOW_HOURS = 6
DEFAULT_MINIMUM_INITIAL_BUY_USD = 100.0
DEFAULT_MIN_PROFITABLE_TRADES = 1

BASE_TOKENS = {
    "So11111111111111111111111111111111111111112",  # SOL
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"   # USDC
}

# =========================
# Helpers
# =========================
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def ensure_keys_present():
    if not MORALIS_API_KEY:
        raise RuntimeError("MORALIS_API_KEY not set in environment")
    if not HELIUS_API_KEY:
        raise RuntimeError("HELIUS_API_KEY not set in environment")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set in environment")

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# Supabase save (delete first if exists)
# =========================
def save_to_supabase(df: pd.DataFrame):
    """Save DataFrame to Supabase in recent_analyses folder using timestamp."""
    supabase = get_supabase_client()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = "recent_analyses"
    os.makedirs(folder, exist_ok=True)
    pkl_filename_local = f"{folder}/analysis_{ts}.pkl"
    json_filename_local = f"{folder}/analysis_{ts}.json"

    # Save locally
    joblib.dump(df, pkl_filename_local)
    with open(json_filename_local, "w") as f:
        json.dump(df.fillna(0).to_dict(orient="records"), f)
    print(f"✅ Saved locally: {pkl_filename_local}, {json_filename_local}")

    # Delete existing file if exists (Supabase free plan cannot upsert)
    for filename in [pkl_filename_local, json_filename_local]:
        try:
            supabase.storage.from_(BUCKET_NAME).remove([filename])
        except Exception:
            pass  # ignore if file doesn't exist

    # Upload files
    for filename in [pkl_filename_local, json_filename_local]:
        with open(filename, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(filename, f.read())
        print(f"✅ Uploaded to Supabase: {filename}")

# =========================
# Moralis trades downloader
# =========================
def get_solana_dex_trades(token_addresses: List[str], limit: int = 100) -> pd.DataFrame:
    """
    Fetch swaps for token addresses from Moralis (cursor pagination).
    Returns a dataframe with important fields and derived_price fallback.
    """
    if not isinstance(token_addresses, list):
        raise TypeError("token_addresses must be a list")
    if len(token_addresses) == 0:
        return pd.DataFrame()
    if len(token_addresses) > 3:
        raise ValueError("A maximum of 3 tokens is allowed per request.")
    ensure_keys_present()

    headers = {"Accept": "application/json", "X-API-Key": MORALIS_API_KEY}
    all_rows = []

    for addr in token_addresses:
        url = f"https://solana-gateway.moralis.io/token/mainnet/{addr}/swaps"
        cursor = None
        print(f"[Moralis] fetching swaps for {addr}")
        while True:
            params = {"limit": limit, "order": "DESC", "transactionTypes": "buy,sell"}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"[Moralis] request error: {e}")
                break

            trades = data.get("result", []) or []
            if not trades:
                break

            for t in trades:
                direction = t.get("transactionType")
                bought = t.get("bought") or {}
                sold = t.get("sold") or {}

                amount_usd = _safe_float(t.get("totalValueUsd"))
                token_bought_amt = _safe_float(bought.get("amount")) if direction == "buy" else _safe_float(sold.get("amount"))
                token_sold_amt = _safe_float(sold.get("amount")) if direction == "buy" else _safe_float(bought.get("amount"))

                price_usd_at_trade = None
                if amount_usd is not None and token_bought_amt not in (None, 0):
                    price_usd_at_trade = amount_usd / token_bought_amt

                bt = pd.to_datetime(t.get("blockTimestamp"))

                all_rows.append({
                    "block_time": bt,
                    "block_date": bt.normalize(),
                    "transaction_hash": t.get("transactionHash"),
                    "transaction_type": direction,
                    "trader_id": t.get("walletAddress"),
                    "pair_address": t.get("pairAddress"),
                    "pair_label": t.get("pairLabel"),
                    "exchange_name": t.get("exchangeName"),
                    "token_bought_mint_address": bought.get("address") if direction == "buy" else sold.get("address"),
                    "token_bought_amount": token_bought_amt,
                    "token_sold_mint_address": sold.get("address") if direction == "buy" else bought.get("address"),
                    "token_sold_amount": token_sold_amt,
                    "amount_usd": amount_usd,
                    "price_usd_at_trade": price_usd_at_trade
                })

            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(0.15)

        print(f"[Moralis] fetched {len(all_rows)} total trades (so far).")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # derived_price fallback
    df["derived_price"] = df["price_usd_at_trade"]
    mask = df["derived_price"].isna() & df["token_bought_amount"].notna() & df["amount_usd"].notna()
    df.loc[mask, "derived_price"] = df.loc[mask, "amount_usd"] / df.loc[mask, "token_bought_amount"].replace({0: np.nan})
    df["block_time"] = pd.to_datetime(df["block_time"])
    return df.sort_values("block_time").reset_index(drop=True)

# =========================
# Price builder (per-minute OHLC + vwap)
# =========================
def build_minute_prices_from_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["mint_address","minute","open","high","low","close","vwap","volume_usd"])

    df = trades_df.copy()
    df["price_for_price_series"] = df["price_usd_at_trade"].fillna(df["derived_price"])
    df["minute"] = df["block_time"].dt.floor("min")
    grouped = df.groupby(["token_bought_mint_address", "minute"], sort=True)

    rows = []
    for (mint, minute), g in grouped:
        prices = g["price_for_price_series"].dropna()
        if prices.empty:
            continue
        open_p = prices.iloc[0]
        high_p = prices.max()
        low_p = prices.min()
        close_p = prices.iloc[-1]
        if g["amount_usd"].notna().any():
            weights = g["amount_usd"].fillna(0.0)
            denom = weights.sum()
            if denom > 0:
                vwap = (prices * weights).sum() / denom
            else:
                vwap = prices.mean()
        else:
            vwap = prices.mean()
        volume_usd = g["amount_usd"].fillna(0.0).sum()
        rows.append({
            "mint_address": mint,
            "minute": minute,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "vwap": vwap,
            "volume_usd": volume_usd
        })

    price_df = pd.DataFrame(rows)
    if not price_df.empty:
        price_df = price_df.sort_values(["mint_address","minute"]).reset_index(drop=True)
    return price_df

def get_price_at_minute(price_minute_df: pd.DataFrame, mint: str, ts: pd.Timestamp, tolerance_minutes: int = 5):
    if price_minute_df.empty:
        return None
    t_min = ts.floor("min")
    subset = price_minute_df[price_minute_df["mint_address"] == mint]
    if subset.empty:
        return None
    if t_min in set(subset["minute"]):
        row = subset[subset["minute"] == t_min].iloc[0]
        return _safe_float(row["vwap"]) or _safe_float(row["close"]) or None
    deltas = (subset["minute"] - t_min).abs()
    valid = subset[deltas <= pd.Timedelta(minutes=tolerance_minutes)]
    if valid.empty:
        return None
    idx = deltas[valid.index].idxmin()
    row = subset.loc[idx]
    return _safe_float(row["vwap"]) or _safe_float(row["close"]) or None

# =========================
# Dexscreener price caching
# =========================
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
        print(f"[Cache] save error: {e}")

async def dexscreener_fetch_price(mint: str, session: aiohttp.ClientSession) -> float:
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            pairs = data.get("pairs") or []
            if not pairs:
                return None
            return _safe_float(pairs[0].get("priceUsd"))
    except Exception:
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

# =========================
# Helius balances + aggregator
# =========================
async def fetch_solana_rpc(payload: dict) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(HELIUS_RPC_URL, json=payload, timeout=30) as resp:
            resp.raise_for_status()
            return await resp.json()

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
        except Exception:
            sol_balance = 0.0

        tokens_payload = {
            "jsonrpc":"2.0","id":"1","method":"getTokenAccountsByOwner",
            "params":[trader, {"programId":"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}, {"encoding":"jsonParsed"}]
        }
        try:
            tok_res = await fetch_solana_rpc(tokens_payload)
            accounts = tok_res.get("result",{}).get("value",[]) or []
        except Exception:
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

# =========================
# Main pipeline
# =========================
def run_pipeline(tokens: List[str],
                 early_trading_window_hours: int = None,
                 minimum_initial_buy_usd: float = None,
                 min_profitable_trades: int = DEFAULT_MIN_PROFITABLE_TRADES):
    ensure_keys_present()
    tokens = [t.strip() for t in tokens]
    if len(tokens) == 0:
        print("No tokens provided.")
        return
    if len(tokens) > 3:
        raise ValueError("A maximum of 3 token addresses is allowed.")

    trades_df = get_solana_dex_trades(tokens)
    if trades_df.empty:
        print("No trades returned from Moralis.")
        return

    minute_price_df = build_minute_prices_from_trades(trades_df)

    token_first_trade_times = trades_df.groupby("token_bought_mint_address", as_index=False).agg(token_launch_time=("block_time","min"))

    # earliest buy per trader/token
    buys_mask = trades_df["token_bought_amount"].notna() & (trades_df["token_bought_amount"] > 0)
    buys_only = trades_df[buys_mask].copy()
    tfbd = buys_only.loc[buys_only.groupby(["trader_id","token_bought_mint_address"])["block_time"].idxmin()].copy()
    tfbd = tfbd.rename(columns={"block_time":"first_personal_buy_time"})
    tfbd = tfbd[["trader_id","token_bought_mint_address","first_personal_buy_time","token_bought_amount","amount_usd","derived_price","block_date"]]

    # compute first buy USD (priority: Moralis -> minute vwap -> derived_price)
    def compute_first_buy_usd(row):
        if _safe_float(row.get("amount_usd")):
            return float(row["amount_usd"])
        minute_price = get_price_at_minute(minute_price_df, row["token_bought_mint_address"], row["first_personal_buy_time"], tolerance_minutes=5)
        if _safe_float(minute_price) and _safe_float(row.get("token_bought_amount")):
            return float(minute_price) * float(row["token_bought_amount"])
        if _safe_float(row.get("derived_price")) and _safe_float(row.get("token_bought_amount")):
            return float(row["derived_price"]) * float(row["token_bought_amount"])
        return 0.0

    tfbd["first_buy_usd_amount"] = tfbd.apply(compute_first_buy_usd, axis=1)

    # Build cohort using optional filters
    merged_for_cohort = pd.merge(tfbd, token_first_trade_times, left_on="token_bought_mint_address", right_on="token_bought_mint_address", how="left")
    merged_for_cohort["time_since_launch"] = merged_for_cohort["first_personal_buy_time"] - merged_for_cohort["token_launch_time"]

    # Start with all rows, then apply filters only if specified
    cohort_df = merged_for_cohort.copy()

    if early_trading_window_hours is not None and early_trading_window_hours > 0:
        cohort_df = cohort_df[cohort_df["time_since_launch"] <= timedelta(hours=early_trading_window_hours)]

    if minimum_initial_buy_usd is not None and minimum_initial_buy_usd > 0:
        cohort_df = cohort_df[cohort_df["first_buy_usd_amount"] >= minimum_initial_buy_usd]

    # If both filters omitted, cohort_df stays as merged_for_cohort (i.e., all first buyers)
    early_traders_cohort = cohort_df["trader_id"].unique().tolist()
    print(f"[Cohort] {len(early_traders_cohort)} traders in cohort (after applied filters).")

    if len(early_traders_cohort) == 0:
        print("No traders meeting conditions. Exiting.")
        return

    # -------------------------------------------------------------------------
    # compute acquisition, sales realized, current networth, combine to token metrics
    # -------------------------------------------------------------------------
    cohort_trades_df = trades_df[trades_df["trader_id"].isin(early_traders_cohort)].copy()

    # Acquisition metrics
    buys = cohort_trades_df[cohort_trades_df["token_bought_amount"].notna() & (cohort_trades_df["token_bought_amount"] > 0)].copy()

    def compute_buy_usd_row(row):
        if _safe_float(row.get("amount_usd")):
            return float(row["amount_usd"])
        minute_price = get_price_at_minute(minute_price_df, row["token_bought_mint_address"], row["block_time"], tolerance_minutes=5)
        if _safe_float(minute_price) and _safe_float(row.get("token_bought_amount")):
            return float(minute_price) * float(row["token_bought_amount"])
        if _safe_float(row.get("derived_price")) and _safe_float(row.get("token_bought_amount")):
            return float(row["derived_price"]) * float(row["token_bought_amount"])
        return 0.0

    buys["buy_usd_spent"] = buys.apply(compute_buy_usd_row, axis=1)

    acquisition = buys.groupby(["trader_id","token_bought_mint_address"], as_index=False).agg(
        total_usd_spent=("buy_usd_spent","sum"),
        total_tokens_bought=("token_bought_amount","sum")
    )
    acquisition["avg_buy_price_usd"] = acquisition["total_usd_spent"] / acquisition["total_tokens_bought"].replace({0: np.nan})
    acquisition = acquisition.rename(columns={"token_bought_mint_address":"mint_address"})

    # Sales & realized profit
    sells = cohort_trades_df[cohort_trades_df["token_sold_amount"].notna() & (cohort_trades_df["token_sold_amount"] > 0)].copy()

    def compute_sell_usd_row(row):
        if _safe_float(row.get("amount_usd")):
            return float(row["amount_usd"])
        minute_price = get_price_at_minute(minute_price_df, row["token_sold_mint_address"], row["block_time"], tolerance_minutes=5)
        if _safe_float(minute_price) and _safe_float(row.get("token_sold_amount")):
            return float(minute_price) * float(row["token_sold_amount"])
        if _safe_float(row.get("derived_price")) and _safe_float(row.get("token_sold_amount")):
            return float(row["derived_price"]) * float(row["token_sold_amount"])
        return 0.0

    if not sells.empty:
        sells["sell_revenue_usd"] = sells.apply(compute_sell_usd_row, axis=1)
    else:
        sells["sell_revenue_usd"] = pd.Series(dtype=float)

    sells = pd.merge(
        sells,
        acquisition[["trader_id","mint_address","avg_buy_price_usd"]],
        left_on=["trader_id","token_sold_mint_address"],
        right_on=["trader_id","mint_address"],
        how="left"
    )
    sells["cost_of_goods_sold"] = sells["token_sold_amount"] * sells["avg_buy_price_usd"].fillna(0.0)
    sells["realized_pnl_usd"] = sells["sell_revenue_usd"].fillna(0.0) - sells["cost_of_goods_sold"].fillna(0.0)

    sales_profit = sells.groupby(["trader_id","token_sold_mint_address"], as_index=False).agg(
        total_sales_revenue_usd=("sell_revenue_usd","sum"),
        realized_profit_usd=("realized_pnl_usd","sum")
    ).rename(columns={"token_sold_mint_address":"mint_address"})

    # Combine acquisition + sales
    trader_token_metrics = pd.merge(
        acquisition,
        sales_profit,
        on=["trader_id","mint_address"],
        how="left"
    ).fillna({"total_sales_revenue_usd":0.0,"realized_profit_usd":0.0})

    # Fetch current balances/prices
    token_mints_to_fetch = trader_token_metrics["mint_address"].dropna().unique().tolist()
    balances_df = asyncio.run(get_current_balances_and_prices(list(early_traders_cohort), token_mints_to_fetch))

    # ensure rows for missing pairs
    if balances_df.empty:
        rows = []
        for t in early_traders_cohort:
            for m in token_mints_to_fetch:
                rows.append({"trader_id":t,"mint_address":m,"token_balance":0.0,"current_price":0.0,"current_value_usd":0.0,"sol_balance":0.0})
        balances_df = pd.DataFrame(rows)

    trader_token_metrics = pd.merge(
        trader_token_metrics,
        balances_df[["trader_id","mint_address","token_balance","current_price","current_value_usd"]],
        on=["trader_id","mint_address"],
        how="left"
    ).fillna({"token_balance":0.0,"current_price":0.0,"current_value_usd":0.0})

    # Compute unrealised & token_total_pnl
    trader_token_metrics["avg_buy_price_usd"] = trader_token_metrics["avg_buy_price_usd"].fillna(0.0)
    trader_token_metrics["unrealized_cost_basis"] = trader_token_metrics["token_balance"] * trader_token_metrics["avg_buy_price_usd"]
    trader_token_metrics["estimated_unrealised_pnl"] = trader_token_metrics["current_value_usd"] - trader_token_metrics["unrealized_cost_basis"]
    trader_token_metrics["token_total_pnl"] = trader_token_metrics["realized_profit_usd"].fillna(0.0) + trader_token_metrics["estimated_unrealised_pnl"].fillna(0.0)
    trader_token_metrics["is_token_profitable"] = (trader_token_metrics["token_total_pnl"] > 0).astype(int)

    # Aggregate to trader level (final summary)
    final_summary = trader_token_metrics.groupby("trader_id", as_index=False).agg(
        overall_total_usd_spent=("total_usd_spent","sum"),
        overall_total_sales_revenue=("total_sales_revenue_usd","sum"),
        overall_realized_profit_usd=("realized_profit_usd","sum"),
        overall_estimated_unrealised_pnl=("estimated_unrealised_pnl","sum"),
        overall_current_value_of_holdings_usd=("current_value_usd","sum"),
        overall_total_pnl=("token_total_pnl","sum"),
        num_unique_tokens_traded=("mint_address","nunique"),
        num_tokens_in_profit=("is_token_profitable","sum")
    )

    final_summary["ROI"] = (final_summary["overall_total_pnl"] / final_summary["overall_total_usd_spent"]).replace([np.inf, np.nan], 0).fillna(0)
    final_summary["win_rate"] = (final_summary["num_tokens_in_profit"] / final_summary["num_unique_tokens_traded"]).fillna(0)

    trade_counts = cohort_trades_df.groupby("trader_id").size().reset_index(name="number_of_trades")
    final_summary = pd.merge(final_summary, trade_counts, on="trader_id", how="left").fillna(0)

    # final filter: require min_profitable_trades profitable tokens
    final_summary = final_summary[final_summary["num_tokens_in_profit"] >= min_profitable_trades]
    final_summary = final_summary.sort_values(by=["ROI","overall_total_usd_spent"], ascending=[False,False]).reset_index(drop=True)

    # display columns
    display_cols = [
        "trader_id","overall_current_value_of_holdings_usd","overall_total_pnl","overall_realized_profit_usd",
        "overall_estimated_unrealised_pnl","ROI","overall_total_usd_spent","overall_total_sales_revenue",
        "num_unique_tokens_traded","num_tokens_in_profit","number_of_trades"
    ]
    for c in display_cols:
        if c not in final_summary.columns:
            final_summary[c] = 0.0

    print("\n--- 🏆 Final Early Trader Profitability Report (top 50) ---")
    print(final_summary[display_cols].head(50).to_string(index=False))
    # Save & upload to Supabase
    save_to_supabase(final_summary)

    return final_summary

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solana Early Trader PNL Analysis (up to 3 tokens)")
    parser.add_argument("tokens", nargs="+", help="List of 1-3 token mint addresses (positional)")
    parser.add_argument("--window", type=int, default=None, help="Early trading window (hours) - optional")
    parser.add_argument("--minbuy", type=float, default=None, help="Minimum initial buy in USD - optional")
    parser.add_argument("--minprof", type=int, default=DEFAULT_MIN_PROFITABLE_TRADES, help="Minimum number of profitable tokens")
    args = parser.parse_args()

    if len(args.tokens) > 3:
        parser.error("A maximum of 3 token addresses is allowed.")

    run_pipeline(tokens=args.tokens,
                 early_trading_window_hours=args.window,
                 minimum_initial_buy_usd=args.minbuy,
                 min_profitable_trades=args.minprof)
