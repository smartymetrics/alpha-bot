from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from alpha import run_pipeline  # your main analysis logic
import datetime
import os
import json

# If you want Supabase integration for history
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# ------------------- Models -------------------
class AnalysisRequest(BaseModel):
    tokens: List[str]
    min_buy: Optional[float] = 100
    min_num_tokens_in_profit: Optional[int] = 1
    window: Optional[int] = None
    trader_type: Optional[str] = "all"


# ------------------- Helpers -------------------
def save_to_supabase(analysis: dict):
    """Optionally save analysis result in Supabase Storage"""
    if not supabase:
        return
    try:
        bucket = "monitor-data"
        folder = "recent_analyses"
        now = int(datetime.datetime.utcnow().timestamp() * 1000)
        file_path = f"{folder}/analysis_{now}.json"
        supabase.storage.from_(bucket).upload(
            file_path,
            json.dumps(analysis),
            {"content-type": "application/json"},
        )
    except Exception as e:
        print("[api.py] Supabase save failed:", e)


# ------------------- Routes -------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}


@app.post("/analyze")
def analyze(req: AnalysisRequest):
    result = run_pipeline(
        tokens=req.tokens,
        early_trading_window_hours=req.window if req.trader_type == "early" else None,
        minimum_initial_buy_usd=req.min_buy,
        min_profitable_trades=req.min_num_tokens_in_profit,
    )

    if result is None or result.empty:
        return {"records": [], "tokens": req.tokens, "params": req.dict()}

    analysis = {
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "tokens": req.tokens,
        "params": req.dict(),
        "records": result.to_dict(orient="records"),
    }

    # save to Supabase for history
    # save_to_supabase(analysis)

    # return only THIS user's analysis
    return analysis
