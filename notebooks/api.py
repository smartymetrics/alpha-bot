# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from alpha import run_pipeline  
import json
from pathlib import Path
from datetime import datetime
from supabase import create_client, Client
import joblib
import io

# Add these constants after other constants
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "monitor-data"
JOBS_FOLDER = "jobs"

def upload_to_supabase(job_id: str, data: Any, suffix: str = "pkl"):
    """Upload data to Supabase storage in monitor-data/jobs folder"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Create a temporary file
        temp_file = f"temp_{job_id}.{suffix}"
        joblib.dump(data, temp_file)
        
        # Upload to Supabase
        with open(temp_file, "rb") as f:
            file_path = f"{JOBS_FOLDER}/{job_id}.{suffix}"
            supabase.storage.from_(BUCKET_NAME).upload(
                path=file_path,
                file=f,
                file_options={"content-type": "application/octet-stream"}
            )
        
        # Clean up temp file
        os.remove(temp_file)
        logger.info(f"Uploaded {suffix} file for job {job_id} to Supabase")
        
    except Exception as e:
        logger.error(f"Failed to upload to Supabase: {e}")
        raise

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger("api")

app = FastAPI(title="Trader ROI FastAPI (thread-offload)")

# ThreadPoolExecutor for background tasks (module-level so it persists for the lifetime of the process)
MAX_WORKERS = int(os.environ.get("API_MAX_WORKERS", "2"))
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)
JOB_STATUS_DIR = Path("job_status")
JOB_STATUS_DIR.mkdir(exist_ok=True)

# simple in-memory map of job futures for optional inspection
JOB_FUTURES: Dict[str, Any] = {}

class AnalysisRequest(BaseModel):
    tokens: List[str]
    min_buy: Optional[float] = 100.0
    min_num_tokens_in_profit: Optional[int] = 1
    window: Optional[int] = None
    trader_type: Optional[str] = "all"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalysisRequest):
    """Starts run_pipeline in a background thread and returns a jobId immediately."""
    try:
        job_id = uuid.uuid4().hex
        
        # Save initial job status to disk
        status_file = JOB_STATUS_DIR / f"{job_id}.json"
        status_data = {
            "jobId": job_id,
            "status": "running",
            "startedAt": datetime.now().isoformat(),
            "tokens": req.tokens
        }
        with open(status_file, "w") as f:
            json.dump(status_data, f)

        # Submit the long-running pipeline to background thread
        future = EXECUTOR.submit(
            run_pipeline,
            tokens=req.tokens,
            early_trading_window_hours=req.window if (req.trader_type == "early" and req.window) else None,
            minimum_initial_buy_usd=req.min_buy,
            min_profitable_trades=req.min_num_tokens_in_profit,
            job_id=job_id
        )

        # Store future in memory (for running status checks)
        JOB_FUTURES[job_id] = future
        
        # Update status file when future completes

        def _on_complete(fut):
            try:
                result = fut.result()  # Get the result
                status = "done"
                error = None
                
                # Upload result to Supabase if successful
                if result is not None:
                    upload_to_supabase(job_id, result)
                    
            except Exception as e:
                status = "failed" 
                error = str(e)
            
            status_data = {
                "jobId": job_id,
                "status": status,
                "error": error,
                "completedAt": datetime.now().isoformat(),
                "supabasePath": f"{BUCKET_NAME}/{JOBS_FOLDER}/{job_id}.pkl" if status == "done" else None
            }
            with open(status_file, "w") as f:
                json.dump(status_data, f)
                
        future.add_done_callback(_on_complete)
        
        logger.info(f"Submitted job {job_id} for tokens={req.tokens}")
        return {"jobId": job_id}

    except Exception as e:
        logger.exception("Failed to submit job")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
def job_status(job_id: str):
    """Check job status and include Supabase storage path if complete"""
    #  First check in-memory future
    fut = JOB_FUTURES.get(job_id)
    logger.info(f"Checking status for job {job_id} (future exists: {fut is not None})")
    
    # Check status file
    status_file = JOB_STATUS_DIR / f"{job_id}.json"
    if status_file.exists():
        try:
            with open(status_file) as f:
                status_data = json.load(f)
            return status_data
        except Exception as e:
            logger.error(f"Error reading status file: {e}")
    
    # If no status file but future exists   
    if fut is not None and fut.done():
        try:
            result = fut.result()
            return {
                "jobId": job_id, 
                "status": "done",
                "supabasePath": f"{BUCKET_NAME}/{JOBS_FOLDER}/{job_id}.pkl"
            }
        except Exception as e:
            return {"jobId": job_id, "status": "failed", "error": str(e)}
    
    # Default fallback
    return {
        "jobId": job_id,
        "status": "unknown",
        "message": "Check Supabase storage for result."
    }