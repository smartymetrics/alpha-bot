# monitor_web.py
"""
monitor_web.py - Web service wrapper for token_monitor.py and wallet.py
Keeps the monitor running 24/7 on Render free tier
Enhanced with Wallet PnL Analysis endpoints
"""
import os
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd


# Import the main monitoring loop
from token_monitor import main_loop
# COMMENTED OUT - Winner monitor moved to separate service
# from winner_monitor import main

# Import wallet PnL functions
from wallet import (
    get_dex_trades_from_moralis,
    get_current_balances_from_helius_rpc,
    process_trade_data,
    compute_behavior_metrics,
    format_behavior_panel,
    get_overall_pnl_summary,
    get_pnl_distribution,
    get_pnl_breakdown_per_token,
    get_wallet_data,
    SUPABASE_AVAILABLE,
    get_token_pnl_paginated,    
    get_trades_paginated,        
    get_holdings_paginated       
)

# import the predictor script
from ml_predictor import SolanaTokenPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the predictor globally (loads models once at startup)
try:
    ml_predictor = SolanaTokenPredictor(model_dir='models')
    ML_PREDICTOR_AVAILABLE = True
    logger.info("✅ ML Predictor loaded successfully")
except Exception as e:
    ml_predictor = None
    ML_PREDICTOR_AVAILABLE = False
    logger.warning(f"⚠️ ML Predictor not available: {e}")

# Track service status
service_status = {
    "started_at": None,
    "last_activity": None,
    "monitor_running": False,
    # COMMENTED OUT - Moved to separate service
    # "winner_monitor_running": False,
    "error": None,
    "wallet_requests": 0,
    "last_wallet_request": None,
    "ml_predictor_available": ML_PREDICTOR_AVAILABLE,  
    "ml_predictions": 0,  
    "last_ml_prediction": None 
}

async def run_monitor():
    """Wrapper to run monitor and handle errors"""
    try:
        await main_loop()
    except Exception as e:
        logger.error(f"Monitor crashed: {e}")
        service_status["monitor_running"] = False
        service_status["error"] = str(e)
        # Attempt to restart after 10 seconds
        await asyncio.sleep(10)
        logger.info("Attempting to restart monitor...")
        asyncio.create_task(run_monitor())

# COMMENTED OUT - Winner monitor moved to separate service
# async def run_winner_monitor():
#     """
#     Wrapper to run winner_monitor.py's main function and handle errors.
#     """
#     try:
#         await main()
#     except Exception as e:
#         logger.error(f"Winner monitor crashed: {e}")
#         service_status["winner_monitor_running"] = False
#         service_status["error"] = str(e)
#         # Attempt to restart after 10 seconds
#         await asyncio.sleep(10)
#         logger.info("Attempting to restart winner monitor...")
#         asyncio.create_task(run_winner_monitor())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    service_status["started_at"] = datetime.utcnow().isoformat()
    service_status["monitor_running"] = True
    # COMMENTED OUT - Moved to separate service
    # service_status["winner_monitor_running"] = True

    logger.info("Starting token monitor background task...")
    # COMMENTED OUT - Winner monitor moved to separate service
    # logger.info("Starting token monitor and winner monitor background tasks...")

    # Run main_loop as background task
    monitor_task = asyncio.create_task(run_monitor())
    # COMMENTED OUT - Moved to separate service
    # winner_monitor_task = asyncio.create_task(run_winner_monitor())

    yield

    # Shutdown
    logger.info("Shutting down token monitor...")
    monitor_task.cancel()

    # COMMENTED OUT - Moved to separate service
    # logger.info("Shutting down winner monitor...")
    # winner_monitor_task.cancel()
    
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    # COMMENTED OUT - Moved to separate service
    # try:
    #     await winner_monitor_task
    # except asyncio.CancelledError:
    #     pass

app = FastAPI(
    title="Token Monitor & Wallet PnL Service",
    description="24/7 Token monitoring with Wallet PnL analysis capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Original Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Token Monitor & Wallet PnL Service with ML Predictor",
        "status": "running" if service_status["monitor_running"] else "stopped",
        "uptime_start": service_status["started_at"],
        "features": [
            "24/7 Token Monitoring",
            "Wallet PnL Analysis",
            "Behavioral Metrics",
            "Phishing Detection",
            "ML-Powered Token Prediction"
        ],
        "endpoints": {
            "monitor": ["/health", "/status"],
            "wallet_analysis": [
                "/wallet/{address}/pnl",
                "/wallet/{address}/behavior",
                "/wallet/{address}/full-analysis",
                "/wallet/{address}/distribution",
                "/wallet/{address}/tokens",
                "/wallet/{address}/token-pnl",
                "/wallet/{address}/trades",
                "/wallet/{address}/holdings",
                "/wallet/{address}/overall-analysis"
            ],
            "ml_predictor": [
                "/token/{mint}/predict",
                "/token/predict/batch",
                "/ml/status",
                "/ml/features"
            ]
        },
        "ml_predictor_status": {
            "available": ML_PREDICTOR_AVAILABLE,
            "total_predictions": service_status["ml_predictions"]
        }
    }

@app.head("/health")
@app.get("/health")
async def health():
    """Health check endpoint for UptimeRobot"""
    service_status["last_activity"] = datetime.utcnow().isoformat()
    
    return JSONResponse(
        status_code=200 if service_status["monitor_running"] else 503,
        content={
            "status": "healthy" if service_status["monitor_running"] else "unhealthy",
            "monitor_running": service_status["monitor_running"],
            "started_at": service_status["started_at"],
            "last_ping": service_status["last_activity"],
            "error": service_status["error"]
        }
    )

@app.get("/status")
async def status():
    """Detailed status endpoint"""
    return {
        "service": "Token Monitor & Wallet PnL Web Service",
        "monitor_status": service_status,
        "environment": {
            "render": os.getenv("RENDER") is not None,
            "port": os.getenv("PORT", "not set"),
            "supabase_available": SUPABASE_AVAILABLE
        }
    }

# ==================== Wallet PnL Endpoints ====================

@app.get("/wallet/{address}/pnl")
async def get_wallet_pnl(
    address: str,
    interval_days: int = Query(7, ge=0, description="Analysis interval in days. Use 0 for all time"),
    refresh: bool = Query(False, description="Force refresh from APIs")
):
    """
    Get PnL summary for a wallet address
    
    - **address**: Solana wallet address
    - **interval_days**: Time interval for analysis (0 = all time)
    - **refresh**: Force refresh data from APIs instead of using cache
    """
    try:
        logger.info(f"PnL request for wallet: {address[:8]}... (interval: {interval_days} days, refresh: {refresh})")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address)
            balances, _ = await asyncio.to_thread(get_current_balances_from_helius_rpc, address, False)
        
        if trades is None or balances is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        # Process trades
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "interval_days": interval_days,
                "message": "No valid trades found for this wallet",
                "pnl_summary": None
            }
        
        # Get PnL summary
        pnl_summary = await asyncio.to_thread(
            get_overall_pnl_summary, 
            processed_trades, 
            balances, 
            interval_days
        )
        
        return {
            "wallet": address,
            "interval_days": interval_days,
            "timestamp": datetime.utcnow().isoformat(),
            "pnl_summary": pnl_summary,
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }
        
    except Exception as e:
        logger.error(f"Error processing wallet PnL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wallet/{address}/behavior")
async def get_wallet_behavior(
    address: str,
    fast_seconds: int = Query(5, ge=1, le=3600, description="Seconds threshold for fast sells"),
    refresh: bool = Query(False, description="Force refresh from APIs instead of using cache")
):
    """
    Get behavioral metrics and phishing check for a wallet
    
    - **address**: Solana wallet address
    - **fast_seconds**: Time threshold to consider a sell as "fast" after buy
    - **refresh**: Force refresh data from APIs instead of using cache
    """
    try:
        logger.info(f"Behavior request for wallet: {address[:8]}... (fast_seconds: {fast_seconds}, refresh: {refresh})")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address)
            balances, _ = await asyncio.to_thread(get_current_balances_from_helius_rpc, address, False)
        
        if trades is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        # Process trades
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "message": "No valid trades found for this wallet",
                "behavior_metrics": None,
                "phishing_panel": "No data available"
            }
        
        # Compute behavior metrics
        behavior_metrics = await asyncio.to_thread(
            compute_behavior_metrics,
            processed_trades,
            fast_seconds
        )
        
        # Format phishing panel
        phishing_panel = await asyncio.to_thread(format_behavior_panel, behavior_metrics)
        
        return {
            "wallet": address,
            "timestamp": datetime.utcnow().isoformat(),
            "fast_seconds_threshold": fast_seconds,
            "behavior_metrics": behavior_metrics,
            "phishing_panel": phishing_panel,
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }
        
    except Exception as e:
        logger.error(f"Error processing wallet behavior: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wallet/{address}/distribution")
async def get_wallet_pnl_distribution(
    address: str,
    refresh: bool = Query(False, description="Force refresh from APIs")
):
    """
    Get PnL distribution breakdown for all sells
    
    - **address**: Solana wallet address
    - **refresh**: Force refresh data from APIs instead of using cache
    """
    try:
        logger.info(f"PnL distribution request for wallet: {address[:8]}... (refresh: {refresh})")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address)
        
        if trades is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        # Process trades
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "message": "No valid trades found for this wallet",
                "distribution": None
            }
        
        # Get distribution
        distribution = await asyncio.to_thread(get_pnl_distribution, processed_trades)
        
        return {
            "wallet": address,
            "timestamp": datetime.utcnow().isoformat(),
            "distribution": distribution,
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }
        
    except Exception as e:
        logger.error(f"Error processing wallet distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wallet/{address}/tokens")
async def get_wallet_token_breakdown(
    address: str,
    limit: int = Query(10, ge=1, le=100, description="Number of tokens to return"),
    sort_by: str = Query("total_combined_pnl_usd", description="Field to sort by"),
    refresh: bool = Query(False, description="Force refresh data from APIs instead of using cache")
):
    """
    Get per-token PnL breakdown
    
    - **address**: Solana wallet address
    - **limit**: Number of top tokens to return
    - **sort_by**: Field to sort results by
    - **refresh**: Force refresh data from APIs instead of using cache
    """
    try:
        logger.info(f"Token breakdown request for wallet: {address[:8]}... (limit: {limit}, refresh: {refresh})")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address)
            balances, _ = await asyncio.to_thread(get_current_balances_from_helius_rpc, address, False)
        
        if trades is None or balances is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        # Process trades
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "message": "No valid trades found for this wallet",
                "tokens": []
            }
        
        # Get breakdown
        breakdown = await asyncio.to_thread(
            get_pnl_breakdown_per_token,
            processed_trades,
            balances
        )
        
        # Sort and limit results
        if isinstance(breakdown, dict) and 'error' not in breakdown:
            sorted_breakdown = sorted(
                breakdown.items(),
                key=lambda item: item[1].get(sort_by, 0.0),
                reverse=True
            )[:limit]
            
            tokens = [
                {"mint": mint, **data}
                for mint, data in sorted_breakdown
            ]
        else:
            tokens = []
        
        return {
            "wallet": address,
            "timestamp": datetime.utcnow().isoformat(),
            "token_count": len(breakdown) if isinstance(breakdown, dict) else 0,
            "tokens_returned": len(tokens),
            "tokens": tokens,
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }
        
    except Exception as e:
        logger.error(f"Error processing token breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===============================================
# Background Job Management for Full Analysis
# ===============================================

import uuid

# Global job state for tracking async analysis jobs
job_results = {}

async def run_full_analysis_task(
    job_id: str, address: str, interval_days: int, fast_seconds: int, token_limit: int, refresh: bool
):
    """Complete implementation of the analysis task with all metrics"""
    job_results[job_id]["status"] = "running"
    
    try:
        logger.info(f"[JOB:{job_id}] Starting full analysis for wallet: {address[:8]}... (interval: {interval_days} days)")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Step 1: Fetch data from cache or APIs
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address, interval_days)
            balances, _ = await asyncio.to_thread(get_current_balances_from_helius_rpc, address, False)
        
        if trades is None or balances is None:
            raise Exception("Failed to fetch wallet data from APIs")
        
        # Step 2: Process the raw trade data
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            # No trades found - return minimal result
            final_result = {
                "wallet": address,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "No valid trades found for this wallet",
                "analysis": {
                    "pnl_summary": {
                        "analysis_interval_days": interval_days,
                        "total_realized_pnl_usd": 0.0,
                        "total_unrealized_pnl_usd": 0.0,
                        "total_pnl_combined_usd": 0.0,
                        "total_sell_trades_in_interval": 0,
                        "total_buy_trades_in_interval": 0,
                        "total_buy_volume_usd_in_interval": 0.0,
                        "win_rate_percent": 0.0,
                        "avg_holding_before_sell_seconds": None,
                        "avg_holding_before_sell_readable": "N/A"
                    },
                    "behavior_metrics": {
                        "sell_trades_fully_not_bought_count": 0,
                        "tokens_more_sold_than_bought_count": 0,
                        "sells_within_seconds_count": 0,
                        "total_sells": 0,
                        "distinct_tokens": 0
                    },
                    "phishing_panel": "No data available",
                    "pnl_distribution": {
                        "distribution_percentage": {},
                        "trade_counts": {}
                    },
                    "total_tokens": 0
                },
                "parameters": {
                    "interval_days": interval_days,
                    "fast_seconds": fast_seconds,
                    "token_limit": token_limit
                },
                "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
            }
        else:
            # Step 3: Run all analysis tasks concurrently for speed
            logger.info(f"[JOB:{job_id}] Processing {len(processed_trades)} trades...")
            
            # CRITICAL: Pass interval_days to ALL functions
            pnl_task = asyncio.to_thread(
                get_overall_pnl_summary, 
                processed_trades, 
                balances, 
                interval_days
            )
            
            behavior_task = asyncio.to_thread(
                compute_behavior_metrics,
                processed_trades,
                fast_seconds,
                interval_days  # Pass interval_days
            )
            
            distribution_task = asyncio.to_thread(
                get_pnl_distribution, 
                processed_trades,
                interval_days  # Pass interval_days
            )
            
            breakdown_task = asyncio.to_thread(
                get_pnl_breakdown_per_token, 
                processed_trades, 
                balances,
                interval_days  # Pass interval_days
            )
            
            # Wait for all tasks to complete
            pnl_summary, behavior_metrics, distribution, breakdown = await asyncio.gather(
                pnl_task, 
                behavior_task, 
                distribution_task, 
                breakdown_task
            )
            
            logger.info(f"[JOB:{job_id}] All metrics computed successfully")
            
            # Step 4: Format the phishing panel
            phishing_panel = await asyncio.to_thread(format_behavior_panel, behavior_metrics)
            
            # Step 5: Build the final result (NO top_tokens sorting here)
            final_result = {
                "wallet": address,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": {
                    "pnl_summary": pnl_summary,
                    "behavior_metrics": behavior_metrics,
                    "phishing_panel": phishing_panel,
                    "pnl_distribution": distribution,
                    "total_tokens": len(breakdown) if isinstance(breakdown, dict) else 0
                },
                "parameters": {
                    "interval_days": interval_days,
                    "fast_seconds": fast_seconds,
                    "token_limit": token_limit
                },
                "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
            }
        
        # Step 6: Mark job as completed
        job_results[job_id]["status"] = "completed"
        job_results[job_id]["result"] = final_result
        logger.info(f"[JOB:{job_id}] Analysis completed successfully for {address[:8]}...")
        
    except Exception as e:
        logger.error(f"[JOB:{job_id}] Error during analysis: {e}", exc_info=True)
        job_results[job_id]["status"] = "failed"
        job_results[job_id]["error"] = str(e)

# --- 1. THE NEW POST HANDLER (Job Starter) ---
@app.post("/wallet/{address}/full-analysis")
@app.post("/wallet/{address}/overview-analysis")
async def start_full_wallet_analysis(
    address: str,
    interval_days: int = Query(7, ge=0),
    fast_seconds: int = Query(5, ge=1, le=3600),
    token_limit: int = Query(10, ge=1, le=100),
    refresh: bool = Query(False)
):
    """Starts the comprehensive wallet analysis as a background job."""
    
    job_id = str(uuid.uuid4())
    
    # Initialize job state
    job_results[job_id] = {
        "status": "pending",
        "timestamp": datetime.utcnow().isoformat(),
        "address": address,
        "result": None,
        "error": None
    }
    
    # Start the analysis function in the background
    asyncio.create_task(
        run_full_analysis_task(
            job_id, address, interval_days, fast_seconds, token_limit, refresh
        )
    )
    
    # IMMEDIATELY return the jobId
    return {
        "jobId": job_id,
        "status": "pending",
        "message": "Analysis started. Poll /job/{jobId} for results."
    }


# --- 2. THE NEW GET HANDLER (Job Poller) ---
@app.get("/job/{jobId}")
async def get_job_status(jobId: str):
    """Checks the status of an analysis job."""
    
    job_data = job_results.get(jobId)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job ID not found")
        
    if job_data["status"] == "completed":
        # Returns the full analysis result when ready (200 OK)
        return {
            "jobId": jobId,
            "status": "completed",
            "result": job_data["result"]
        }
    
    elif job_data["status"] == "failed":
        # Returns a 500 if the job itself failed
        raise HTTPException(status_code=500, detail=job_data["error"])

    else:
        # Returns a 202 Accepted if the job is still running/pending
        return {
            "jobId": jobId,
            "status": job_data["status"],
            "message": "Processing in progress..."
        }

@app.get("/wallet/{address}/token-pnl")
async def get_wallet_token_pnl_paginated(
    address: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(9, ge=1, le=50, description="Items per page"),
    refresh: bool = Query(False, description="Force refresh from APIs"),
    interval_days: int = Query(9, ge=1, le=50, description="Interval in days")
):
    """
    Get paginated token PnL data ordered by last trade time (descending).
    
    - **address**: Solana wallet address
    - **page**: Page number (starts at 1)
    - **per_page**: Number of items per page (max 50)
    - **refresh**: Force refresh data from APIs instead of using cache
    """
    try:
        logger.info(f"Token PnL paginated request for wallet: {address[:8]}... (page: {page}, per_page: {per_page})")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address, interval_days)
            balances, _ = await asyncio.to_thread(get_current_balances_from_helius_rpc, address, False)
        
        if trades is None or balances is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        # Process trades
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "message": "No valid trades found for this wallet",
                "tokens": [],
                "total_tokens": 0,
                "page": page,
                "per_page": per_page,
                "total_pages": 0
            }
        
        # Get paginated token PnL
        result = await asyncio.to_thread(
            get_token_pnl_paginated,
            processed_trades,
            balances,
            page,
            per_page
        )
        
        return {
            "wallet": address,
            "timestamp": datetime.utcnow().isoformat(),
            **result,
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }
        
    except Exception as e:
        logger.error(f"Error processing token PnL pagination: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wallet/{address}/trades")
async def get_wallet_trades_paginated(
    address: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(9, ge=1, le=50, description="Items per page"),
    refresh: bool = Query(False, description="Force refresh from APIs")
):
    """
    Get paginated individual trades ordered by trade time (descending).
    
    - **address**: Solana wallet address
    - **page**: Page number (starts at 1)
    - **per_page**: Number of items per page (max 50)
    - **refresh**: Force refresh data from APIs instead of using cache
    """
    try:
        logger.info(f"Trades paginated request for wallet: {address[:8]}... (page: {page}, per_page: {per_page})")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address)
        
        if trades is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        # Process trades
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "message": "No valid trades found for this wallet",
                "trades": [],
                "total_trades": 0,
                "page": page,
                "per_page": per_page,
                "total_pages": 0
            }
        
        # Get paginated trades
        result = await asyncio.to_thread(
            get_trades_paginated,
            processed_trades,
            page,
            per_page
        )
        
        return {
            "wallet": address,
            "timestamp": datetime.utcnow().isoformat(),
            **result,
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }
        
    except Exception as e:
        logger.error(f"Error processing trades pagination: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wallet/{address}/holdings")
async def get_wallet_holdings_paginated(
    address: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(9, ge=1, le=50, description="Items per page"),
    refresh: bool = Query(False, description="Force refresh from APIs")
):
    """
    Get paginated current holdings with PnL data.
    Only returns tokens with current balance > 0.
    
    - **address**: Solana wallet address
    - **page**: Page number (starts at 1)
    - **per_page**: Number of items per page (max 50)
    - **refresh**: Force refresh data from APIs instead of using cache
    """
    try:
        logger.info(f"Holdings paginated request for wallet: {address[:8]}... (page: {page}, per_page: {per_page})")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address)
            balances, _ = await asyncio.to_thread(get_current_balances_from_helius_rpc, address, False)
        
        if trades is None or balances is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        # Process trades
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "message": "No valid trades found for this wallet",
                "holdings": [],
                "total_holdings": 0,
                "page": page,
                "per_page": per_page,
                "total_pages": 0
            }
        
        # Get paginated holdings
        result = await asyncio.to_thread(
            get_holdings_paginated,
            processed_trades,
            balances,
            page,
            per_page
        )
        
        return {
            "wallet": address,
            "timestamp": datetime.utcnow().isoformat(),
            **result,
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }
        
    except Exception as e:
        logger.error(f"Error processing holdings pagination: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wallet/{address}/full-analysis")
@app.get("/wallet/{address}/overview-analysis")
async def get_or_poll_full_analysis(
    address: str,
    jobId: Optional[str] = Query(None, description="Job ID to poll for results"),
    interval_days: int = Query(7, ge=0),
    fast_seconds: int = Query(5, ge=1, le=3600),
    token_limit: int = Query(10, ge=1, le=100),
    refresh: bool = Query(False)
):
    """
    GET endpoint that either:
    1. Polls an existing job if jobId is provided
    2. Returns cached/live data immediately if no jobId
    """
    
    # If jobId provided, poll that job
    if jobId:
        job_data = job_results.get(jobId)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job ID not found")
            
        if job_data["status"] == "completed":
            return job_data["result"]
        
        elif job_data["status"] == "failed":
            raise HTTPException(status_code=500, detail=job_data.get("error", "Job failed"))
        
        else:
            # Still processing - return 202 with status
            return JSONResponse(
                status_code=202,
                content={
                    "jobId": jobId,
                    "status": job_data["status"],
                    "message": "Analysis still in progress..."
                }
            )

    # No jobId - run synchronously (or return cached if available)
    else:
        logger.info(f"Synchronous full analysis request for wallet: {address[:8]}...")
        service_status["wallet_requests"] += 1
        service_status["last_wallet_request"] = datetime.utcnow().isoformat()
        
        # Fetch data
        if SUPABASE_AVAILABLE:
            balances, trades = await asyncio.to_thread(get_wallet_data, address, refresh)
        else:
            trades = await asyncio.to_thread(get_dex_trades_from_moralis, address)
            balances, _ = await asyncio.to_thread(get_current_balances_from_helius_rpc, address, False)
        
        if trades is None or balances is None:
            raise HTTPException(status_code=500, detail="Failed to fetch wallet data")
        
        processed_trades = await asyncio.to_thread(process_trade_data, trades)
        
        if processed_trades is None or processed_trades.empty:
            return {
                "wallet": address,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "No valid trades found",
                "analysis": {
                    "pnl_summary": {},
                    "behavior_metrics": {},
                    "phishing_panel": "No data",
                    "pnl_distribution": {},
                    "total_tokens": 0
                }
            }
        
        # Run all analysis WITH interval_days passed to ALL functions
        pnl_summary, behavior_metrics, distribution, breakdown = await asyncio.gather(
            asyncio.to_thread(get_overall_pnl_summary, processed_trades, balances, interval_days),
            asyncio.to_thread(compute_behavior_metrics, processed_trades, fast_seconds, interval_days),
            asyncio.to_thread(get_pnl_distribution, processed_trades, interval_days),
            asyncio.to_thread(get_pnl_breakdown_per_token, processed_trades, balances, interval_days)
        )
        
        phishing_panel = await asyncio.to_thread(format_behavior_panel, behavior_metrics)
        
        return {
            "wallet": address,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {
                "pnl_summary": pnl_summary,
                "behavior_metrics": behavior_metrics,
                "phishing_panel": phishing_panel,
                "pnl_distribution": distribution,
                "total_tokens": len(breakdown) if isinstance(breakdown, dict) else 0
            },
            "parameters": {
                "interval_days": interval_days,
                "fast_seconds": fast_seconds,
                "token_limit": token_limit
            },
            "data_source": "cache" if not refresh and SUPABASE_AVAILABLE else "live"
        }

# ==================== ML PREDICTOR ENDPOINTS ====================

@app.get("/token/{mint}/predict")
async def predict_token(
    mint: str,
    threshold: float = Query(0.70, ge=0.0, le=1.0, description="Probability threshold for BUY signal")
):
    """
    Predict if a token will reach 50% gain using ML model
    
    - **mint**: Token mint address
    - **threshold**: Probability threshold for BUY recommendation (default 0.70)
    
    Returns:
    - action: BUY/CONSIDER/SKIP/AVOID
    - win_probability: Probability of 50%+ gain (0-1)
    - confidence: HIGH/MEDIUM/LOW/VERY LOW
    - risk_tier: Risk assessment
    - key_metrics: Important token metrics
    - warnings: List of risk warnings
    """
    if not ML_PREDICTOR_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ML Predictor service not available. Ensure models are trained and saved in 'models/' directory."
        )
    
    try:
        logger.info(f"ML prediction request for token: {mint[:8]}... (threshold: {threshold})")
        service_status["ml_predictions"] += 1
        service_status["last_ml_prediction"] = datetime.utcnow().isoformat()
        
        # Run prediction in thread pool to avoid blocking
        result = await asyncio.to_thread(
            ml_predictor.predict,
            mint,
            threshold
        )
        
        # Check if prediction failed
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return {
            "mint": mint,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": result,
            "model_info": {
                "test_auc": ml_predictor.metadata['performance']['test_auc'],
                "cv_auc": ml_predictor.metadata['performance']['cv_auc_mean'],
                "features_used": len(ml_predictor.selected_features),
                "trained_at": ml_predictor.metadata['trained_at']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ML prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/token/predict/batch")
async def predict_tokens_batch(
    mints: list[str],
    threshold: float = Query(0.70, ge=0.0, le=1.0, description="Probability threshold for BUY signal")
):
    """
    Predict multiple tokens at once (max 10)
    
    - **mints**: List of token mint addresses (max 10)
    - **threshold**: Probability threshold for BUY recommendation
    
    Returns predictions for all tokens
    """
    if not ML_PREDICTOR_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ML Predictor service not available"
        )
    
    if len(mints) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 tokens per batch request"
        )
    
    try:
        logger.info(f"Batch ML prediction request for {len(mints)} tokens")
        service_status["ml_predictions"] += len(mints)
        service_status["last_ml_prediction"] = datetime.utcnow().isoformat()
        
        # Run predictions concurrently
        tasks = [
            asyncio.to_thread(ml_predictor.predict, mint, threshold)
            for mint in mints
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        predictions = []
        for mint, result in zip(mints, results):
            if isinstance(result, Exception):
                predictions.append({
                    "mint": mint,
                    "error": str(result),
                    "success": False
                })
            elif 'error' in result:
                predictions.append({
                    "mint": mint,
                    "error": result['error'],
                    "success": False
                })
            else:
                predictions.append({
                    "mint": mint,
                    "prediction": result,
                    "success": True
                })
        
        # Summary statistics
        successful = sum(1 for p in predictions if p.get('success', False))
        buy_signals = sum(
            1 for p in predictions 
            if p.get('success') and p['prediction']['action'] == 'BUY'
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tokens": len(mints),
            "successful_predictions": successful,
            "failed_predictions": len(mints) - successful,
            "buy_signals": buy_signals,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/ml/status")
async def ml_predictor_status():
    """
    Get ML predictor service status and model information
    """
    if not ML_PREDICTOR_AVAILABLE:
        return {
            "available": False,
            "message": "ML Predictor not loaded. Ensure models are trained and saved."
        }
    
    return {
        "available": True,
        "model_info": {
            "type": ml_predictor.metadata['model_type'],
            "trained_at": ml_predictor.metadata['trained_at'],
            "training_samples": ml_predictor.metadata['training_samples'],
            "test_samples": ml_predictor.metadata['test_samples'],
            "features_total": len(ml_predictor.metadata['all_features']),
            "features_selected": len(ml_predictor.metadata['selected_features']),
        },
        "performance": {
            "test_auc": ml_predictor.metadata['performance']['test_auc'],
            "cv_auc_mean": ml_predictor.metadata['performance']['cv_auc_mean'],
            "cv_auc_std": ml_predictor.metadata['performance']['cv_auc_std'],
        },
        "ensemble_weights": ml_predictor.metadata['ensemble_weights'],
        "usage_stats": {
            "total_predictions": service_status["ml_predictions"],
            "last_prediction": service_status["last_ml_prediction"]
        }
    }


@app.get("/ml/features")
async def get_model_features():
    """
    Get list of features used by the ML model
    """
    if not ML_PREDICTOR_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ML Predictor not available"
        )
    
    # Get feature importance if available
    feature_importance = ml_predictor.metadata.get('feature_importance', {})
    
    # Sort by importance
    features_sorted = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "total_features": len(ml_predictor.metadata['all_features']),
        "selected_features": len(ml_predictor.metadata['selected_features']),
        "core_features": ml_predictor.metadata.get('core_features', []),
        "derived_features": ml_predictor.metadata.get('derived_features', []),
        "top_10_important_features": [
            {"name": name, "importance": importance}
            for name, importance in features_sorted[:10]
        ],
        "all_selected_features": ml_predictor.metadata['selected_features']
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting web service on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )