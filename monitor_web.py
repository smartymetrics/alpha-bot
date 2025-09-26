#!/usr/bin/env python3
"""
monitor_web.py - Web service wrapper for token_monitor.py
Keeps the monitor running 24/7 on Render free tier
"""
import os
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Import the main monitoring loop
from token_monitor import main_loop

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Token Monitor Service")

# Track service status
service_status = {
    "started_at": None,
    "last_activity": None,
    "monitor_running": False,
    "error": None
}

@app.on_event("startup")
async def startup_event():
    """Start the token monitor in background on service startup"""
    service_status["started_at"] = datetime.utcnow().isoformat()
    service_status["monitor_running"] = True
    
    logger.info("Starting token monitor background task...")
    
    # Run main_loop as a background task
    asyncio.create_task(run_monitor())

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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Token Monitor",
        "status": "running" if service_status["monitor_running"] else "stopped",
        "uptime_start": service_status["started_at"]
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
        "service": "Token Monitor Web Service",
        "monitor_status": service_status,
        "environment": {
            "render": os.getenv("RENDER") is not None,
            "port": os.getenv("PORT", "not set")
        }
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