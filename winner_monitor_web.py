# winner_monitor_web.py
"""
winner_monitor_web.py - Dedicated web service for winner_monitor.py
Keeps the winner monitor running 24/7 on Render free tier
Separate from main token monitor service
"""
import os
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the winner monitor main loop
from winner_monitor import main

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Track service status
service_status = {
    "started_at": None,
    "last_activity": None,
    "winner_monitor_running": False,
    "error": None,
    "total_restarts": 0,
    "last_restart": None
}

async def run_winner_monitor():
    """Wrapper to run winner monitor and handle errors"""
    try:
        logger.info("Starting winner monitor main loop...")
        await main()
    except Exception as e:
        logger.error(f"Winner monitor crashed: {e}", exc_info=True)
        service_status["winner_monitor_running"] = False
        service_status["error"] = str(e)
        service_status["total_restarts"] += 1
        service_status["last_restart"] = datetime.utcnow().isoformat()
        
        # Attempt to restart after 10 seconds
        await asyncio.sleep(10)
        logger.info("Attempting to restart winner monitor...")
        service_status["winner_monitor_running"] = True
        service_status["error"] = None
        asyncio.create_task(run_winner_monitor())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    service_status["started_at"] = datetime.utcnow().isoformat()
    service_status["winner_monitor_running"] = True

    logger.info("🚀 Starting winner monitor background task...")

    # Run winner monitor as background task
    winner_monitor_task = asyncio.create_task(run_winner_monitor())

    yield

    # Shutdown
    logger.info("🛑 Shutting down winner monitor...")
    winner_monitor_task.cancel()
    
    try:
        await winner_monitor_task
    except asyncio.CancelledError:
        logger.info("Winner monitor task cancelled successfully")
        pass

app = FastAPI(
    title="Winner Monitor Service",
    description="24/7 monitoring service for winning tokens and wallets",
    version="1.0.0",
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

# ==================== Service Endpoints ====================

@app.head("/")
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Winner Monitor Service",
        "status": "running" if service_status["winner_monitor_running"] else "stopped",
        "uptime_start": service_status["started_at"],
        "description": "24/7 monitoring service for winning tokens and smart wallets",
        "features": [
            "Real-time winner token detection",
            "Smart wallet tracking",
            "Automated alerts and notifications",
            "Performance monitoring"
        ],
        "endpoints": {
            "service": ["/", "/health", "/status"],
            "monitoring": ["/winners/recent", "/winners/stats"]
        }
    }


@app.head("/health")
@app.get("/health")
async def health():
    """Health check endpoint for UptimeRobot and monitoring"""
    service_status["last_activity"] = datetime.utcnow().isoformat()
    
    is_healthy = service_status["winner_monitor_running"]
    
    return JSONResponse(
        status_code=200 if is_healthy else 503,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "winner_monitor_running": service_status["winner_monitor_running"],
            "started_at": service_status["started_at"],
            "last_ping": service_status["last_activity"],
            "uptime_seconds": _calculate_uptime(),
            "error": service_status["error"]
        }
    )


@app.get("/status")
async def status():
    """Detailed status endpoint"""
    return {
        "service": "Winner Monitor Service",
        "monitor_status": {
            "running": service_status["winner_monitor_running"],
            "started_at": service_status["started_at"],
            "last_activity": service_status["last_activity"],
            "uptime_seconds": _calculate_uptime(),
            "total_restarts": service_status["total_restarts"],
            "last_restart": service_status["last_restart"],
            "current_error": service_status["error"]
        },
        "environment": {
            "render": os.getenv("RENDER") is not None,
            "port": os.getenv("PORT", "not set"),
            "python_version": os.sys.version
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/winners/stats")
async def get_winner_stats():
    """
    Get statistics about monitored winners
    This is a placeholder - implement based on your winner_monitor.py data structure
    """
    return {
        "message": "Winner statistics endpoint",
        "note": "Implement this based on your winner_monitor.py data storage",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/winners/recent")
async def get_recent_winners(
    limit: int = Query(10, ge=1, le=100, description="Number of recent winners to return")
):
    """
    Get recent winner tokens
    This is a placeholder - implement based on your winner_monitor.py data structure
    """
    return {
        "message": "Recent winners endpoint",
        "limit": limit,
        "note": "Implement this based on your winner_monitor.py data storage",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/monitor/restart")
async def restart_monitor():
    """
    Manually restart the winner monitor
    Useful for remote maintenance
    """
    if not service_status["winner_monitor_running"]:
        service_status["winner_monitor_running"] = True
        service_status["error"] = None
        service_status["total_restarts"] += 1
        service_status["last_restart"] = datetime.utcnow().isoformat()
        
        asyncio.create_task(run_winner_monitor())
        
        return {
            "message": "Winner monitor restart initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "message": "Winner monitor is already running",
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/monitor/logs")
async def get_monitor_logs(
    lines: int = Query(50, ge=1, le=500, description="Number of log lines to return")
):
    """
    Get recent monitor logs
    This is a placeholder - implement based on your logging setup
    """
    return {
        "message": "Monitor logs endpoint",
        "lines": lines,
        "note": "Implement this based on your logging configuration",
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== Helper Functions ====================

def _calculate_uptime() -> int:
    """Calculate uptime in seconds"""
    if service_status["started_at"]:
        started = datetime.fromisoformat(service_status["started_at"])
        now = datetime.utcnow()
        return int((now - started).total_seconds())
    return 0


# ==================== Error Handlers ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The endpoint {request.url.path} does not exist",
            "available_endpoints": [
                "/", "/health", "/status", "/winners/stats", "/winners/recent"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please check the logs.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==================== Startup ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  # Default to 8001 to avoid conflict
    logger.info(f"🚀 Starting Winner Monitor web service on port {port}")
    logger.info(f"📊 Service will run 24/7 and auto-restart on failures")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )