
#!/usr/bin/env python3
"""
bot.py - Enhanced Telegram Bot with Supabase Integration
- Downloads overlap_results.pkl from Supabase at startup
- Uploads state files to Supabase with overwrite support
- Tracks market cap/FDV, alerts, user prefs, stats
"""

import os
import time
import asyncio
import logging
import joblib
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from supabase_utils import (
    download_overlap_results,
    upload_file,
)

# Load environment variables early
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Paths
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
OVERLAP_FILE = DATA_DIR / "overlap_results.pkl"

# Telegram imports (python-telegram-bot v20+)
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    Defaults,
)

# ========== Functions ==========

def load_latest_tokens_from_overlap() -> Dict[str, Dict[str, Any]]:
    """Download and load overlap_results.pkl from Supabase."""
    logging.info("Downloading latest overlap_results.pkl from Supabase...")
    download_overlap_results(str(OVERLAP_FILE))

    logging.info("Looking for overlap file at: %s", OVERLAP_FILE.absolute())
    if not OVERLAP_FILE.exists():
        logging.error("overlap_results.pkl does not exist.")
        return {}
    if OVERLAP_FILE.stat().st_size == 0:
        logging.error("overlap_results.pkl is empty!")
        return {}

    try:
        data = joblib.load(OVERLAP_FILE)
        logging.info("Loaded data keys: %s", list(data.keys())[:5])
        latest_tokens = {}
        for token_id, history in data.items():
            if not history:
                continue
            latest_check = history[-1]
            result = latest_check.get("result", {})
            latest_tokens[token_id] = {
                "grade": result.get("grade", "NONE"),
                "token_metadata": {
                    "mint": token_id,
                    "name": result.get("token_metadata", {}).get("name"),
                    "symbol": result.get("token_metadata", {}).get("symbol", "")
                },
                "overlap_percentage": result.get("overlap_percentage", 0.0),
                "concentration": result.get("concentration", 0.0),
                "checked_at": result.get("checked_at")
            }
        return latest_tokens
    except Exception as e:
        logging.exception("Failed to load overlap_results.pkl: %s", e)
        return {}

# Example handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is running! Monitoring tokens...")

# Background alert loop
async def background_loop(app: Application):
    logging.info("Background alert loop started...")
    while True:
        # Here you would load tokens and send alerts
        tokens = load_latest_tokens_from_overlap()
        logging.info("DEBUG: Loaded %d tokens from overlap_results.pkl", len(tokens))
        await asyncio.sleep(60)  # Check every minute

# ========== Main ==========

def main():
    logging.info("Starting enhanced telegram bot with market cap tracking...")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

    defaults = Defaults(parse_mode="HTML")
    app = Application.builder().token(token).defaults(defaults).build()

    app.add_handler(CommandHandler("start", start))

    # Start background loop
    app.create_task(background_loop(app))

    logging.info("Bot startup complete. Monitoring for token alerts...")
    app.run_polling()

if __name__ == "__main__":
    main()
