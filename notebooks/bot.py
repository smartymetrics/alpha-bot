#!/usr/bin/env python3
"""
unified_bot_corrected.py

Full corrected unified Telegram bot (local-first).
- Reads overlap_results.pkl from ./data (local)
- Runs background loop to detect new/changed token grades and send alerts
- Persists user prefs, stats, alerts_state locally (./data/*.pkl)
- DOES NOT upload/download to Supabase inside the background loop.
- Supabase usage is optional and controlled by env vars:
    USE_SUPABASE (default: false) - when true the bot can do a single startup download
    DOWNLOAD_OVERLAP_ON_STARTUP (default: false) - if true and USE_SUPABASE, download overlap file once at startup
    SUPABASE_DAILY_SYNC (default: false) - if true and USE_SUPABASE, upload bot-state once per day
- Default behaviour is local-only (no Supabase activity).
"""

import os
import asyncio
import logging
import joblib
import requests
from threading import Lock
from pathlib import Path
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    Defaults,
)

# Add this to the top of your unified_bot_corrected.py after the imports

import platform

# Detect if running on Railway
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
IS_LOCAL = platform.system() in ['Windows', 'Darwin'] and not IS_RAILWAY

# Configure data directory based on environment
if IS_RAILWAY:
    # Railway persistent volume mount point
    DATA_DIR = Path("/data")
    POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "300"))  # 5 min for Railway
else:
    # Local development
    DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
    POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "60"))

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Optional supabase helpers (import safe; functions will only be used when configured)
try:
    from supabase_utils import (
        download_overlap_results,
        upload_file,
        download_file,
    )
except Exception:
    # If supabase_utils is missing or broken, we still run in local-only mode.
    download_overlap_results = None
    upload_file = None
    download_file = None

# Load env
load_dotenv()

# ----------------------
# Config
# ----------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "monitor-data")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

OVERLAP_FILE = DATA_DIR / "overlap_results.pkl"
USER_PREFS_FILE = DATA_DIR / "bot_user_prefs.pkl"
USER_STATS_FILE = DATA_DIR / "bot_user_stats.pkl"
ALERTS_STATE_FILE = DATA_DIR / "bot_alerts_state.pkl"

ALL_GRADES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "60"))

# Supabase behaviour flags (opt-in)
USE_SUPABASE = True # os.getenv("USE_SUPABASE", "false").lower() in ("1", "true", "yes")
DOWNLOAD_OVERLAP_ON_STARTUP = True # os.getenv("DOWNLOAD_OVERLAP_ON_STARTUP", "false").lower() in ("1", "true", "yes")
SUPABASE_DAILY_SYNC = True # os.getenv("SUPABASE_DAILY_SYNC", "false").lower() in ("1", "true", "yes")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------
# Thread-safe joblib file access
# ----------------------
file_lock = Lock()

def safe_load(path: Path, default):
    with file_lock:
        try:
            if not path.exists():
                return default
            return joblib.load(path)
        except Exception as e:
            logging.exception("Failed loading %s: %s", path, e)
            return default

def safe_save(path: Path, data):
    with file_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(data, path)
        except Exception as e:
            logging.exception("Failed saving %s: %s", path, e)

# ----------------------
# Supabase sync functions (opt-in)
# ----------------------
_last_upload = 0
def upload_bot_data_to_supabase():
    global _last_upload
    now = time.time()
    if not USE_SUPABASE or upload_file is None:
        logging.debug("Supabase upload skipped (disabled or helper missing).")
        return
    for file in [USER_PREFS_FILE, USER_STATS_FILE, ALERTS_STATE_FILE]:
        if file.exists():
            try:
                if now - _last_upload < 43200:  # only allow once every 12 hrs
                    return
                upload_file(str(file), bucket=BUCKET_NAME)
                _last_upload = now
            except Exception as e:
                logging.exception("Failed to upload %s to Supabase: %s", file, e)

def download_bot_data_from_supabase():
    if not USE_SUPABASE or download_file is None:
        logging.debug("Supabase download skipped (disabled or helper missing).")
        return
    for file in [USER_PREFS_FILE, USER_STATS_FILE, ALERTS_STATE_FILE]:
        try:
            download_file(str(file), os.path.basename(file), bucket=BUCKET_NAME)
        except Exception as e:
            logging.debug("Could not download %s from Supabase: %s", file, e)

async def daily_supabase_sync():
    if not (USE_SUPABASE and SUPABASE_DAILY_SYNC):
        logging.debug("Daily Supabase sync disabled by configuration.")
        return
    logging.info("Daily Supabase sync task started.")
    while True:
        try:
            upload_bot_data_to_supabase()
            logging.info("✅ Daily sync with Supabase complete")
        except Exception as e:
            logging.exception("Supabase daily sync failed: %s", e)
        await asyncio.sleep(24 * 3600)  # wait 1 day

# ----------------------
# Market cap utilities
# ----------------------
def format_marketcap_display(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:,.0f}"

def fetch_marketcap_and_fdv(mint: str) -> Tuple[Optional[float], Optional[float], str]:
    try:
        if not mint:
            return None, None, "N/A"
        url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None, None, "N/A"
        data = resp.json()
        pairs = data.get("pairs", [])
        if not pairs:
            return None, None, "N/A"
        mc = pairs[0].get("marketCap")
        fdv = pairs[0].get("fdv")
        display = format_marketcap_display(mc if mc else fdv)
        return mc, fdv, display
    except Exception as e:
        logging.error(f"Error fetching marketcap for {mint}: {e}")
        return None, None, "N/A"

# ----------------------
# User Manager
# ----------------------
class UserManager:
    def __init__(self, prefs_file: Path, stats_file: Path):
        self.prefs_file = prefs_file
        self.stats_file = stats_file

    def get_user_prefs(self, chat_id: str) -> Dict[str, Any]:
        prefs = safe_load(self.prefs_file, {})
        return prefs.get(chat_id, {
            "grades": [],
            "created_at": self.now_iso(),
            "updated_at": self.now_iso(),
            "active": False,             # not active until admin adds them
            "subscribed": False,         # new flag
            "total_alerts_received": 0,
            "last_alert_at": None,
            "expires_at": None
        })


    def update_user_prefs(self, chat_id: str, updates: Dict[str, Any]) -> bool:
        try:
            prefs = safe_load(self.prefs_file, {})
            if chat_id not in prefs:
                prefs[chat_id] = {
                    "grades": ALL_GRADES.copy(),
                    "created_at": self.now_iso(),
                    "active": True,
                    "total_alerts_received": 0
                }
            prefs[chat_id].update(updates)
            prefs[chat_id]["updated_at"] = self.now_iso()
            safe_save(self.prefs_file, prefs)
            # Note: not uploading to Supabase immediately; daily sync handles that if enabled
            return True
        except Exception as e:
            logging.exception("Failed to update user prefs for %s: %s", chat_id, e)
            return False

    def deactivate_user(self, chat_id: str) -> bool:
        return self.update_user_prefs(chat_id, {"active": False, "deactivated_at": self.now_iso()})

    def activate_user(self, chat_id: str) -> bool:
        return self.update_user_prefs(chat_id, {"active": True, "reactivated_at": self.now_iso()})

    def get_active_users(self) -> Dict[str, Dict[str, Any]]:
        prefs = safe_load(self.prefs_file, {})
        return {k: v for k, v in prefs.items() if v.get("active", True)}

    def get_user_stats(self, chat_id: str) -> Dict[str, Any]:
        stats = safe_load(self.stats_file, {})
        return stats.get(chat_id, {
            "alerts_received": 0,
            "last_alert_at": None,
            "joined_at": None,
            "grade_breakdown": {g: 0 for g in ALL_GRADES}
        })

    def update_user_stats(self, chat_id: str, grade: str = None):
        try:
            stats = safe_load(self.stats_file, {})
            if chat_id not in stats:
                stats[chat_id] = {
                    "alerts_received": 0,
                    "last_alert_at": None,
                    "joined_at": self.now_iso(),
                    "grade_breakdown": {g: 0 for g in ALL_GRADES}
                }
            stats[chat_id]["alerts_received"] += 1
            stats[chat_id]["last_alert_at"] = self.now_iso()
            if grade and grade in stats[chat_id]["grade_breakdown"]:
                stats[chat_id]["grade_breakdown"][grade] += 1
            safe_save(self.stats_file, stats)
            # Do not upload immediately; daily sync will handle uploads if enabled
        except Exception as e:
            logging.exception("Failed to update stats for %s: %s", chat_id, e)

    def get_all_stats(self) -> Dict[str, Any]:
        prefs = safe_load(self.prefs_file, {})
        stats = safe_load(self.stats_file, {})
        total_users = len(prefs)
        active_users = len([u for u in prefs.values() if u.get("active", True)])
        total_alerts = sum(s.get("alerts_received", 0) for s in stats.values())
        grade_totals = {g: 0 for g in ALL_GRADES}
        for user_stats in stats.values():
            for grade, count in user_stats.get("grade_breakdown", {}).items():
                if grade in grade_totals:
                    grade_totals[grade] += count
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_alerts_sent": total_alerts,
            "grade_breakdown": grade_totals,
            "generated_at": self.now_iso()
        }
    
    def mark_notified(self, chat_id: str):
        """Mark that an expired user has been notified this month."""
        prefs = safe_load(self.prefs_file, {})
        if chat_id in prefs:
            prefs[chat_id]["last_notified"] = self.now_iso()
            safe_save(self.prefs_file, prefs)
            upload_bot_data_to_supabase()

    def add_user_with_expiry(self, chat_id: str, days_valid: int):
        """Add or update a user with subscription expiry."""
        try:
            chat_id = str(chat_id)  # Ensure string
            prefs = safe_load(self.prefs_file, {})
            now = self.now_iso()
            expiry_date = (datetime.utcnow() + timedelta(days=days_valid)).replace(microsecond=0).isoformat() + "Z"

            if chat_id not in prefs:
                prefs[chat_id] = {
                    "grades": ALL_GRADES.copy(),
                    "created_at": now,
                    "total_alerts_received": 0
                }

            # Update user data
            prefs[chat_id].update({
                "updated_at": now,
                "expires_at": expiry_date,
                "active": True,
                "subscribed": True  # This is key!
            })
            
            # Save to file immediately
            safe_save(self.prefs_file, prefs)
            logging.info(f"✅ Saved user {chat_id} to file with subscribed=True, expires_at={expiry_date}")
            
            # Verify the save worked
            verify_prefs = safe_load(self.prefs_file, {})
            verify_user = verify_prefs.get(chat_id, {})
            logging.info(f"🔍 Verification - User {chat_id}: subscribed={verify_user.get('subscribed')}, active={verify_user.get('active')}")
            
            # Optional: upload to Supabase if enabled
            if USE_SUPABASE:
                upload_bot_data_to_supabase()
            
            return expiry_date
            
        except Exception as e:
            logging.exception(f"❌ Error in add_user_with_expiry for {chat_id}: {e}")
            raise


    def is_subscription_expired(self, chat_id: str) -> bool:
        """Check if a user's subscription has expired (admin never expires)."""
        ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")
        if ADMIN_USER_ID and str(chat_id) == ADMIN_USER_ID:
            return False  # admin never expires

        prefs = safe_load(self.prefs_file, {})
        user = prefs.get(chat_id)
        if not user:
            return True  # not found = expired

        expires_at = user.get("expires_at")
        if not expires_at:
            return False  # no expiry = lifetime user

        try:
            expiry_date = datetime.fromisoformat(expires_at.rstrip("Z"))
            return datetime.utcnow() > expiry_date
        except Exception as e:
            logging.warning(f"⚠️ Could not parse expiry for {chat_id}: {e}")
            return False


        # safe_save(self.prefs_file, prefs)
        # upload_bot_data_to_supabase()
        # logging.info(f"✅ Added/updated user {chat_id} with expiry {expiry_date}")
        # return expiry_date


    def save_to_supabase(self):
        # helper that respects USE_SUPABASE
        upload_bot_data_to_supabase()

    @staticmethod
    def now_iso():
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

user_manager = UserManager(USER_PREFS_FILE, USER_STATS_FILE)

# ----------------------
# Load overlap file (local-only)
# ----------------------
def load_latest_tokens_from_overlap() -> Dict[str, Dict[str, Any]]:
    """
    Load overlap_results.pkl from local disk only.
    If you want to pull from Supabase once at startup, set USE_SUPABASE and DOWNLOAD_OVERLAP_ON_STARTUP.
    """
    if not OVERLAP_FILE.exists() or OVERLAP_FILE.stat().st_size == 0:
        return {}

    try:
        data = joblib.load(OVERLAP_FILE)
        latest_tokens = {}
        for token_id, history in data.items():
            if not history:
                continue
            result = history[-1].get("result", {})
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
        logging.exception("Failed to load overlap file: %s", e)
        return {}

# ----------------------
# Alert formatting & sending
# ----------------------
def format_alert_html(
    token_data,
    alert_type,
    previous_grade=None,
    initial_mc=None,
    initial_fdv=None,
    first_alert_at=None
):
    token_meta = token_data.get("token_metadata") or {}
    name = token_meta.get("name") or token_data.get("token") or "Unknown"
    symbol = token_meta.get("symbol") or ""
    grade = token_data.get("grade", "NONE")
    mint = token_meta.get("mint", "") or token_data.get("token", "")

    current_mc, current_fdv, current_display = fetch_marketcap_and_fdv(mint)

    # Only include "was ..." if NOT the first alert (i.e., CHANGE alerts)
    if alert_type == "CHANGE" and current_mc is not None and initial_mc is not None:
        mc_line = f"💰 <b>Market Cap:</b> {format_marketcap_display(current_mc)} (was {format_marketcap_display(initial_mc)})"
    elif alert_type == "CHANGE" and current_fdv is not None and initial_fdv is not None:
        mc_line = f"🏷️ <b>FDV:</b> {format_marketcap_display(current_fdv)} (was {format_marketcap_display(initial_fdv)})"
    else:
        mc_line = f"💰 <b>Market Cap/FDV:</b> {current_display}"

    lines = [
        "🚀 <b>New Token Detected</b>" if alert_type == "NEW" else "🔁 <b>Grade Changed</b>",
        f"<b>{name}</b> ({symbol})" if symbol else f"<b>{name}</b>",
        f"<b>Grade:</b> {grade}" + (f" (was {previous_grade})" if previous_grade and alert_type == "CHANGE" else ""),
        mc_line,
        f"<b>Overlap:</b> {token_data.get('overlap_percentage')}%",
        f"<b>Concentration:</b> {token_data.get('concentration')}%"
    ]

    if first_alert_at:
        lines.append(f"🕒 <b>First alert:</b> {first_alert_at[:10]}")

    lines.append("")
    if mint:
        lines.append(f'<a href="https://solscan.io/token/{mint}">Solscan</a> | <a href="https://gmgn.ai/sol/token/{mint}">GMGN</a>')

    return "\n".join(lines)


async def testalert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a test alert for a known token. Supports NEW and CHANGE alerts for all grades."""
    chat_id = str(update.effective_chat.id)
    args = context.args or []
    alert_type = "CHANGE"  # Default to CHANGE
    grade = "CRITICAL"     # Default to CRITICAL
    previous_grade = "HIGH" if alert_type == "CHANGE" else None

    # Parse arguments if provided
    if args:
        if args[0].upper() in ["NEW", "CHANGE"]:
            alert_type = args[0].upper()
        if len(args) > 1 and args[1].upper() in ALL_GRADES:
            grade = args[1].upper()
        if alert_type == "CHANGE" and len(args) > 2 and args[2].upper() in ALL_GRADES:
            previous_grade = args[2].upper()

    token_id = "G8cGYUUdnwvQ8W1iMy37TMD2xpMnYS4NCh1YKQJepump"  # Example token
    try:
        # Query DexScreener for token info
        resp = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{token_id}", timeout=10)
        data = resp.json()
        pairs = data.get("pairs", []) or []
        mc = None
        fdv = None
        base = {}
        if pairs:
            mc = pairs[0].get("marketCap")
            fdv = pairs[0].get("fdv")
            base = pairs[0].get("baseToken", {}) or {}

        token_data = {
            "token": token_id,
            "grade": grade,
            "token_metadata": {
                "name": base.get("name", "TestToken"),
                "symbol": base.get("symbol", "TT"),
            },
            "overlap_percentage": 85.3,
            "concentration": 42.1
        }

        # Simulate initial values for test
        initial_mc = mc * 0.6 if mc else None  # Simulate 40% growth
        initial_fdv = fdv * 0.6 if fdv else None
        first_alert = (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z"

        message = format_alert_html(
            token_data,
            alert_type,
            previous_grade=previous_grade if alert_type == "CHANGE" else None,
            initial_mc=initial_mc,
            initial_fdv=initial_fdv,
            first_alert_at=first_alert
        )

        await update.message.reply_html(f"🔔 Test Alert ({alert_type})\n\n{message}")
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to fetch token data: {e}")

async def send_alert_to_subscribers(
    app: Application,
    token_data: Dict[str, Any],
    grade: str,
    previous_grade: str = None,
    initial_mc: Optional[float] = None,
    initial_fdv: Optional[float] = None,
    first_alert_at: Optional[str] = None
):
    """
    Send an alert to every active user who subscribes to this grade.
    """
    active_users = user_manager.get_active_users()
    if not active_users:
        logging.debug("No active users to send alerts to.")
        return

    message = format_alert_html(
        token_data,
        "CHANGE" if previous_grade else "NEW",
        previous_grade,
        initial_mc=initial_mc,
        initial_fdv=initial_fdv,
        first_alert_at=first_alert_at
    )

    for chat_id, prefs in active_users.items():
        subscribed_grades = prefs.get("grades", ALL_GRADES.copy())
        if isinstance(subscribed_grades, (list, tuple)):
            if grade not in subscribed_grades:
                continue
        else:
            if grade not in ALL_GRADES:
                continue

        try:
            await app.bot.send_message(
                chat_id=int(chat_id),
                text=message,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            user_manager.update_user_stats(chat_id, grade)
        except Exception as e:
            logging.warning("Failed to send alert to %s: %s", chat_id, e)

        # small delay to avoid hitting rate limits
        await asyncio.sleep(0.1)

async def monthly_expiry_notifier(app: Application):
    """Notify expired users once per month."""
    while True:
        try:
            prefs = safe_load(USER_PREFS_FILE, {})
            for chat_id, user in prefs.items():
                if user_manager.is_subscription_expired(chat_id):
                    last_notified = user.get("last_notified")
                    should_notify = True
                    if last_notified:
                        try:
                            last_dt = datetime.fromisoformat(last_notified.rstrip("Z"))
                            # Only notify if more than 30 days passed
                            if (datetime.utcnow() - last_dt).days < 30:
                                should_notify = False
                        except:
                            pass
                    
                    if should_notify:
                        try:
                            await app.bot.send_message(
                                chat_id=int(chat_id),
                                text="⚠️ Your subscription has expired. Please contact the admin to renew."
                            )
                            user_manager.mark_notified(chat_id)
                            logging.info(f"Notified expired user {chat_id}")
                        except Exception as e:
                            logging.warning(f"Failed to notify expired user {chat_id}: {e}")
        except Exception as e:
            logging.exception("Error in monthly_expiry_notifier: %s", e)

        # Sleep 24h before checking again
        await asyncio.sleep(24 * 3600)

def is_subscribed(chat_id: str) -> bool:
    ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")
    if ADMIN_USER_ID and str(chat_id) == ADMIN_USER_ID:
        return True  # admin bypasses subscription
    
    prefs = safe_load(USER_PREFS_FILE, {})
    user = prefs.get(str(chat_id))  # Ensure chat_id is string
    
    if not user:
        logging.debug(f"User {chat_id} not found in preferences")
        return False
    
    # Check if user has subscribed flag
    if not user.get("subscribed", False):
        logging.debug(f"User {chat_id} not subscribed: {user.get('subscribed', False)}")
        return False
    
    # Check if subscription has expired
    if user_manager.is_subscription_expired(str(chat_id)):
        logging.debug(f"User {chat_id} subscription expired")
        return False
    
    logging.debug(f"User {chat_id} subscription valid")
    return True


# ----------------------
# Background loop
# ----------------------
async def background_loop(app: Application):
    logging.info("Background alert loop started...")
    alerts_state = safe_load(ALERTS_STATE_FILE, {})
    first_run = True
    while True:
        try:
            tokens = load_latest_tokens_from_overlap()
            if first_run:
                logging.info("DEBUG: Loaded %d tokens from overlap_results.pkl", len(tokens))
                sample_items = list(tokens.items())[:3]
                for tid, info in sample_items:
                    logging.info("DEBUG sample token: %s grade=%s checked_at=%s", tid, info.get("grade"), info.get("checked_at"))
                first_run = False

            for token_id, token in tokens.items():
                grade = token.get("grade")
                if not grade:
                    continue

                current_state = alerts_state.get(token_id)
                last_grade = current_state.get("last_grade") if isinstance(current_state, dict) else None

                if grade != last_grade:
                    logging.info("New/changed grade for %s: %s -> %s", token_id, last_grade, grade)

                    if last_grade is None:
                        mc, fdv, _ = fetch_marketcap_and_fdv(token_id)
                        alerts_state[token_id] = {
                            "last_grade": grade,
                            "initial_marketcap": mc,
                            "initial_fdv": fdv,
                            "first_alert_at": datetime.utcnow().isoformat() + "Z"
                        }
                        logging.info("Captured initial market data for %s: MC=%s, FDV=%s",
                                     token_id, format_marketcap_display(mc), format_marketcap_display(fdv))
                    else:
                        alerts_state[token_id]["last_grade"] = grade

                    # use the up-to-date state for sending alert
                    state = alerts_state.get(token_id, {})
                    await send_alert_to_subscribers(
                        app,
                        token,
                        grade,
                        previous_grade=last_grade,
                        initial_mc=state.get("initial_marketcap"),
                        initial_fdv=state.get("initial_fdv"),
                        first_alert_at=state.get("first_alert_at")
                    )

            # Persist alerts_state locally only (no Supabase upload in loop)
            safe_save(ALERTS_STATE_FILE, alerts_state)

        except Exception as e:
            logging.exception("Error in background loop: %s", e)

        await asyncio.sleep(POLL_INTERVAL_SECS)

# ----------------------
# Telegram Commands & Handlers
# ----------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    
    logging.info(f"🚀 User {chat_id} started bot")
    
    # Debug: check subscription status step by step
    prefs = safe_load(USER_PREFS_FILE, {})
    user_data = prefs.get(chat_id, {})
    
    logging.info(f"🔍 Debug user {chat_id}:")
    logging.info(f"  - Found in prefs: {chat_id in prefs}")
    logging.info(f"  - subscribed: {user_data.get('subscribed', False)}")
    logging.info(f"  - active: {user_data.get('active', False)}")
    logging.info(f"  - expires_at: {user_data.get('expires_at')}")
    
    is_sub = is_subscribed(chat_id)
    is_expired = user_manager.is_subscription_expired(chat_id)
    
    logging.info(f"  - is_subscribed(): {is_sub}")
    logging.info(f"  - is_subscription_expired(): {is_expired}")
    
    if not is_sub:
        await update.message.reply_html(
            f"👋 Welcome!\n\n"
            f"❌ You are not subscribed to alerts.\n"
            f"Please contact the admin to activate your subscription.\n\n"
            f"🔍 <b>Debug info:</b>\n"
            f"• User found: {chat_id in prefs}\n"
            f"• Subscribed: {user_data.get('subscribed', False)}\n"
            f"• Active: {user_data.get('active', False)}\n"
            f"• Expired: {is_expired}"
        )
        return

    # Rest of start command...
    user_prefs = user_manager.get_user_prefs(chat_id)

    if not user_prefs.get("created_at"):
        user_manager.update_user_prefs(chat_id, {
            "grades": ALL_GRADES.copy(),
            "active": True,
            "created_at": user_manager.now_iso()
        })
        user_prefs = user_manager.get_user_prefs(chat_id)
    else:
        user_manager.activate_user(chat_id)

    keyboard = [
        [
            InlineKeyboardButton("🔴 CRITICAL Only", callback_data="preset_critical"),
            InlineKeyboardButton("🔥 CRITICAL + HIGH", callback_data="preset_critical_high")
        ],
        [
            InlineKeyboardButton("📊 All Grades", callback_data="preset_all"),
            InlineKeyboardButton("⚙️ Custom Setup", callback_data="custom_setup")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    welcome_msg = (
        f"👋 <b>Welcome to Token Grade Alerts!</b>\n\n"
        f"🎯 Current subscription: <b>{', '.join(user_prefs.get('grades', ALL_GRADES))}</b>\n\n"
        f"This bot monitors new Solana tokens and alerts you when they show "
        f"overlap with yesterday's winning tokens based on holder analysis.\n\n"
        f"<b>Grade Meanings:</b>\n"
        f"🔴 <b>CRITICAL</b> - Very high overlap (50%+ or strong concentration)\n"
        f"🟠 <b>HIGH</b> - Significant overlap (30%+ overlap)\n"
        f"🟡 <b>MEDIUM</b> - Notable overlap (15%+ overlap)\n"
        f"🟢 <b>LOW</b> - Some overlap (5%+ overlap)\n\n"
        f"Choose your alert preferences:"
    )

    await update.message.reply_html(welcome_msg, reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = str(query.from_user.id)
    if not is_subscribed(chat_id):
        await update.message.reply_text("⛔ You are not subscribed. Please contact the admin.")
        return
    await query.answer()

    if query.data == "preset_critical":
        user_manager.update_user_prefs(chat_id, {"grades": ["CRITICAL"]})
        await query.edit_message_text("✅ Preferences updated: CRITICAL only.", parse_mode="HTML")
    elif query.data == "preset_critical_high":
        user_manager.update_user_prefs(chat_id, {"grades": ["CRITICAL", "HIGH"]})
        await query.edit_message_text("✅ Preferences updated: CRITICAL + HIGH.", parse_mode="HTML")
    elif query.data == "preset_all":
        user_manager.update_user_prefs(chat_id, {"grades": ALL_GRADES.copy()})
        await query.edit_message_text("✅ Preferences updated: ALL grades.", parse_mode="HTML")
    elif query.data == "custom_setup":
        await query.edit_message_text(
            "⚙️ Custom Setup\n\nUse /setalerts GRADE1 GRADE2 ...\nAvailable: CRITICAL, HIGH, MEDIUM, LOW",
            parse_mode="HTML"
        )

async def setalerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if not is_subscribed(chat_id):
        await update.message.reply_text("⛔ You are not subscribed. Please contact the admin.")
        return
    args = context.args or []
    valid = set(ALL_GRADES)
    chosen = [a.upper() for a in args if a.upper() in valid]

    if not chosen:
        keyboard = [
            [InlineKeyboardButton("🔴 CRITICAL", callback_data="preset_critical"),
             InlineKeyboardButton("🔥 CRITICAL + HIGH", callback_data="preset_critical_high")],
            [InlineKeyboardButton("📊 All Grades", callback_data="preset_all")]
        ]
        await update.message.reply_html(
            "⚠️ Usage: /setalerts GRADE1 GRADE2 ...\nAvailable: CRITICAL, HIGH, MEDIUM, LOW",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    success = user_manager.update_user_prefs(chat_id, {"grades": chosen, "active": True})
    if success:
        await update.message.reply_html(f"✅ Alert preferences updated! You will receive: <b>{', '.join(chosen)}</b>")
    else:
        await update.message.reply_text("❌ Failed to save preferences. Please try again.")

async def myalerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if not is_subscribed(chat_id):
        await update.message.reply_text("⛔ You are not subscribed. Please contact the admin.")
        return
    prefs = user_manager.get_user_prefs(chat_id)
    stats = user_manager.get_user_stats(chat_id)
    if not prefs.get("active", False):
        await update.message.reply_text("You are not currently subscribed. Use /start to subscribe.")
        return

    total_alerts = stats.get("alerts_received", 0)
    last_alert = stats.get("last_alert_at")
    last_alert_str = "Never" if not last_alert else f"<i>{last_alert[:10]}</i>"

    breakdown_lines = []
    for grade in ALL_GRADES:
        count = stats.get("grade_breakdown", {}).get(grade, 0)
        if count > 0:
            breakdown_lines.append(f"  • {grade}: {count}")
    breakdown_text = "\n".join(breakdown_lines) if breakdown_lines else "  • No alerts received yet"

    msg = (
        f"📊 <b>Your Alert Settings</b>\n\n"
        f"🎯 <b>Subscribed to:</b> {', '.join(prefs.get('grades', ALL_GRADES))}\n"
        f"📈 <b>Total alerts received:</b> {total_alerts}\n"
        f"🕐 <b>Last alert:</b> {last_alert_str}\n\n"
        f"<b>Breakdown by grade:</b>\n{breakdown_text}\n\n"
        f"Use /setalerts to change your preferences."
    )
    await update.message.reply_html(msg)

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    success = user_manager.deactivate_user(chat_id)
    if success:
        await update.message.reply_html("😔 You have been unsubscribed. Use /start to reactivate.")
    else:
        await update.message.reply_text("❌ Failed to unsubscribe. Please try again.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 <b>Token Grade Alerts - Help</b>\n\n"
        "Commands:\n"
        "• /start - Subscribe and set preferences\n"
        "• /setalerts GRADE1 GRADE2 - Set alert grades\n"
        "• /myalerts - View your settings and stats\n"
        "• /stop - Unsubscribe (keeps your data)\n"
        "• /help - Show this help message\n\n"
        "Grades: CRITICAL, HIGH, MEDIUM, LOW\n"
    )
    await update.message.reply_html(help_text)

def is_admin_update(update: Update) -> bool:
    ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")
    if not ADMIN_USER_ID:
        return False
    return str(update.effective_user.id) == ADMIN_USER_ID

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_stats = user_manager.get_user_stats(chat_id)
    msg = (
        f"📊 <b>Your Statistics</b>\n\n"
        f"📬 Total alerts received: <b>{user_stats.get('alerts_received', 0)}</b>\n"
        f"📅 Member since: <i>{user_stats.get('joined_at', 'Unknown')[:10] if user_stats.get('joined_at') else 'Unknown'}</i>\n"
    )
    if is_admin_update(update):
        platform_stats = user_manager.get_all_stats()
        msg += (
            f"\n🏢 <b>Platform Statistics (Admin)</b>\n"
            f"• Total users: <b>{platform_stats['total_users']}</b>\n"
            f"• Active users: <b>{platform_stats['active_users']}</b>\n"
            f"• Total alerts sent: <b>{platform_stats['total_alerts_sent']}</b>\n"
        )
    await update.message.reply_html(msg)

async def admin_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin_update(update):
        await update.message.reply_text("Access denied.")
        return
    platform_stats = user_manager.get_all_stats()
    prefs = safe_load(USER_PREFS_FILE, {})
    inactive_users = len([u for u in prefs.values() if not u.get("active", True)])
    recent_users = len([u for u in prefs.values() if u.get("created_at") and (datetime.utcnow() - datetime.fromisoformat(u["created_at"].rstrip("Z"))).days <= 7])
    msg = (
        f"👑 <b>Admin Dashboard</b>\n\n"
        f"• Total registered: {platform_stats['total_users']}\n"
        f"• Active users: {platform_stats['active_users']}\n"
        f"• Inactive users: {inactive_users}\n"
        f"• New users (7 days): {recent_users}\n\n"
        f"• Total alerts sent: {platform_stats['total_alerts_sent']}\n"
    )
    await update.message.reply_html(msg)

async def broadcast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin_update(update):
        await update.message.reply_text("Access denied.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
    message = " ".join(context.args)
    active_users = user_manager.get_active_users()
    sent = 0
    failed = 0
    for chat_id in active_users.keys():
        try:
            await context.bot.send_message(chat_id=int(chat_id), text=f"📢 <b>Announcement</b>\n\n{message}", parse_mode="HTML")
            sent += 1
        except Exception as e:
            logging.warning("Failed broadcast to %s: %s", chat_id, e)
            failed += 1
        await asyncio.sleep(0.1)
    await update.message.reply_html(f"✅ Broadcast complete!\n• Sent: {sent}\n• Failed: {failed}")

async def periodic_overlap_download():
    while True:
        try:
            logging.info("⏬ Refreshing overlap_results.pkl from Supabase...")
            download_overlap_results(str(OVERLAP_FILE), bucket=BUCKET_NAME)
        except Exception as e:
            logging.error(f"❌ Failed to refresh overlap_results.pkl: {e}")
        await asyncio.sleep(180)  # 3 minutes

# ----------------------
# Startup hook
# ----------------------
async def on_startup(app: Application):
    # ensure data dir and baseline files exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_save(USER_PREFS_FILE, safe_load(USER_PREFS_FILE, {}))
    safe_save(ALERTS_STATE_FILE, safe_load(ALERTS_STATE_FILE, {}))
    safe_save(USER_STATS_FILE, safe_load(USER_STATS_FILE, {}))

    # Optional: download bot data & overlap once at startup (opt-in)
    if USE_SUPABASE:
        logging.info("Supabase integration enabled (opt-in). Attempting startup downloads...")
        download_bot_data_from_supabase()
        if DOWNLOAD_OVERLAP_ON_STARTUP and download_overlap_results is not None:
            try:
                download_overlap_results(str(OVERLAP_FILE), bucket=BUCKET_NAME)
                logging.info("Downloaded overlap_results.pkl at startup (opt-in).")
            except Exception as e:
                logging.debug("Startup overlap download failed: %s", e)

    # Start background loop (no supabase I/O in the loop)
    try:
        app.create_task(background_loop(app))
        app.create_task(monthly_expiry_notifier(app))
    except Exception:
        asyncio.create_task(background_loop(app))

    # Start daily supabase sync task only if both USE_SUPABASE and SUPABASE_DAILY_SYNC are true
    if USE_SUPABASE and SUPABASE_DAILY_SYNC:
        try:
            app.create_task(daily_supabase_sync())
            app.create_task(periodic_overlap_download())
        except Exception:
            asyncio.create_task(daily_supabase_sync())

    logging.info("Bot startup complete. Monitoring for token alerts...")

async def adduser_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command: add a user with expiry in days."""
    if not is_admin_update(update):
        await update.message.reply_text("⛔ Access denied. Admins only.")
        return
    
    if len(context.args) < 2:
        await update.message.reply_text("⚠️ Usage: /adduser <chat_id> <days>")
        return
    
    try:
        chat_id = str(context.args[0])  # Ensure string
        days = int(context.args[1])
        
        logging.info(f"🔧 Admin adding user {chat_id} with {days} days validity")
        
        # Add user with expiry
        expiry_date = user_manager.add_user_with_expiry(chat_id, days)
        
        # Immediately test the subscription status
        is_sub_after = is_subscribed(chat_id)
        
        await update.message.reply_text(
            f"✅ User {chat_id} added/updated with expiry {expiry_date}\n"
            f"🔍 Subscription check: {is_sub_after}\n"
            f"📝 Tell user to try /start now"
        )
        
    except Exception as e:
        logging.exception("❌ Error in /adduser:")
        await update.message.reply_text(f"❌ Failed to add user: {e}")

async def debug_user_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug command to check user status."""
    if not is_admin_update(update):
        await update.message.reply_text("⛔ Access denied. Admins only.")
        return
        
    if not context.args:
        chat_id = str(update.effective_chat.id)
    else:
        chat_id = str(context.args[0])
    
    prefs = safe_load(USER_PREFS_FILE, {})
    user_data = prefs.get(chat_id, {})
    
    is_sub = is_subscribed(chat_id)
    is_expired = user_manager.is_subscription_expired(chat_id)
    
    debug_msg = (
        f"🔍 <b>Debug User {chat_id}</b>\n\n"
        f"<b>Raw data:</b>\n"
        f"• Found in prefs: {chat_id in prefs}\n"
        f"• subscribed: {user_data.get('subscribed', 'NOT SET')}\n"
        f"• active: {user_data.get('active', 'NOT SET')}\n"
        f"• expires_at: {user_data.get('expires_at', 'NOT SET')}\n\n"
        f"<b>Function results:</b>\n"
        f"• is_subscribed(): {is_sub}\n"
        f"• is_subscription_expired(): {is_expired}\n\n"
        f"<b>All user data:</b>\n"
        f"<code>{user_data}</code>"
    )
    
    await update.message.reply_html(debug_msg)

# ----------------------
# Main
# ----------------------
def main():
    defaults = Defaults(parse_mode="HTML")
    app = Application.builder().token(BOT_TOKEN).defaults(defaults).build()

    # register handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("setalerts", setalerts_cmd))
    app.add_handler(CommandHandler("myalerts", myalerts_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("admin", admin_stats_cmd))
    app.add_handler(CommandHandler("broadcast", broadcast_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CommandHandler("testalert", testalert_cmd))
    app.add_handler(CommandHandler("adduser", adduser_cmd))
    app.add_handler(CommandHandler("debuguser", debug_user_cmd))

    # set startup hook
    app.post_init = on_startup

    logging.info("Starting unified telegram bot (local-first)...")
    app.run_polling(allowed_updates=None, poll_interval=1.0)

if __name__ == "__main__":
    main()
