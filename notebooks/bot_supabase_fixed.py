#!/usr/bin/env python3
"""
Unified Telegram Bot
- Combines Supabase upload/download reliability (from bot_supabase_fixed.py)
- Combines full feature set (market cap tracking, prefs, alerts, commands) (from bot_updated_with_initial_mc.py)
"""

import os
import asyncio
import logging
import joblib
import requests
from threading import Lock
from pathlib import Path
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

# --- Supabase helpers (robust from supabase_fixed) ---
from supabase_utils import (
    download_overlap_results,
    upload_file,
    download_file,
)

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

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------
# Safe load/save
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
# Supabase sync functions
# ----------------------
def upload_bot_data_to_supabase():
    for file in [USER_PREFS_FILE, USER_STATS_FILE, ALERTS_STATE_FILE]:
        if file.exists():
            upload_file(str(file), bucket=BUCKET_NAME)

def download_bot_data_from_supabase():
    for file in [USER_PREFS_FILE, USER_STATS_FILE, ALERTS_STATE_FILE]:
        download_file(str(file), os.path.basename(file), bucket=BUCKET_NAME)

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
            "grades": ALL_GRADES.copy(),
            "created_at": self.now_iso(),
            "updated_at": self.now_iso(),
            "active": True,
            "total_alerts_received": 0,
            "last_alert_at": None
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
            upload_bot_data_to_supabase()
            return True
        except Exception as e:
            logging.exception("Failed to update user prefs: %s", e)
            return False

    def deactivate_user(self, chat_id: str) -> bool:
        return self.update_user_prefs(chat_id, {"active": False, "deactivated_at": self.now_iso()})

    def activate_user(self, chat_id: str) -> bool:
        return self.update_user_prefs(chat_id, {"active": True, "reactivated_at": self.now_iso()})

    def get_active_users(self):
        prefs = safe_load(self.prefs_file, {})
        return {k: v for k, v in prefs.items() if v.get("active", True)}

    def get_user_stats(self, chat_id: str):
        stats = safe_load(self.stats_file, {})
        return stats.get(chat_id, {
            "alerts_received": 0,
            "last_alert_at": None,
            "joined_at": None,
            "grade_breakdown": {g: 0 for g in ALL_GRADES}
        })

    def update_user_stats(self, chat_id: str, grade: str = None):
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
        upload_bot_data_to_supabase()

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
    def save_to_supabase(self):
        upload_bot_data_to_supabase()

    @staticmethod
    def now_iso():
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

user_manager = UserManager(USER_PREFS_FILE, USER_STATS_FILE)

# ----------------------
# Load overlap file
# ----------------------
def load_latest_tokens_from_overlap() -> Dict[str, Dict[str, Any]]:
    logging.info("Downloading overlap_results.pkl...")
    download_overlap_results(str(OVERLAP_FILE), bucket=BUCKET_NAME)
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
def format_alert_html(token_data, alert_type, previous_grade=None, initial_mc=None, initial_fdv=None, first_alert_at=None):
    token_meta = token_data.get("token_metadata") or {}
    name = token_meta.get("name") or token_data.get("token") or "Unknown"
    symbol = token_meta.get("symbol") or ""
    grade = token_data.get("grade", "NONE")
    mint = token_meta.get("mint", "")

    current_mc, current_fdv, current_display = fetch_marketcap_and_fdv(mint)
    if current_mc and initial_mc:
        mc_line = f"💰 <b>Market Cap:</b> {format_marketcap_display(current_mc)} (was {format_marketcap_display(initial_mc)})"
    elif current_fdv and initial_fdv:
        mc_line = f"🏷️ <b>FDV:</b> {format_marketcap_display(current_fdv)} (was {format_marketcap_display(initial_fdv)})"
    else:
        mc_line = f"💰 <b>Market Cap/FDV:</b> {current_display}"

    lines = [
        "🚀 <b>New Token Detected</b>" if alert_type == "NEW" else "🔁 <b>Grade Changed</b>",
        f"<b>{name}</b> ({symbol})" if symbol else f"<b>{name}</b>",
        f"<b>Grade:</b> {grade}" + (f" (was {previous_grade})" if previous_grade else ""),
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
        pairs = data.get("pairs", [])
        mc = None
        fdv = None
        if pairs:
            mc = pairs[0].get("marketCap")
            fdv = pairs[0].get("fdv")

        token_data = {
            "token": token_id,
            "grade": grade,
            "token_metadata": {
                "name": pairs[0].get("baseToken", {}).get("name", "TestToken"),
                "symbol": pairs[0].get("baseToken", {}).get("symbol", "TT"),
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
    token_data: the latest check dict from overlap_results
    """
    active_users = user_manager.get_active_users()
    if not active_users:
        logging.debug("No active users to send alerts to.")
        return

    # prepare message once
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
        # ensure subscribed_grades is normalized list
        if isinstance(subscribed_grades, (list, tuple)):
            if grade not in subscribed_grades:
                continue
        else:
            # fallback to defaults if malformed
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

# ----------------------
# Background loop
# ----------------------
async def background_loop(app: Application):
    logging.info("Background alert loop started...")
    alerts_state = safe_load(ALERTS_STATE_FILE, {})  # Enhanced structure
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

                # Get current alert state or create new one
                current_state = alerts_state.get(token_id, {})
                last_grade = current_state.get("last_grade")

                if grade != last_grade:
                    logging.info("New/changed grade for %s: %s -> %s", token_id, last_grade, grade)

                    # For first-time alerts, capture initial market cap/FDV
                    if last_grade is None:
                        mc, fdv, _ = fetch_marketcap_and_fdv(token_id)

                        # Create new alert state entry
                        alerts_state[token_id] = {
                            "last_grade": grade,
                            "initial_marketcap": mc,
                            "initial_fdv": fdv,
                            "first_alert_at": datetime.utcnow().isoformat() + "Z"
                        }

                        logging.info("Captured initial market data for %s: MC=%s, FDV=%s",
                                   token_id, format_marketcap_display(mc), format_marketcap_display(fdv))
                    else:
                        # Update existing state
                        alerts_state[token_id]["last_grade"] = grade

                    # Send alert with historical market cap data
                    await send_alert_to_subscribers(
                        app,
                        token,
                        grade,
                        previous_grade=last_grade,
                        initial_mc=current_state.get("initial_marketcap"),
                        initial_fdv=current_state.get("initial_fdv"),
                        first_alert_at=current_state.get("first_alert_at")
                    )

            # Persist alerts_state and push to Supabase
            safe_save(ALERTS_STATE_FILE, alerts_state)
            upload_bot_data_to_supabase()

        except Exception as e:
            logging.exception("Error in background loop: %s", e)

        await asyncio.sleep(POLL_INTERVAL_SECS)

# ----------------------
# Telegram Commands (subset, can extend)
# ----------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
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

# ----------------------
# Startup
# ----------------------
async def on_startup(app: Application):
    # ensure data dir and baseline files exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_save(USER_PREFS_FILE, safe_load(USER_PREFS_FILE, {}))
    safe_save(ALERTS_STATE_FILE, safe_load(ALERTS_STATE_FILE, {}))
    safe_save(USER_STATS_FILE, safe_load(USER_STATS_FILE, {}))

    # Attempt to download bot data files from Supabase (best-effort)
    download_bot_data_from_supabase()

    # Start background loop
    # Use Application.create_task so PTB can manage lifecycle
    try:
        app.create_task(background_loop(app))
    except Exception:
        # fallback if create_task not available for some PTB versions
        asyncio.create_task(background_loop(app))

    logging.info("Bot startup complete. Monitoring for token alerts...")

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

    # set startup hook (assign post_init so run_polling doesn't receive unknown args on some PTB versions)
    app.post_init = on_startup

    logging.info("Starting enhanced telegram bot with market cap tracking...")
    # only call run_polling once
    app.run_polling(allowed_updates=None, poll_interval=1.0)

if __name__ == "__main__":
    main()
