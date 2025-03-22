import configparser
import logging
import random
import re
import asyncio
from typing import Any, Dict, List, Tuple
from io import BytesIO
import tempfile
import os

import openai
from telegram.request import _httpxrequest

# Monkey patch: Remove 'proxies' from telegram's HTTPXRequest kwargs
original_build_client = _httpxrequest.HTTPXRequest._build_client
def patched_build_client(self):
    self._client_kwargs.pop("proxies", None)
    return original_build_client(self)
_httpxrequest.HTTPXRequest._build_client = patched_build_client
# End monkey patch

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global statistics and conversation history
stats: Dict[str, int] = {"messages_processed": 0, "errors": 0}
conversation_history: Dict[Tuple[str, int], List[str]] = {}
model_weights: Dict[str, float] = {}

# Load and clean config
config = configparser.ConfigParser()
config.read("config.ini")

def clean_config_value(value: str) -> str:
    """Clean configuration value by splitting at semicolon and stripping extra text."""
    return value.split(";", 1)[0].strip()

# Multiple bot tokens, system prompts, and bot usernames (comma-separated lists)
BOT_TOKENS: List[str] = [
    clean_config_value(token)
    for token in config.get("bots", "telegram_tokens").split(",")
    if token.strip()
]
SYSTEM_PROMPTS: List[str] = [
    clean_config_value(prompt)
    for prompt in config.get("bots", "system_prompts").split(",")
    if prompt.strip()
]
BOT_USERNAMES: List[str] = [
    clean_config_value(username)
    for username in config.get("bots", "bot_usernames").split(",")
    if username.strip()
]

if not (len(BOT_TOKENS) == len(SYSTEM_PROMPTS) == len(BOT_USERNAMES)):
    logger.critical("The number of telegram_tokens, system_prompts, and bot_usernames must match.")
    exit(1)

# Other settings remain global
VLLM_ENDPOINTS: List[str] = [
    endpoint.strip()
    for endpoint in config.get("vllm_endpoints", "endpoints").split(",")
    if endpoint.strip()
]
DEFAULT_SYSTEM_PROMPT: str = clean_config_value(config.get("settings", "system_prompt", fallback=""))
MAX_MESSAGE_LENGTH: int = 4096
REPLY_CHANCE: float = float(clean_config_value(config.get("settings", "reply_chance", fallback="0.1")))
MAX_TOKENS: int = int(clean_config_value(config.get("settings", "max_tokens", fallback="200")))
BLACKLISTED_MODELS: set = set(
    model.strip()
    for model in clean_config_value(config.get("settings", "blacklist_models", fallback="")).split(",")
    if model.strip()
)
MAX_MODEL_RETRIES: int = int(clean_config_value(config.get("settings", "max_model_retries", fallback="3")))
RETRY_DELAY: float = float(clean_config_value(config.get("settings", "retry_delay", fallback="10")))

# TTS Settings
TTS_API_KEY: str = clean_config_value(config.get("tts", "api_key", fallback="YOLO"))
TTS_ENDPOINT: str = clean_config_value(config.get("tts", "endpoint", fallback="https://api.tts.example.com/synthesize"))
TTS_MODEL: str = clean_config_value(config.get("tts", "model", fallback="tts-1"))
TTS_VOICE: str = clean_config_value(config.get("tts", "voice", fallback="echo"))
TTS_SPEED: float = float(clean_config_value(config.get("tts", "speed", fallback="0.9")))

# Model management
MODEL_CLIENT_MAP: Dict[str, Any] = {}

def fetch_models() -> List[Dict[str, str]]:
    """Fetch available models from the provided VLLM endpoints and initialize weights."""
    models: List[Dict[str, str]] = []
    for endpoint in VLLM_ENDPOINTS:
        try:
            client = openai.OpenAI(base_url=endpoint, api_key="not-needed")
            resp = client.models.list()
            for m in resp.data:
                model_id = m.id
                if model_id and (model_id not in MODEL_CLIENT_MAP) and (model_id not in BLACKLISTED_MODELS):
                    MODEL_CLIENT_MAP[model_id] = client
                    models.append({"id": model_id, "endpoint": endpoint})
                    model_weights[model_id] = 1.0
                    logger.info(f"[✅] Registered model '{model_id}' from {endpoint}")
                elif model_id in BLACKLISTED_MODELS:
                    logger.info(f"[🚫] Skipped blacklisted model '{model_id}' from {endpoint}")
        except Exception as e:
            logger.error(f"[⚠️] Error with endpoint {endpoint}: {e}")
    return models

models = fetch_models()
if not models:
    logger.critical("[🔥] No models loaded. Shutting down.")
    exit(1)

def weighted_model_selection(available_models: List[Dict[str, str]]) -> Dict[str, str]:
    """Select a model using weighted random selection based on model_weights."""
    total_weight = sum(model_weights.get(m["id"], 1.0) for m in available_models)
    rnd = random.uniform(0, total_weight)
    upto = 0
    for m in available_models:
        weight = model_weights.get(m["id"], 1.0)
        if upto + weight >= rnd:
            return m
        upto += weight
    return random.choice(available_models)

# Telegram Utilities
async def send_long_message(update: Update, text: str) -> None:
    """Send a long message in chunks if needed."""
    for i in range(0, len(text), MAX_MESSAGE_LENGTH):
        await update.message.reply_text(
            text[i: i + MAX_MESSAGE_LENGTH],
            reply_to_message_id=update.message.message_id,
        )

def extract_final_reply(response: str) -> str:
    """
    Extract the final reply from the model's response.
    Removes HTML tags and extracts text after defined markers.
    """
    cleaned_response = re.sub(r"<.*?>", "", response, flags=re.DOTALL | re.IGNORECASE)
    markers = ["final answer:", "answer:", "assistant:", "conclusion:"]
    for marker in markers:
        pattern = re.compile(re.escape(marker), re.IGNORECASE)
        match = pattern.search(cleaned_response)
        if match:
            return cleaned_response[match.end():].strip()
    return cleaned_response.strip()

# TTS Functions
async def text_to_speech(text: str) -> BytesIO:
    """
    Convert text to speech using the TTS endpoint.
    Streams audio to a temporary file then loads it into a BytesIO.
    """
    def sync_tts():
        tts_client = openai.OpenAI(api_key=TTS_API_KEY, base_url=TTS_ENDPOINT)
        audio_buffer = BytesIO()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_filename = tmp.name
        try:
            with tts_client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                speed=TTS_SPEED,
                input=text
            ) as response:
                response.stream_to_file(temp_filename)
            with open(temp_filename, "rb") as f:
                audio_buffer.write(f.read())
        finally:
            os.remove(temp_filename)
        audio_buffer.seek(0)
        return audio_buffer
    return await asyncio.to_thread(sync_tts)

async def send_voice_message(update: Update, text: str) -> None:
    """Generate and send a voice note from text."""
    try:
        audio_file = await text_to_speech(text)
        await update.message.reply_voice(voice=audio_file, caption="Voice reply:")
    except Exception as e:
        logger.error(f"[TTS Error] Could not send voice note: {e}")

# Command Handlers

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text("lmao u found me, glowies r watching, /credits 4 more")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = (
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/models - List available models\n"
        "/credits - Show credits\n"
        "/stats - Show bot usage stats\n"
        "/setpersonality <text> - Change the personality for this chat\n"
        "/clearpersonality - Clear custom personality and revert to default"
    )
    await update.message.reply_text(help_text)

async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List registered models."""
    msg = "\n".join(
        f"{i+1}. {m['id']} @ {re.sub(r'https?://', '[REDACTED]/', m['endpoint'])}"
        for i, m in enumerate(models)
    )
    await send_long_message(update, f"brainz online:\n{msg}")

async def credits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show credits."""
    await update.message.reply_text("Disruptive Collective, x.com/DisruptiveCLCTV, we own ur soul now")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Return current usage stats."""
    stat_text = f"Messages processed: {stats['messages_processed']}\nErrors: {stats['errors']}"
    await update.message.reply_text(stat_text)

async def set_personality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Set a custom personality (system prompt override) for this chat.
    Usage: /setpersonality <new prompt text>
    """
    if not context.args:
        await update.message.reply_text("Usage: /setpersonality <new prompt>")
        return
    new_prompt = " ".join(context.args)
    if "personalities" not in context.bot_data:
        context.bot_data["personalities"] = {}
    chat_id = update.effective_chat.id
    context.bot_data["personalities"][chat_id] = new_prompt
    await update.message.reply_text("Personality updated for this chat!")

async def clear_personality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear the custom personality for the current chat."""
    chat_id = update.effective_chat.id
    personalities = context.bot_data.get("personalities", {})
    if chat_id in personalities:
        del personalities[chat_id]
        await update.message.reply_text("Custom personality cleared. Reverting to default system prompt.")
    else:
        await update.message.reply_text("No custom personality set for this chat.")

async def schizo_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Process incoming messages and reply using weighted model selection.
    Includes conversation history and uses either the chat-specific personality
    or the bot's default system prompt.
    """
    if not update.message or not update.message.text:
        logger.info("[📩] Received non-text update, skipping response.")
        return

    stats["messages_processed"] += 1
    user_message = update.message.text
    chat_id = update.effective_chat.id
    bot_username = context.bot_data.get("bot_username", "").lower()
    personality_overrides = context.bot_data.get("personalities", {})
    system_prompt = personality_overrides.get(chat_id, context.bot_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT))

    force_reply = False
    if f"@{bot_username}" in user_message.lower():
        force_reply = True
    if update.message.reply_to_message and update.message.reply_to_message.from_user.username:
        if update.message.reply_to_message.from_user.username.lower() == bot_username:
            force_reply = True

    if not force_reply and random.random() > REPLY_CHANCE:
        return

    conv_key = (bot_username, chat_id)
    history = conversation_history.get(conv_key, [])
    history.append(f"User: {user_message}")
    if len(history) > 5:
        history = history[-5:]
    conversation_history[conv_key] = history
    context_text = "\n".join(history) + "\n"

    attempts = 0
    tried_models = set()

    while attempts < MAX_MODEL_RETRIES:
        available_models = [m for m in models if m["id"] not in tried_models]
        if not available_models:
            break
        model = weighted_model_selection(available_models)
        tried_models.add(model["id"])
        dynamic_max_tokens = min(len(user_message) * 2, MAX_TOKENS)
        prompt = f"{system_prompt}\n{context_text}User: {user_message}\nSchizo:"
        logger.info(f"[🤡] Attempt {attempts + 1}/{MAX_MODEL_RETRIES}: Using {model['id']} @ {model['endpoint']}")
        client = MODEL_CLIENT_MAP.get(model["id"])
        if not client:
            logger.error(f"[💥] No client found for {model['id']}")
            attempts += 1
            continue
        try:
            resp = client.chat.completions.create(
                model=model["id"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=dynamic_max_tokens,
                temperature=0.75,
            )
            if not resp.choices or not resp.choices[0].message.content:
                raise ValueError("No valid response content from model")
            reply = extract_final_reply(resp.choices[0].message.content)
            await send_long_message(update, reply)
            await send_voice_message(update, reply)
            logger.info(f"[🎭] Schizo’d: {reply[:100]}...")
            model_weights[model["id"]] += 0.1
            history.append(f"Schizo: {reply}")
            conversation_history[conv_key] = history[-5:]
            return
        except Exception as e:
            logger.error(f"[💥] Error with {model['id']} @ {model['endpoint']}: {e}")
            stats["errors"] += 1
            attempts += 1
            model_weights[model["id"]] = max(model_weights[model["id"]] - 0.2, 0.1)

    logger.warning(f"[❌] Exhausted {MAX_MODEL_RETRIES} retries for message: '{user_message[:50]}...'")
    await update.message.reply_text("Ricky is retarded rn, try later", reply_to_message_id=update.message.message_id)

# Function to register all bot commands
async def register_bot_commands(bot_app):
    bot_app.add_handlers([
        CommandHandler("start", start),
        CommandHandler("help", help_cmd),
        CommandHandler("models", list_models),
        CommandHandler("credits", credits),
        CommandHandler("stats", stats_cmd),
        CommandHandler("setpersonality", set_personality),
        CommandHandler("clearpersonality", clear_personality),
        MessageHandler(filters.TEXT & (~filters.COMMAND), schizo_reply),
    ])

# Function to start an individual Telegram bot instance
async def start_telegram_bot_instance(token: str, system_prompt: str, bot_username: str) -> None:
    """Start a Telegram bot instance with its own token, system prompt, and username."""
    bot_app = Application.builder().token(token).build()
    bot_app.bot_data["system_prompt"] = system_prompt
    bot_app.bot_data["bot_username"] = bot_username
    bot_app.bot_data["personalities"] = {}

    await register_bot_commands(bot_app)

    logger.info(f"Starting Telegram bot for @{bot_username} (token ending in {token[-5:]})...")
    await bot_app.initialize()
    await bot_app.start()

    attempt = 0
    base_delay = RETRY_DELAY
    max_delay = 60
    while True:
        try:
            await bot_app.updater.start_polling(timeout=15, drop_pending_updates=True)
            attempt = 0
            await asyncio.Future()
        except Exception as e:
            attempt += 1
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.error(f"Polling error for bot @{bot_username} (attempt {attempt}): {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

# FastAPI Server Setup
app = FastAPI()

@app.get("/")
async def index() -> Dict[str, str]:
    """Root endpoint returning a simple message."""
    return {"message": "SchitzoChat V1: Telegram bot + schizo web, hit /models for brainz"}

@app.get("/models")
async def get_models() -> List[Dict[str, str]]:
    """Endpoint listing available models."""
    return [{"id": m["id"], "endpoint": m["endpoint"]} for m in models]

@app.get("/stats")
async def get_stats() -> Dict[str, int]:
    """Endpoint returning current usage stats."""
    return stats

@app.post("/reload_config")
async def reload_config() -> Dict[str, str]:
    """Hot-reload config settings from config.ini (only non-bot settings)."""
    config.read("config.ini")
    global DEFAULT_SYSTEM_PROMPT, REPLY_CHANCE, MAX_TOKENS, MAX_MODEL_RETRIES, RETRY_DELAY
    DEFAULT_SYSTEM_PROMPT = clean_config_value(config.get("settings", "system_prompt", fallback=""))
    REPLY_CHANCE = float(clean_config_value(config.get("settings", "reply_chance", fallback="0.1")))
    MAX_TOKENS = int(clean_config_value(config.get("settings", "max_tokens", fallback="200")))
    MAX_MODEL_RETRIES = int(clean_config_value(config.get("settings", "max_model_retries", fallback="3")))
    RETRY_DELAY = float(clean_config_value(config.get("settings", "retry_delay", fallback="10")))
    return {"message": "Config reloaded successfully."}

# Combined Startup: Run multiple Telegram bots and the FastAPI server concurrently.
async def main() -> None:
    telegram_tasks = [
        start_telegram_bot_instance(token, prompt, username)
        for token, prompt, username in zip(BOT_TOKENS, SYSTEM_PROMPTS, BOT_USERNAMES)
    ]
    server = uvicorn.Server(uvicorn.Config(app=app, host="0.0.0.0", port=8007, loop="asyncio", log_level="info"))
    await asyncio.gather(server.serve(), *telegram_tasks)

if __name__ == "__main__":
    asyncio.run(main())
