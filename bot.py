"""
TISM v5 – Totally Insane Synthetic Machines (Version 5)

A Telegram bot framework that uses vllms for roleplaying, with per-chat model and personality settings,
TTS support, FastAPI server with enhanced metrics, logging, and real-time chat log streaming.
"""

import configparser
import logging
import random
import re
import asyncio
import datetime
import json
from typing import Any, Dict, List, Tuple, Optional
from io import BytesIO
import tempfile
import os

import openai
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from fastapi import FastAPI, Request, Response
import uvicorn

# Prometheus for metrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import StreamingResponse

# ----------------------
# Configuration & Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_file: str = "config.ini") -> configparser.ConfigParser:
    """Load and parse the configuration file."""
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    return cfg

def clean_config_value(value: str) -> str:
    """Clean a config value by stripping comments and whitespace."""
    return value.split(";", 1)[0].strip()

config = load_config()

# ----------------------
# Global State & Metrics
# ----------------------
stats: Dict[str, int] = {"messages_processed": 0, "errors": 0}
conversation_history: Dict[Tuple[str, int], List[str]] = {}
model_weights: Dict[str, float] = {}

# Global chat log: Each entry is a dict with metadata about the conversation turn.
chat_logs: List[Dict[str, Any]] = []

# Prometheus metrics
MESSAGES_PROCESSED = Counter('tism_messages_processed', 'Total messages processed')
ERROR_COUNTER = Counter('tism_errors_total', 'Total errors encountered')
MODEL_SELECTION_LATENCY = Histogram('tism_model_selection_seconds', 'Latency for model selection and response')

# ----------------------
# Bot Settings from Config
# ----------------------
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
AVAILABLE_VOICES: List[str] = [
    v.strip() for v in config.get("tts", "available_voices", fallback="echo,default").split(",") if v.strip()
]

# Mapping from model ID to OpenAI client instance
MODEL_CLIENT_MAP: Dict[str, Any] = {}

# ----------------------
# Monkey Patch Telegram HTTPX Client
# ----------------------
# Remove 'proxies' from telegram's HTTPXRequest kwargs (quick hack to bypass network issues)
from telegram.request import _httpxrequest
original_build_client = _httpxrequest.HTTPXRequest._build_client
def patched_build_client(self):
    self._client_kwargs.pop("proxies", None)
    return original_build_client(self)
_httpxrequest.HTTPXRequest._build_client = patched_build_client

# ----------------------
# Model Management
# ----------------------
def fetch_models() -> List[Dict[str, str]]:
    """
    Query all VLLM endpoints for available models, register non-blacklisted ones,
    and set an initial weight.
    Clears existing MODEL_CLIENT_MAP and model_weights before fetching.
    """
    global MODEL_CLIENT_MAP, model_weights
    MODEL_CLIENT_MAP.clear()
    model_weights.clear()

    fetched_models: List[Dict[str, str]] = []
    logger.info("Starting model fetch from endpoints...")
    for endpoint in VLLM_ENDPOINTS:
        try:
            client = openai.OpenAI(base_url=endpoint, api_key="not-needed")
            resp = client.models.list()
            for m in resp.data:
                model_id = m.id
                if model_id and (model_id not in BLACKLISTED_MODELS):
                    if model_id not in MODEL_CLIENT_MAP:
                        MODEL_CLIENT_MAP[model_id] = client
                        fetched_models.append({"id": model_id, "endpoint": endpoint})
                        model_weights[model_id] = 1.0
                        logger.info(f"[✅] Registered model '{model_id}' from {endpoint}")
                    else:
                        logger.info(f"[ℹ️] Model '{model_id}' already registered from another endpoint, skipping duplicate from {endpoint}")
                elif model_id in BLACKLISTED_MODELS:
                    logger.info(f"[🚫] Skipped blacklisted model '{model_id}' from {endpoint}")
        except Exception as e:
            logger.error(f"[⚠️] Error fetching models from endpoint {endpoint}: {e}")

    if not fetched_models:
        logger.warning("[⚠️] No models were successfully fetched from any endpoint.")
    else:
        logger.info(f"Finished model fetch. Registered {len(fetched_models)} models.")

    return fetched_models

models = fetch_models()
if not models:
    logger.critical("[🔥] No models loaded during initial startup. Check VLLM endpoints and configuration. Shutting down.")
    exit(1)

def weighted_model_selection(available_models: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Select a model based on weighted probabilities."""
    valid_models = [m for m in available_models if m["id"] in model_weights]
    if not valid_models:
        return None

    total_weight = sum(model_weights[m["id"]] for m in valid_models)
    if total_weight <= 0:
         return random.choice(valid_models) if valid_models else None

    rnd = random.uniform(0, total_weight)
    upto = 0
    for m in valid_models:
        weight = model_weights[m["id"]]
        if upto + weight >= rnd:
            return m
        upto += weight
    return valid_models[-1] if valid_models else None

# ----------------------
# Telegram Utilities & Command Handlers
# ----------------------
async def send_long_message(update: Update, text: str) -> None:
    """Send a long text message by chunking if needed."""
    if not update.message: return
    for i in range(0, len(text), MAX_MESSAGE_LENGTH):
        chunk = text[i: i + MAX_MESSAGE_LENGTH]
        await update.message.reply_text(
            chunk,
            reply_to_message_id=update.message.message_id,
        )
        await asyncio.sleep(0.2)

def extract_final_reply(response: str) -> str:
    """Clean and extract the final answer from the model's response."""
    cleaned_response = re.sub(r"<.*?>", "", response, flags=re.DOTALL | re.IGNORECASE)
    markers = ["final answer:", "answer:", "assistant:", "conclusion:", "response:", "reply:"]
    lower_cleaned = cleaned_response.lower()
    best_pos = -1
    for marker in markers:
        pos = lower_cleaned.rfind(marker)
        if pos > best_pos:
            best_pos = pos + len(marker)
    if best_pos != -1:
        return cleaned_response[best_pos:].strip()
    else:
        return cleaned_response.strip()

async def send_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Convert text to speech and send as a voice message."""
    if not update.message: return
    try:
        chat_id = update.effective_chat.id
        voice = context.bot_data.get("chat_voice", {}).get(chat_id, TTS_VOICE)
        tts_client = openai.OpenAI(api_key=TTS_API_KEY, base_url=TTS_ENDPOINT)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            temp_filename = tmp.name

        try:
            with tts_client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=voice,
                speed=TTS_SPEED,
                input=text,
                response_format="mp3"
            ) as response:
                response.stream_to_file(temp_filename)

            await update.message.reply_voice(voice=open(temp_filename, "rb"), caption="Voice reply:")

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    except Exception as e:
        logger.error(f"[TTS Error] Could not generate or send voice note for chat {update.effective_chat.id}: {e}", exc_info=True)

# Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    await update.message.reply_text("Hey there! Welcome to TISM. Try /help for commands.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    help_text = (
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/models - List available models\n"
        "/setmodel <number> - Select model for this chat\n"
        "/unsetmodel - Clear chat-specific model setting\n"
        "/reload_models - Refresh the list of available models\n"
        "/credits - Show credits\n"
        "/stats - Show bot usage stats\n"
        "/setpersonality <text> - Change personality for this chat\n"
        "/clearpersonality - Clear custom personality\n"
        "/tts_on - Enable text-to-speech responses\n"
        "/tts_off - Disable text-to-speech responses\n"
        "/listvoices - List available TTS voices\n"
        "/setvoice <number> - Select TTS voice for this chat"
    )
    await update.message.reply_text(help_text)

async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    if not models:
         await update.message.reply_text("No models are currently loaded. Try /reload_models or check the logs.")
         return
    msg = "\n".join(
        f"{i+1}. {m.get('id', 'Unknown ID')} @ {re.sub(r'https?://', '[REDACTED]/', m.get('endpoint', 'Unknown Endpoint'))}"
        for i, m in enumerate(models)
    )
    await send_long_message(update, f"Available models:\n{msg}")

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model_number>\nUse /models to see the list.")
        return
    if not models:
         await update.message.reply_text("No models are currently loaded. Cannot set a model.")
         return
    try:
        index = int(context.args[0]) - 1
        if index < 0 or index >= len(models):
            await update.message.reply_text(f"Invalid model number. Please choose between 1 and {len(models)}.")
            return

        selected_model = models[index]
        model_id = selected_model.get('id')
        if not model_id:
            await update.message.reply_text("Error: Selected model data is incomplete. Cannot set model.")
            logger.error(f"Incomplete model data at index {index}: {selected_model}")
            return

        context.bot_data.setdefault("chat_model", {})[update.effective_chat.id] = model_id
        await update.message.reply_text(f"Model for this chat set to: {model_id}")
        logger.info(f"Chat {update.effective_chat.id} set to use model '{model_id}'")

    except (ValueError, IndexError):
        await update.message.reply_text("Invalid input. Please provide a valid model number from the /models list.")

async def unset_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    chat_id = update.effective_chat.id
    chat_models = context.bot_data.setdefault("chat_model", {})

    if chat_id in chat_models:
        del chat_models[chat_id]
        await update.message.reply_text("Chat-specific model setting cleared. Now using default model selection.")
        logger.info(f"Cleared model preference for chat {chat_id}")
    else:
        await update.message.reply_text("No chat-specific model was set. Using default selection.")

async def reload_models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global models
    if not update.message: return

    await update.message.reply_text("🔄 Refreshing model list from backend endpoints...")
    logger.info(f"Manual model reload triggered by user {update.effective_user.id} in chat {update.effective_chat.id}")

    try:
        new_models = fetch_models()
        models = new_models

        if models:
            await update.message.reply_text(f"✅ Model list refreshed successfully. Found {len(models)} models.")
            logger.info(f"Manual model reload complete. {len(models)} models loaded.")
        else:
            await update.message.reply_text("⚠️ Model list refreshed, but no models were found. Please check endpoint configurations and logs.")
            logger.warning("Manual model reload resulted in zero models.")

    except Exception as e:
        await update.message.reply_text(f"❌ An error occurred while reloading models: {e}")
        logger.error(f"Error during manual model reload: {e}", exc_info=True)

async def credits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    await update.message.reply_text("Built by the Disruptive Collective. Transparency is key!")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    stat_text = (
        f"Messages processed: {stats.get('messages_processed', 0)}\n"
        f"Errors encountered: {stats.get('errors', 0)}\n"
        f"Currently loaded models: {len(models)}"
    )
    await update.message.reply_text(stat_text)

async def set_personality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    if not context.args:
        await update.message.reply_text("Usage: /setpersonality <your custom system prompt text>")
        return
    new_prompt = " ".join(context.args)
    context.bot_data.setdefault("personalities", {})[update.effective_chat.id] = new_prompt
    await update.message.reply_text("Personality updated for this chat!")
    logger.info(f"Custom personality set for chat {update.effective_chat.id}")

async def clear_personality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    chat_id = update.effective_chat.id
    personalities = context.bot_data.get("personalities", {})
    if chat_id in personalities:
        del personalities[chat_id]
        await update.message.reply_text("Custom personality cleared. Reverting to default system prompt for this bot.")
        logger.info(f"Cleared custom personality for chat {chat_id}")
    else:
        await update.message.reply_text("No custom personality set for this chat.")

async def tts_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    chat_id = update.effective_chat.id
    context.bot_data.setdefault("tts_enabled", {})[chat_id] = True
    await update.message.reply_text("Text-to-speech is now ON for this chat.")
    logger.info(f"TTS enabled for chat {chat_id}")

async def tts_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    chat_id = update.effective_chat.id
    context.bot_data.setdefault("tts_enabled", {})[chat_id] = False
    await update.message.reply_text("Text-to-speech is now OFF for this chat.")
    logger.info(f"TTS disabled for chat {chat_id}")

async def list_voices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    if AVAILABLE_VOICES:
        lines = [f"{i+1}. {voice}" for i, voice in enumerate(AVAILABLE_VOICES)]
        await update.message.reply_text("Available TTS voices:\n" + "\n".join(lines))
    else:
        await update.message.reply_text("No TTS voices are configured in config.ini.")

async def set_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    if not context.args:
        await update.message.reply_text("Usage: /setvoice <voice_number>\nUse /listvoices to see the options.")
        return
    if not AVAILABLE_VOICES:
        await update.message.reply_text("No TTS voices configured. Cannot set voice.")
        return
    try:
        index = int(context.args[0]) - 1
        if index < 0 or index >= len(AVAILABLE_VOICES):
            await update.message.reply_text(f"Invalid voice number. Please choose between 1 and {len(AVAILABLE_VOICES)}.")
            return

        selected_voice = AVAILABLE_VOICES[index]
        context.bot_data.setdefault("chat_voice", {})[update.effective_chat.id] = selected_voice
        await update.message.reply_text(f"TTS voice for this chat set to: {selected_voice}")
        logger.info(f"Chat {update.effective_chat.id} set TTS voice to '{selected_voice}'")

    except (ValueError, IndexError):
        await update.message.reply_text("Invalid input. Please provide a valid voice number from the /listvoices command.")

async def schizo_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main chat handler that processes messages and produces responses using available models."""
    if not update.message or not update.message.text:
        logger.debug("[📩] Received non-text or empty update, skipping.")
        return

    stats["messages_processed"] = stats.get("messages_processed", 0) + 1
    MESSAGES_PROCESSED.inc()

    user_message = update.message.text
    chat_id = update.effective_chat.id
    bot_username = context.bot_data.get("bot_username", "").lower()

    personality_overrides = context.bot_data.get("personalities", {})
    system_prompt = personality_overrides.get(chat_id, context.bot_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT))

    force_reply = False
    if bot_username and f"@{bot_username}" in user_message.lower():
        force_reply = True
    if update.message.reply_to_message and update.message.reply_to_message.from_user.username:
        replied_to_username = update.message.reply_to_message.from_user.username.lower()
        if bot_username and replied_to_username == bot_username:
            force_reply = True

    should_reply = force_reply or (random.random() < REPLY_CHANCE)
    if not should_reply:
        logger.debug(f"Skipping reply based on chance ({REPLY_CHANCE}) and context.")
        return

    conv_key = (bot_username, chat_id)
    history = conversation_history.get(conv_key, [])
    history.append(f"User: {user_message}")
    history = history[-5:]
    conversation_history[conv_key] = history
    context_text = "\n".join(history)

    prompt = f"{system_prompt}\n\n{context_text}\nSchizo:"
    logger.debug(f"Constructed prompt for chat {chat_id}:\n{prompt}")

    selected_model_info: Optional[Dict[str, str]] = None
    chat_specific_model_id = context.bot_data.get("chat_model", {}).get(chat_id)

    if chat_specific_model_id:
        found_model = next((m for m in models if m.get("id") == chat_specific_model_id), None)
        if found_model and found_model.get("id") in MODEL_CLIENT_MAP:
            selected_model_info = found_model
            logger.info(f"[🎯] Using chat-specific model '{chat_specific_model_id}' for chat {chat_id}")
        else:
            logger.warning(f"[⚠️] Chat {chat_id} requested model '{chat_specific_model_id}' but it's not available or loaded. Falling back to default selection.")
            if chat_id in context.bot_data.get("chat_model", {}):
                del context.bot_data["chat_model"][chat_id]

    if not selected_model_info:
        if not models:
             logger.error("[🔥] No models available globally. Cannot generate reply.")
             await update.message.reply_text("Sorry, no AI models are available right now.")
             return
        selected_model_info = weighted_model_selection(models)
        if selected_model_info:
             logger.info(f"[🎲] Using weighted random model '{selected_model_info.get('id')}' for chat {chat_id}")
        else:
             logger.error("[🔥] Weighted selection failed to return a model. Cannot generate reply.")
             await update.message.reply_text("Sorry, could not select an AI model to handle your request.")
             return

    reply = None
    model_id = selected_model_info.get("id")
    client = MODEL_CLIENT_MAP.get(model_id)

    if not client or not model_id:
         logger.error(f"[💥] Critical error: Client or Model ID missing for selected model: {selected_model_info}")
         await update.message.reply_text("Internal error: Could not find the client for the selected model.")
         return

    @MODEL_SELECTION_LATENCY.time()
    async def attempt_generation(current_model_id, current_client):
        nonlocal reply
        try:
            dynamic_max_tokens = min(max(100, len(user_message) * 2 + len(prompt)//2), MAX_TOKENS)
            logger.info(f"[💬] Generating reply using '{current_model_id}' (max_tokens: {dynamic_max_tokens})")
            resp = await asyncio.to_thread(
                current_client.chat.completions.create,
                model=current_model_id,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": context_text + "\nSchizo:"}],
                max_tokens=dynamic_max_tokens,
                temperature=0.75,
                stop=["User:", "\nUser:"]
            )

            if not resp.choices or not resp.choices[0].message or not resp.choices[0].message.content:
                raise ValueError(f"No valid response content received from model '{current_model_id}'")

            generated_text = resp.choices[0].message.content
            reply = extract_final_reply(generated_text)
            logger.info(f"[✅] Successfully generated reply from '{current_model_id}'. Preview: {reply[:100]}...")
            if current_model_id in model_weights:
                model_weights[current_model_id] = min(model_weights[current_model_id] + 0.1, 5.0)
            return True
        except Exception as e:
            logger.error(f"[💥] Error during generation with '{current_model_id}': {e}", exc_info=False)
            ERROR_COUNTER.inc()
            stats["errors"] = stats.get("errors", 0) + 1
            if current_model_id in model_weights:
                 model_weights[current_model_id] = max(model_weights[current_model_id] - 0.2, 0.1)
            return False

    success = await attempt_generation(model_id, client)

    if not success and not chat_specific_model_id:
        logger.warning(f"[🔄] Initial attempt with '{model_id}' failed. Trying fallbacks...")
        attempts = 1
        tried_models = {model_id}
        backoff = RETRY_DELAY

        while attempts < MAX_MODEL_RETRIES:
            available_fallback_models = [m for m in models if m.get("id") not in tried_models]
            if not available_fallback_models:
                logger.warning("[❌] No more fallback models available.")
                break

            fallback_model_info = weighted_model_selection(available_fallback_models)
            if not fallback_model_info:
                 logger.warning("[❌] Could not select a fallback model.")
                 break

            fallback_model_id = fallback_model_info.get("id")
            fallback_client = MODEL_CLIENT_MAP.get(fallback_model_id)
            tried_models.add(fallback_model_id)

            if not fallback_client or not fallback_model_id:
                 logger.error(f"[💥] Fallback model '{fallback_model_id}' missing client or ID. Skipping.")
                 attempts += 1
                 continue

            logger.info(f"[🔄] Retrying (Attempt {attempts + 1}/{MAX_MODEL_RETRIES}) with fallback '{fallback_model_id}' after {backoff:.1f}s delay...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

            success = await attempt_generation(fallback_model_id, fallback_client)
            if success:
                logger.info(f"[✅] Successfully generated reply using fallback '{fallback_model_id}'.")
                break
            else:
                 attempts += 1

        if not success:
             logger.error(f"[❌] Exhausted all {MAX_MODEL_RETRIES} retries for message in chat {chat_id}.")

    if reply:
        await send_long_message(update, reply)
        history.append(f"Schizo: {reply}")
        conversation_history[conv_key] = history[-5:]
        
        # Append the question and reply to the global chat log
        chat_logs.append({
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "chat_id": chat_id,
            "bot_username": bot_username,
            "question": user_message,
            "reply": reply
        })
        # Optionally limit the log size
        if len(chat_logs) > 500:
            chat_logs[:] = chat_logs[-500:]
        
        tts_enabled_for_chat = context.bot_data.get("tts_enabled", {}).get(chat_id, True)
        if tts_enabled_for_chat:
            asyncio.create_task(send_voice_message(update, context, reply))
    elif update.message:
        await update.message.reply_text("Sorry, I couldn't come up with a reply this time.",
                                          reply_to_message_id=update.message.message_id)

def register_bot_commands(bot_app: Application) -> None:
    handlers = [
        CommandHandler("start", start),
        CommandHandler("help", help_cmd),
        CommandHandler("models", list_models),
        CommandHandler("setmodel", set_model),
        CommandHandler("unsetmodel", unset_model),
        CommandHandler("reload_models", reload_models_cmd),
        CommandHandler("credits", credits),
        CommandHandler("stats", stats_cmd),
        CommandHandler("setpersonality", set_personality),
        CommandHandler("clearpersonality", clear_personality),
        CommandHandler("tts_on", tts_on),
        CommandHandler("tts_off", tts_off),
        CommandHandler("listvoices", list_voices),
        CommandHandler("setvoice", set_voice),
        MessageHandler(filters.TEXT & (~filters.COMMAND), schizo_reply),
    ]
    bot_app.add_handlers(handlers)

async def start_telegram_bot_instance(token: str, system_prompt: str, bot_username: str) -> None:
    bot_app = Application.builder().token(token).build()

    bot_app.bot_data["system_prompt"] = system_prompt
    bot_app.bot_data["bot_username"] = bot_username
    bot_app.bot_data["personalities"] = {}
    bot_app.bot_data["tts_enabled"] = {}
    bot_app.bot_data["chat_model"] = {}
    bot_app.bot_data["chat_voice"] = {}

    register_bot_commands(bot_app)
    logger.info(f"Starting Telegram bot instance for @{bot_username} (token ending in ...{token[-5:]})")

    await bot_app.initialize()
    await bot_app.start()
    logger.info(f"Bot @{bot_username} initialized and started.")

    attempt = 0
    base_delay = 5
    max_delay = 120

    while True:
        try:
            logger.info(f"Starting polling for @{bot_username}...")
            await bot_app.updater.start_polling(
                timeout=30,
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
            logger.info(f"Polling started successfully for @{bot_username}.")
            await asyncio.Future()
        except asyncio.CancelledError:
             logger.info(f"Polling cancelled for bot @{bot_username}. Stopping.")
             await bot_app.updater.stop()
             await bot_app.stop()
             await bot_app.shutdown()
             logger.info(f"Bot @{bot_username} shut down gracefully.")
             break
        except Exception as e:
            attempt += 1
            delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, base_delay)
            delay = min(delay, max_delay)
            logger.error(f"Polling error for bot @{bot_username} (Attempt {attempt}): {e}. Retrying in {delay:.2f} seconds...", exc_info=True)
            await asyncio.sleep(delay)

# ----------------------
# FastAPI Server & Endpoints
# ----------------------
fastapi_app = FastAPI(
    title="TISM v5 Monitor",
    description="API for monitoring and controlling the TISM Telegram bot framework.",
    version="5.0.0"
)

@fastapi_app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"[API] Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"[API] Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"[API] Error handling request {request.method} {request.url.path}: {e}", exc_info=True)
        return Response("Internal Server Error", status_code=500)

@fastapi_app.get("/")
async def index() -> Dict[str, Any]:
    return {
        "message": "TISM V5: Telegram bot framework running.",
        "loaded_models": len(models),
        "configured_bots": len(BOT_TOKENS),
        "endpoints": {
            "/models": "List currently loaded models",
            "/stats": "Get basic usage statistics",
            "/reload_config": "Reload non-critical settings from config.ini (POST)",
            "/metrics": "Prometheus metrics endpoint",
            "/chat": "Latest chat logs (JSON)",
            "/chat/stream": "Real-time chat log stream (SSE)"
        }
     }

@fastapi_app.get("/models")
async def get_models() -> List[Dict[str, Any]]:
    return [
        {
            "id": m.get("id"),
            "endpoint": m.get("endpoint"),
            "current_weight": model_weights.get(m.get("id"), "N/A")
         } for m in models
     ]

@fastapi_app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    return {
        "messages_processed": stats.get("messages_processed", 0),
        "errors": stats.get("errors", 0),
        "model_weights": model_weights
    }

@fastapi_app.post("/reload_config")
async def reload_config_endpoint() -> Dict[str, str]:
    global DEFAULT_SYSTEM_PROMPT, REPLY_CHANCE, MAX_TOKENS, MAX_MODEL_RETRIES, RETRY_DELAY, BLACKLISTED_MODELS
    global TTS_API_KEY, TTS_ENDPOINT, TTS_MODEL, TTS_VOICE, TTS_SPEED, AVAILABLE_VOICES
    logger.info("[API] Received request to reload configuration...")
    try:
        new_config = load_config()

        DEFAULT_SYSTEM_PROMPT = clean_config_value(new_config.get("settings", "system_prompt", fallback=DEFAULT_SYSTEM_PROMPT))
        REPLY_CHANCE = float(clean_config_value(new_config.get("settings", "reply_chance", fallback=str(REPLY_CHANCE))))
        MAX_TOKENS = int(clean_config_value(new_config.get("settings", "max_tokens", fallback=str(MAX_TOKENS))))
        MAX_MODEL_RETRIES = int(clean_config_value(new_config.get("settings", "max_model_retries", fallback=str(MAX_MODEL_RETRIES))))
        RETRY_DELAY = float(clean_config_value(new_config.get("settings", "retry_delay", fallback=str(RETRY_DELAY))))
        BLACKLISTED_MODELS = set(
            m.strip() for m in clean_config_value(new_config.get("settings", "blacklist_models", fallback=",".join(BLACKLISTED_MODELS))).split(",") if m.strip()
        )

         # Reload TTS section
        TTS_API_KEY = clean_config_value(new_config.get("tts", "api_key", fallback=TTS_API_KEY))
        TTS_ENDPOINT = clean_config_value(new_config.get("tts", "endpoint", fallback=TTS_ENDPOINT))
        TTS_MODEL = clean_config_value(new_config.get("tts", "model", fallback=TTS_MODEL))
        TTS_VOICE = clean_config_value(new_config.get("tts", "voice", fallback=TTS_VOICE))
        TTS_SPEED = float(clean_config_value(new_config.get("tts", "speed", fallback=str(TTS_SPEED))))
        AVAILABLE_VOICES = [v.strip() for v in new_config.get("tts", "available_voices", fallback=",".join(AVAILABLE_VOICES)).split(",") if v.strip()]

        logger.info("[API] Configuration reloaded successfully.")
        return {"message": "Config reloaded successfully."}
    except Exception as e:
        logger.error(f"[API] Failed to reload configuration: {e}", exc_info=True)
        return {"message": f"Error reloading config: {e}"}

@fastapi_app.get("/metrics")
async def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@fastapi_app.get("/chat")
async def get_chat_logs() -> List[Dict[str, Any]]:
    """
    Return the latest chat logs.
    """
    return chat_logs[-50:]

@fastapi_app.get("/chat/stream")
async def stream_chat_logs():
    """
    Real-time streaming of chat logs using Server-Sent Events (SSE).
    """
    async def event_generator():
        last_index = 0
        while True:
            if last_index < len(chat_logs):
                new_logs = chat_logs[last_index:]
                for log_entry in new_logs:
                    yield f"data: {json.dumps(log_entry)}\n\n"
                last_index = len(chat_logs)
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ----------------------
# Combined Startup: Run Telegram bots and the FastAPI server concurrently.
# ----------------------
async def main() -> None:
    telegram_tasks = [
        asyncio.create_task(start_telegram_bot_instance(token, prompt, username))
        for token, prompt, username in zip(BOT_TOKENS, SYSTEM_PROMPTS, BOT_USERNAMES)
    ]

    uvicorn_config = uvicorn.Config(app=fastapi_app, host="0.0.0.0", port=8007, loop="asyncio", log_level="info")
    server = uvicorn.Server(config=uvicorn_config)
    server_task = asyncio.create_task(server.serve())

    logger.info("Starting FastAPI server and Telegram bot instances...")
    done, pending = await asyncio.wait(
        telegram_tasks + [server_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in done:
        try:
            await task
            logger.info(f"Task {task.get_name()} completed normally.")
        except Exception as e:
            logger.error(f"Task {task.get_name()} failed: {e}", exc_info=True)

    logger.info("One task finished or failed, cancelling pending tasks...")
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Task {task.get_name()} cancelled successfully.")
        except Exception as e:
             logger.error(f"Error during cancellation of task {task.get_name()}: {e}", exc_info=True)

    logger.info("Application shutdown.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt.")
    except Exception as e:
         logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
