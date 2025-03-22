# TISM V1 TOTALLY INSANE SIMULATED MACHINES

Yo, welcome to **TISM** – a no-BS, rapid-deployment Telegram bot and web API that lets you chat with AI models like a boss. This project is all about bootstrapping, keeping things real, and using weighted model selection to deliver smart (and sometimes hilarious) replies. If you're all about getting sh*t done without over-engineering, you're in the right place.

---

## Features

- **Multi-Bot Setup:** Run multiple Telegram bot instances with unique tokens, system prompts, and personalities.
- **Weighted Model Selection:** Dynamically picks AI models based on performance weights.
- **Conversation History:** Maintains context for your chats (last 5 messages per chat) to keep the conversation flow.
- **Text-to-Speech (TTS):** Converts bot replies to voice messages using a TTS API.
- **Hot Config Reload:** Change non-bot settings on the fly with a simple API call.
- **FastAPI Server:** Provides endpoints for stats, model listing, and config reloading.
- **Robust Error Handling:** Retries model requests with exponential backoff in case of failure.
- **Monkey Patching Telegram:** Removes unwanted proxy settings from Telegram's HTTPX request for smoother operations.

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- [Telegram Bot API Token(s)](https://core.telegram.org/bots#6-botfather)
- [VLLM Endpoints](https://openai.com) (or your preferred AI service endpoints)
- [TTS API Key](#configuration) for text-to-speech functionality

### Installation

1. **Clone the Repo**

   ```bash
   git clone https://gitlab.com/DisruptiveCollective/tism.git
   cd tism
   ```

2. **Create a Virtual Environment & Install Dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Your Environment**

   Create a `config.ini` file in the root directory. Check out the [Configuration](#configuration) section below.

4. **Run the Bot & API Server**

   ```bash
   python main.py
   ```

---

## Configuration

Edit `config.ini` to set up your bots, models, and TTS settings. Here's a sample configuration:

```ini
[bots]
telegram_tokens = YOUR_TELEGRAM_TOKEN_1, YOUR_TELEGRAM_TOKEN_2
system_prompts = "You are a helpful assistant.", "You are a witty conversationalist."
bot_usernames = your_bot_username1, your_bot_username2

[vllm_endpoints]
endpoints = https://api.vllm1.example.com, https://api.vllm2.example.com

[settings]
system_prompt = "Default system prompt here."
reply_chance = 0.1
max_tokens = 200
blacklist_models = some-bad-model
max_model_retries = 3
retry_delay = 10

[tts]
api_key = YOUR_TTS_API_KEY
endpoint = https://api.tts.example.com/synthesize
model = tts-1
voice = echo
speed = 0.9
```

*Notes:*
- **Token Lists:** Make sure the number of tokens, prompts, and usernames match.
- **Model Weights:** These get adjusted dynamically based on performance.

---

## Telegram Bot Commands

- **/start**  
  Kick things off with a fun welcome message.

- **/help**  
  List all available commands and their usage.

- **/models**  
  Get a rundown of registered AI models and their endpoints.

- **/credits**  
  See who made the magic happen (spoiler: it's us, and we own your soul now 😜).

- **/stats**  
  Check the current usage stats (messages processed, errors, etc.).

- **/setpersonality \<text\>**  
  Override the default system prompt for your chat with a custom personality.

- **/clearpersonality**  
  Revert to the default system prompt by clearing the custom one.

---

## API Endpoints

The FastAPI server runs on port **8007** by default.

- **GET /**  
  Returns a basic welcome message.

- **GET /models**  
  Lists all available AI models with their endpoints.

- **GET /stats**  
  Returns usage statistics (e.g., messages processed, errors).

- **POST /reload_config**  
  Hot-reloads non-bot configuration settings from `config.ini`.

---

## How It Works

1. **Startup:**  
   - The script reads from `config.ini`, loads Telegram bot tokens, system prompts, and endpoints.
   - It fetches available AI models from the provided endpoints, applying a blacklist if necessary.

2. **Handling Messages:**  
   - Incoming Telegram messages are processed.
   - Uses weighted model selection to choose an AI model for generating replies.
   - Maintains a short conversation history for context.
   - Replies are sent both as text (split into chunks if too long) and voice messages using TTS.

3. **Error Handling & Retries:**  
   - If a model fails, the bot will retry with different models using exponential backoff.
   - Adjusts model weights based on performance to favor the best responders.

4. **API & Hot Reloading:**  
   - FastAPI serves endpoints for stats, models, and config reload.
   - This allows real-time monitoring and adjustments without restarting the bots.

---

## Contributing

Got ideas to make TISM even more badass? Fork the repo, make your changes, and submit a merge request. We value practical, no-fluff contributions that solve real problems.

---

## License

Distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Final Words

TISM is built for the disruptors, the digital nomads, and anyone tired of overcomplicated shit. Get it up and running, tweak it, and make it your own. Remember, action beats perfection any day. Now go out there and disrupt the status quo, one bot message at a time!

Peace out, and happy hacking.
