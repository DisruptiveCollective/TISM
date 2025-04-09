# TISM V4 – Totally Insane Synthetic Machines

Welcome to **TISM V4** – the next evolution of our Telegram bot framework for insane synthetic roleplaying powered by vllms. This version is leaner, meaner, and packed with modular design, robust error handling, and killer metrics. It's built to help you bootstrap your AI-powered Telegram bots faster than you can say "Nomad List"!

---

## Overview

TISM V4 is designed for:
- **Roleplaying and Chatbots:** Create wild, unpredictable conversations with AI.
- **Per-Chat Customization:** Set different personalities, models, and even TTS voices per chat.
- **Resilience & Observability:** Built-in error handling, exponential backoff, and Prometheus metrics keep your bot robust and production-ready.
- **FastAPI Integration:** Monitor your bot via an HTTP API with endpoints for status, models, config reloads, and metrics.

---

## Features

- **Multi-Bot Support:** Run multiple Telegram bot instances simultaneously.
- **Custom Configurations:** Easily adjust system prompts, model selections, and TTS settings via `config.ini`.
- **Dynamic Model Management:** Auto-fetches models from your vllm endpoints with weighted selection and retries.
- **Text-to-Speech (TTS):** Converts text responses to voice messages with multiple voice options.
- **Prometheus Metrics:** Expose real-time metrics for monitoring and observability.
- **FastAPI Server:** Lightweight web server to interact with your bot and view stats.

---

## Installation

### Requirements

Make sure you have the following Python libraries installed:

```txt
openai
python-telegram-bot>=20.0
fastapi
uvicorn
prometheus_client
```

You can install them using pip:

```bash
pip install openai python-telegram-bot>=20.0 fastapi uvicorn prometheus_client
```

### Clone the Repo

```bash
git clone https://github.com/DisruptiveCollective/tism.git
cd tism
```

### Configuration

Create a `config.ini` file in the project root. Here's a sample configuration:

```ini
[bots]
telegram_tokens = YOUR_TELEGRAM_TOKEN_1, YOUR_TELEGRAM_TOKEN_2
system_prompts = "Welcome to TISM!", "Hello from TISM!"
bot_usernames = tismbot1, tismbot2

[vllm_endpoints]
endpoints = https://vllm.endpoint1.com, https://vllm.endpoint2.com

[settings]
system_prompt = "Default system prompt for TISM"
reply_chance = 0.1
max_tokens = 200
blacklist_models = modelA, modelB
max_model_retries = 3
retry_delay = 10

[tts]
api_key = YOUR_TTS_API_KEY
endpoint = https://api.tts.example.com/synthesize
model = tts-1
voice = echo
speed = 0.9
available_voices = echo,default
```

---

## Usage

### Running the Bot

Simply run the main script:

```bash
python main.py
```

This will start:
- Your Telegram bot(s) (via the latest python-telegram-bot v20+).
- A FastAPI server on port 8007 with endpoints for info, stats, config reload, and Prometheus metrics.

### FastAPI Endpoints

- **GET /**: Basic info about TISM V4.
- **GET /models**: List of available models.
- **GET /stats**: Bot usage stats.
- **POST /reload_config**: Reload configuration from `config.ini`.
- **GET /metrics**: Prometheus metrics endpoint.

---

## Commands (Telegram)

- `/start` – Start the bot.
- `/help` – Display this help message.
- `/models` – List available models.
- `/setmodel <number>` – Select a model for the current chat.
- `/credits` – Show credits.
- `/stats` – Display bot usage statistics.
- `/setpersonality <text>` – Change the personality for the chat.
- `/clearpersonality` – Clear custom personality and revert to default.
- `/tts_on` – Enable text-to-speech responses.
- `/tts_off` – Disable text-to-speech responses.
- `/listvoices` – List available TTS voices.
- `/setvoice <number>` – Select a TTS voice for this chat.

---

## Contributing

Got ideas? Found a bug? Feel free to submit a pull request or open an issue on GitHub. We love community input—it's what keeps TISM wild and ever-evolving.

---

## License

TISM V4 is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Final Words

TISM V4 is all about rapid iteration and real-world problem solving. Get it up, tweak it, and make it your own. Whether you're building a quirky chat assistant or an AI roleplaying legend, this bot framework is your launchpad to disruption.

Happy hacking, and remember: keep it insane, keep it synthetic!  
— Pieter (if he were here, he'd say, "Just ship it.")

---

*Enjoy TISM V4 – where your wildest bot ideas come to life!*
