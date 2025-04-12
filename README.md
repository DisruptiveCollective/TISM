# TISM v5 – Totally Insane Synthetic Machines (Version 5)

TISM v5 is an advanced Telegram bot framework built for roleplaying and interactive chat experiences using vLLM endpoints. This version introduces enhanced metrics, improved logging, and a real-time chat log streaming feature through FastAPI endpoints.

## Features

- **Telegram Bot Framework:**  
  Interact with users via Telegram with support for custom personalities, dynamic model selection, and text-to-speech (TTS) responses.

- **vLLM Integration:**  
  Utilizes weighted model selection to choose between multiple vLLM endpoints for generating context-aware responses.

- **Enhanced Logging & Metrics:**  
  - Global logging of every conversation (questions and replies) with timestamps and metadata.
  - Prometheus integration for monitoring message processing, errors, and model response latencies.

- **Chat Log Endpoints:**  
  - `/chat`: A JSON endpoint returning the last 50 conversation entries.
  - `/chat/stream`: A Server-Sent Events (SSE) endpoint for real-time streaming of new log entries.

- **FastAPI Integration:**  
  Provides various REST endpoints for monitoring, configuration reload, and metrics exposure.

- **Bot Commands:**  
  Built-in commands including:
  - `/start`, `/help`, `/models`
  - `/setmodel`, `/unsetmodel`
  - `/reload_models`, `/credits`, `/stats`
  - `/setpersonality`, `/clearpersonality`
  - `/tts_on`, `/tts_off`, `/listvoices`, `/setvoice`

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**

   Ensure you have Python 3.8+ installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration:**

   - Create a `config.ini` file in the project root.
   - Populate it with your Telegram bot tokens, system prompts, vLLM endpoints, TTS settings, and any other configuration details as needed.

## Usage

To start TISM v5, run the main script:

```bash
python tism_v5.py
```

This will:
- Initialize the Telegram bot instances.
- Start a FastAPI server (default on port 8007) exposing various endpoints (e.g., metrics at `/metrics`, chat logs at `/chat`, and SSE stream at `/chat/stream`).

## API Endpoints

- **GET `/`**  
  Returns basic information about the TISM instance, including the number of loaded models and available endpoints.

- **GET `/models`**  
  Lists detailed information about each registered model including current weight and endpoint.

- **GET `/stats`**  
  Displays usage statistics like messages processed, errors encountered, and current model weights.

- **POST `/reload_config`**  
  Reloads non-critical configuration values from `config.ini`.

- **GET `/metrics`**  
  Exports Prometheus metrics for monitoring (compatible with Prometheus server).

- **GET `/chat`**  
  Returns the latest 50 conversation log entries (JSON format).

- **GET `/chat/stream`**  
  Provides a real-time stream of chat log entries using Server-Sent Events (SSE).

## Bot Commands

Users interacting with the Telegram bot can utilize the following commands:

- **/start:**  
  Initiates a conversation with the bot.

- **/help:**  
  Displays a help message listing available commands.

- **/models:**  
  Lists currently loaded models.

- **/setmodel &lt;number&gt; /unsetmodel:**  
  Set or clear a chat-specific model.

- **/reload_models:**  
  Refreshes the list of available models from vLLM endpoints.

- **/credits:**  
  Displays credits for the project.

- **/stats:**  
  Shows usage statistics.

- **/setpersonality &lt;text&gt; /clearpersonality:**  
  Customizes or clears the chat personality settings.

- **/tts_on /tts_off:**  
  Enable or disable TTS functionality.

- **/listvoices /setvoice &lt;number&gt;:**  
  Manage TTS voice configurations.

## Logging

- **Global Chat Logging:**  
  Every conversation turn (user question & bot reply) is logged with a timestamp, chat ID, bot username, question, and reply.  
  This log is accessible via REST endpoints.

- **Real-Time Streaming:**  
  The `/chat/stream` endpoint uses SSE to allow clients to subscribe to a real-time feed of new conversation entries.

## Contributing

Contributions and suggestions for improvement are welcome! Feel free to create issues or open pull requests. Please ensure you adhere to the existing coding style and add appropriate tests when necessary.

## License

This project is open source and available under the [MIT License](LICENSE).
