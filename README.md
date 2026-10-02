# AI Factuality Comparison & Question Generator Tool

A high-performance desktop application engineered for benchmarking and factuality evaluation between Large Language Models (LLMs) across complex multi-turn dialogues.

Integrates **bidirectional Telegram mobile interaction**, **Google Gemini 3.1 Pro forensic evaluation**, and **customizable desktop notification controls** for an automated, distraction-free evaluation loop.

---

## 🌟 Key Features & Capabilities

### 1. 🧠 Flagship Gemini 3.1 Pro Evaluation with Natural Human Reviewer Tone
- **Flagship Engine**: Powered by Google's flagship reasoning model **`gemini-3.1-pro-preview`**, supporting Tier 1 concurrency to deliver comprehensive forensic reports in ~5 seconds.
- **Natural Human Reviewer Voice**: Completely strips away formulaic academic jargon, generic textbook boilerplate, and evasive filler. Delivers direct, incisive, and pragmatic critique.
- **Strict Word Budget**: Structured into concise paragraphs strictly enforced to P1 ≤ 150 words and P2 ≤ 150 words (total ≤ 300 words), highlighting the overarching Verdict alongside specific Flaws and Ground Truth for each model.

### 2. 📱 Bidirectional Telegram Mobile Interaction
- **Dynamic Round Configuration**: Enter a dialogue topic on desktop; your Telegram bot immediately inquires: *"How many rounds or how long should this benchmark last?"*.
- **Interactive Question Generation**: Reply directly from your phone (e.g., `5 rounds`, `10 mins`, or `3-turn deep dive`). The bot generates **Round 1's precision factuality probe question** with interactive inline buttons:
  - **`[🔄 Re-roll Question]`**: Re-generates the question from a completely fresh perspective using AI.
  - **`[➡️ Confirm & Next Round]`**: Advances to the next deeper, sequential test round.
- **Instant Mobile Delivery**: As soon as evaluation completes, the concise forensic report is pushed directly to Telegram.

### 3. 🔔 Customizable Desktop Notification Switch
- An integrated **`[√] 🔔 桌面弹窗提醒`** (`Desktop Popups`) checkbox located on the bottom control bar.
- **Instant Toggle**:
  - **Enabled**: Displays macOS native banner notifications with subtle sound feedback upon dispatching tasks or completing evaluations.
  - **Disabled**: Runs in 100% silent focus mode (no desktop banners), while the status bar within the window continues to update smoothly.
- **Persistent State**: The toggle state automatically persists to `config.json` (`"enable_notifications": true/false`) across app launches.

### 4. ⚡ Rapid Evaluation Workflow
- **Asynchronous Execution**: Submitting dialogues clears the inputs in milliseconds and hands off the evaluation to a background daemon, keeping the UI completely fluid and responsive.
- **Keyboard Shortcut**: Press `Cmd + Return` (or `Option + Command + Return`) to trigger evaluation instantly.

---

## 🔄 Closed-Loop Workflow Overview

```
[1. Enter Topic on Desktop]          ➡️ [2. Reply target rounds on Telegram]
                                                ⬇️
[4. Enter Model A & B Dialogues]    ⬅️ [3. Receive probe questions & interactive re-rolls]
         ⬇️
[5. Receive concise English forensic report on phone (Verdict / Flaws / Ground Truth)]
```

---

## 🛠️ Configuration (`config.json`)

Configure core service credentials and options in `config.json`:

```json
{
  "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE",
  "gemini_model": "gemini-3.1-pro-preview",
  "push_channel": "telegram",
  "enable_notifications": true,
  "telegram": {
    "bot_token": "YOUR_TELEGRAM_BOT_TOKEN_HERE",
    "chat_id": "YOUR_TELEGRAM_CHAT_ID_HERE"
  },
  "ntfy": {
    "topic": "YOUR_NTFY_TOPIC_HERE"
  },
  "bark": {
    "device_key": "YOUR_BARK_KEY_HERE",
    "server_url": "https://api.day.app"
  }
}
```

- `gemini_model`: Recommended: `gemini-3.1-pro-preview` (flagship reasoning) or `gemini-2.5-flash` (high throughput).
- `enable_notifications`: Controls macOS desktop banner alerts (toggleable directly from the bottom bar in the UI).
- `telegram`: Fill in your Telegram Bot Token and Chat ID to enable mobile notifications and interactive generation.

---

## 🚀 Getting Started & Usage

### Method 1: Double-Click in Finder (Recommended)
Simply double-click the launcher script in the project directory:
👉 **`FactualityCheck.command`**
*(Automatically handles virtual environment activation and prevents terminal window auto-closure on errors).*

### Method 2: Launch via Terminal
```bash
./run.sh
```

### Running Automated Tests
Run the comprehensive test suite covering factuality assessment, dialogue auditing, and configuration persistence:
```bash
.venv/bin/python test_factuality.py
```
*(All 20 unit tests automated and passing).*
