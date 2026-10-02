# AI Factuality Comparison & Question Generator Tool

A high-performance, distraction-free desktop application engineered for benchmarking and factuality evaluation between Large Language Models (LLMs) across complex multi-turn dialogues.

Integrates **bidirectional Telegram mobile control**, **system-wide zero-click global hotkeys**, **Google Gemini 3.1 Pro forensic evaluation**, **advanced human-like typing simulation**, and **customizable desktop notification controls** for a fully automated, frictionless evaluation loop.

---

## 🌟 Key Features & Capabilities

### 1. ⚡ Zero-Click & Zero-Window-Switching Workflow (Global Hotkeys)
Evaluate models and generate test questions from any browser, editor, or chat window without ever switching back to the app window:
- **`⌥⌘3` (`Option + Command + 3`)**: Copy any topic or query (`⌘C`) in any application and hit `⌥⌘3`. The app instantly captures your clipboard, updates the UI, and **immediately dispatches it to your Telegram bot** to initiate interactive benchmark generation. Zero clicks required.
- **`⌥⌘1` (`Option + Command + 1`)**: Copy Model A's multi-turn transcript and store it into **Input 1 (Model A)**.
- **`⌥⌘2` (`Option + Command + 2`)**: Copy Model B's multi-turn transcript and store it into **Input 2 (Model B)**.
- **Auto-Evaluation on Dual Fill**: As soon as both model inputs contain text, the system **automatically triggers factuality evaluation** and pushes the report directly to your phone. No mouse clicks needed.
- **Manual Resend Backup**: `⌥⌘S` or `⌥⌘↩` (`Option + Command + Return`) to re-trigger evaluation at any time.
- *Engineered with custom `RobustGlobalHotKeys`: Fully handles macOS Option dead-key character mappings (`£`, `¡`, `™`, `ß`) via native virtual keycode normalization for 100% reliable shortcut triggers.*

### 2. 🧠 Flagship Gemini 3.1 Pro Evaluation with Natural Human Reviewer Tone
- **Flagship Engine**: Powered by Google's flagship reasoning model **`gemini-3.1-pro-preview`**, with full Tier 1 concurrency support delivering comprehensive forensic reports in ~5 seconds.
- **Natural Human Reviewer Voice**: Completely strips away formulaic academic jargon, generic textbook boilerplate, and evasive filler. Delivers direct, incisive, and pragmatic critique.
- **Strict Word Budget**: Structured into concise paragraphs strictly enforced to P1 ≤ 150 words and P2 ≤ 150 words (total ≤ 300 words), highlighting the overarching Verdict alongside specific Flaws and Ground Truth for each model.

### 3. ⌨️ Advanced Human-Like Typer Simulation (Auto-Type)
When receiving suggested questions or prompts on Telegram, tap the in-chat button to auto-type directly into your active target input with realistic human typing characteristics:
- **Physical QWERTY Adjacent Typos**: Simulates genuine finger slips to physically adjacent keys on the keyboard (e.g., mistyping `e` as `w`/`r`, or `k` as `j`/`l`).
- **Cognitive Hesitation**: Introduces realistic 180ms ~ 380ms pauses upon making an error before backspacing.
- **Hardware Backspace Correction**: Emulates physical macOS Backspace keypresses (`keycode 51`) to delete mistakes and retype the correct characters.
- **Natural Cadence**: Incorporates typing bursts, variable inter-key intervals, and irregular thinking pauses, eliminating robotic constancy.

### 4. 🔔 Customizable Desktop Notification Switch
- An integrated **`[√] 🔔 桌面弹窗提醒`** (`Desktop Popups`) checkbox located directly on the bottom control bar.
- **Instant Toggle**:
  - **Enabled**: Displays macOS native banner notifications with subtle sound feedback upon capturing text or completing evaluations.
  - **Disabled**: Runs in 100% silent focus mode (no desktop banners), while the status bar within the window continues to update smoothly.
- **Persistent State**: The toggle state automatically persists to `config.json` (`"enable_notifications": true/false`) across app launches.

### 5. 📱 Bidirectional Telegram Mobile Loop
- Press `⌥⌘3` on desktop; your Telegram bot immediately inquires: *"How many rounds or how long should this benchmark last?"*.
- Reply directly from your phone (e.g., `5 rounds`, `10 mins`, or `3-turn deep dive`).
- The bot instantly generates and sends **Round 1's precision factuality probe question** with interactive inline buttons:
  - **`[🔄 Re-roll Question]`**: Re-generates the question from a completely fresh perspective using AI.
  - **`[➡️ Confirm & Next Round]`**: Advances to the next deeper, sequential test round.
- After running the dialogues with both models, capture them via `⌥⌘1` and `⌥⌘2` to immediately receive the forensic verdict on your phone.

---

## 🔄 Closed-Loop Workflow Overview

```
[1. Copy topic in any app (⌘C)] ➡️ Press ⌥⌘3 (Auto-extracts & dispatches)
                                           ⬇️
[2. Reply target rounds on Telegram] ➡️ [3. Receive probe questions & interactive re-rolls]
                                           ⬇️
[4. Run dialogue in Model A & B]     ➡️ Press ⌥⌘1 & ⌥⌘2 (Auto-evaluates once both filled)
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
Run the comprehensive test suite covering factuality assessment, human-like typing simulation, macOS dead-key hotkey normalization, and configuration persistence:
```bash
.venv/bin/python test_factuality.py
```
*(All 20 unit tests automated and passing).*
