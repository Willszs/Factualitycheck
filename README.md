# AI Factuality Comparison Tool (AI 答案真实性对比评估工具)

一个极简本地 Python 桌面应用，用于多轮对话场景下模型 A 与 模型 B 的事实性（Factuality）对比评估。

## 特性亮点
- **极简双栏录入**：左侧粘贴 Model A 多轮对话，右侧粘贴 Model B 多轮对话。
- **即投即清，无缝输入**：点击“发送并评估”或按快捷键 `Cmd+Enter`，输入框立即清空，光标自动复位，支持连续批量录入。
- **界面绝对无结果**：GUI 界面绝不呈现任何评估报告，保证工作台整洁高效。
- **异步后台评估**：多线程队列处理 Gemini 大模型 API 分析与推送，UI 绝不卡顿或假死。
- **手机即时推送**：分析报告自适应长度生成纯英文总结，直接推送至手机（首选 Telegram，亦支持 ntfy/Bark 等），包含：
  - `The Verdict`（整体评价与对比）
  - `Model A's Flaws`（错误轮数 Round #、错误点、真实 Ground Truth，无错则标注 Flawless）
  - `Model B's Flaws`（错误轮数 Round #、错误点、真实 Ground Truth，无错则标注 Flawless）

---

## 快速配置指引 (`config.json`)

打开项目目录下的 `config.json`：

```json
{
  "gemini_api_key": "你的_GEMINI_API_KEY",
  "gemini_model": "gemini-2.5-flash",
  "push_channel": "telegram",
  "telegram": {
    "bot_token": "你的_TELEGRAM_BOT_TOKEN",
    "chat_id": "你的_TELEGRAM_CHAT_ID"
  },
  "ntfy": {
    "topic": "自定义topic"
  },
  "bark": {
    "device_key": "你的_BARK_KEY",
    "server_url": "https://api.day.app"
  }
}
```

### 1. 获取 Gemini API Key
前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免费创建获取。

### 2. 获取 Telegram Bot Token & Chat ID（推荐，约 1 分钟）
1. 在 Telegram 搜索并打开 `@BotFather`，发送 `/newbot`，按提示设置后复制生成的 **Bot Token**。
2. 在 Telegram 搜索并打开 `@userinfobot`，点击 Start，复制它回复给你的 **Id**（纯数字，即 `chat_id`）。
3. 在 Telegram 搜索打开你刚刚创建的 Bot，点击一次 **Start** 激活对话。
4. 将 `bot_token` 和 `chat_id` 填入 `config.json` 的 `telegram` 字段。

> **备选（免注册 ntfy 方案）**：
> 如果临时不想创建 Telegram Bot，可将 `push_channel` 设为 `"ntfy"`，手机下载 `ntfy` App，订阅你在 `config.json` 里写的随机 topic（如 `fact-check-shuzi`），即可免注册接收推送。

---

## 启动方式

在终端运行：
```bash
./run.sh
```
或者：
```bash
.venv/bin/python app.py
```

### 快捷键
- `Cmd + Enter` (macOS)：快速提交并评估
