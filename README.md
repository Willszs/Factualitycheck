# AI Factuality Comparison & Question Generator Tool (AI 事实性对比测评与测试助手)

一个专为大语言模型（LLM）多轮深度对话设计的高性能、免切窗口事实性（Factuality）对比评测与测试集全自动生成工具。

集成了 **Telegram 手机端双向交互**、**桌面全局免切窗口快捷键**、**Gemini 3.1 Pro 旗舰级深度评测**、**高级拟人化打字模拟** 以及 **桌面弹窗提醒自由开关**，实现全自动化、无干扰的闭环评测体验。

---

## 🌟 核心特色与功能

### 1. ⚡ 免切窗口零点击工作流（Zero-Click Global Hotkeys）
在任何浏览器、文档或聊天软件中，无需切换回评测软件窗口，直接通过系统全局快捷键完成录入与发送：
- **`⌥⌘3`（Option + Command + 3）**：在其他应用中复制题目（`⌘C`）后直接按快捷键，系统自动读取剪贴板主题、更新界面并**立即发送到手机 Telegram 机器人**发起测试出题，无需点击任何发送按钮。
- **`⌥⌘1`（Option + Command + 1）**：在任意窗口复制 Model A 的多轮输出后直接存入【粘贴1】。
- **`⌥⌘2`（Option + Command + 2）**：在任意窗口复制 Model B 的多轮输出后直接存入【粘贴2】。
- **双模型填满即发**：当【粘贴1】与【粘贴2】均存入内容时，系统**自动触发事实性深度评测**并异步推送至手机，全程无需用鼠标点击软件界面。
- **备用手动发送**：支持 `⌥⌘S` 或 `⌥⌘↩`（Option + Command + Return）重新手动提交评测。
- *注：底层基于自研 `RobustGlobalHotKeys`，已针对 macOS 系统 Option 死键转译（如 `£`, `¡`, `™`, `ß`）进行物理键码原生适配，灵敏可靠。*

### 2. 🧠 旗舰级 Gemini 3.1 Pro 评测与自然人语言风格
- **模型升级**：默认搭载 Google 旗舰级深度推理模型 **`gemini-3.1-pro-preview`**，完全支持 Tier 1 账号高并发，平均 5 秒内极速生成高可信度对比报告。
- **自然审阅语言风格**：彻底摒弃教科书大段八股与学术论文冗余套话，采用富有穿透力、直击要害的自然评测语言。
- **严格字数约束**：报告各段严格精炼受控（P1 ≤ 150 词，P2 ≤ 150 词，全篇不超过 300 词），包含整体裁决（Verdict）与两模型事实缺陷（Flaws & Ground Truth）。

### 3. ⌨️ 高级拟人打字模拟器（Advanced Human Typer）
手机 Telegram 端收到评测问题或修改意见后，支持一键点击自动打字输入（Auto-Type）到工作窗口，具备极高拟真度：
- **QWERTY 物理相邻键打错字**：模拟人类打字时手指滑键击中临近按键（如 `e` 误打为 `w`/`r`，`k` 误打为 `j`/`l`）。
- **拟人迟疑反应**：发现错字后有 180ms ~ 380ms 的认知停顿。
- **真实退格删除**：模拟发送物理 Backspace 退格键信号逐字撤回错字并重新敲入正确字符。
- **节奏起伏**：模拟人类打字的短突发（bursts）与随机长短思考停顿，完全消除机械匀速感。

### 4. 🔔 桌面弹窗提醒自由开关（Notification Switch）
- 界面底部配有 **【🔔 桌面弹窗提醒】** 独立开关。
- **可随时勾选开启或关闭**：关闭后进入完全静音免打扰模式（不弹出任何 macOS 桌面通知横幅），界面底部状态栏仍会静默反馈；开启后享受实时横幅与音效反馈。
- **配置持久化**：开关状态自动同步保存至 `config.json`（`"enable_notifications": true/false`），下次启动自动沿用。

### 5. 📱 手机 Telegram 双向全闭环交互
- 桌面按下 `⌥⌘3` 后，手机 Telegram 立即弹出提问：“这段对话大概持续多久/需要几轮？”。
- 手机直接回复（如 `5轮`、`10分钟` 或 `3轮对比`），Bot 立即推送**第 1 轮高难度事实性考察问题**。
- 附带内嵌快捷按钮：
  - **`[🔄 换一个提问内容]`**：AI 变换考察维度重新生成当轮题目。
  - **`[➡️ 确认，获取下一轮]`**：确认当前轮次完成，获取深入递进的下一轮问题。
- 两段对话收集完毕后，通过 `⌥⌘1` 和 `⌥⌘2` 录入，手机端即刻接收纯英文事实性对比报告。

---

## 🔄 全闭环流程示意图

```
[1. 任何窗口 ⌘C 复制题目] ➡️ 按 ⌥⌘3 自动提取并发送
                                    ⬇️
[2. 手机 Telegram 回复轮数] ➡️ [3. 手机接收高难度测试题 & 交互换题]
                                    ⬇️
[4. 在两模型中对话并复制] ➡️ 按 ⌥⌘1 与 ⌥⌘2（两边齐全后自动开始评测）
                                    ⬇️
[5. 手机 Telegram 接收精炼英文对比报告 (Verdict / Flaws / Ground Truth)]
```

---

## 🛠️ 配置文件说明 (`config.json`)

项目根目录的 `config.json` 控制核心服务行为：

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

- `gemini_model`: 推荐使用 `gemini-3.1-pro-preview`（旗舰深度评测）或 `gemini-2.5-flash`（超高并发）。
- `enable_notifications`: 是否开启桌面 macOS 弹窗提醒，可直接在软件界面底部勾选切换。
- `telegram`: 配置专属 Bot Token 与目标 Chat ID 即可启用手机端双向联动。

---

## 🚀 启动与使用方式

### 方式一：Finder 中直接双击（推荐）
在访达（Finder）的文件夹中直接双击：
👉 **`FactualityCheck.command`**
*（已配置虚拟环境与防误退保护，双击即开即用）*

### 方式二：终端命令行启动
```bash
# 激活环境并启动桌面客户端
./run.sh
```

### 运行自动化测试
项目中包含了对事实性评测、拟人打字机、macOS 物理键码转译及配置持久化的全面测试：
```bash
.venv/bin/python test_factuality.py
```
*(全部 20 项单元测试自动化通过)*
