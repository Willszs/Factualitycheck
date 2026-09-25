"""
Interactive Telegram Bot Service for Factuality Check Tool.
Handles bidirectional communication: polling user replies and inline keyboard actions
to orchestrate multi-turn question design and regeneration.
"""

import re
import json
import time
import logging
import threading
import html
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

from question_generator import QuestionGenerator

logger = logging.getLogger("factuality.telegram_bot")


class TelegramBotService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        tg_cfg = config.get("telegram", {})
        self.bot_token = tg_cfg.get("bot_token", "").strip()
        self.target_chat_id = str(tg_cfg.get("chat_id", "")).strip()

        self.generator = QuestionGenerator(config)

        # Session state
        self.state = "IDLE"  # IDLE, WAITING_DURATION, INTERACTIVE_QUESTIONS
        self.topic = ""
        self.total_rounds = 3
        self.current_round = 1
        self.duration_desc = ""
        self.history_questions = []

        self.last_update_id = 0
        self.is_running = False
        self.poll_thread = None

    def start(self):
        """Starts the background polling thread."""
        if not self.bot_token or not self.target_chat_id or "YOUR_" in self.bot_token:
            logger.warning("Telegram bot not configured; interactive service disabled.")
            return

        if self.is_running:
            return

        self.is_running = True
        self.poll_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.poll_thread.start()
        logger.info("Telegram interactive bot service started.")

    def stop(self):
        self.is_running = False

    def start_topic(self, topic: str):
        """Called by Desktop GUI when user submits a new topic."""
        topic = topic.strip()
        if not topic:
            return

        self.topic = topic
        self.state = "WAITING_DURATION"
        self.history_questions = []
        self.current_round = 1
        self.total_rounds = 3

        prompt_msg = (
            f"🎯 <b>收到新测评主题：</b>\n"
            f"<blockquote>{html.escape(topic)}</blockquote>\n\n"
            f"请直接回复这段对话大概需要持续<b>多少轮</b>或<b>多长时间</b>？\n"
            f"（例如直接回复：<code>5轮</code>、<code>10分钟</code>、<code>3轮对比</code>）"
        )
        self._send_message(prompt_msg)

    def _polling_loop(self):
        """Long polling loop for incoming messages and callback queries."""
        while self.is_running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self.last_update_id = update["update_id"] + 1
                    self._handle_update(update)
            except Exception as e:
                logger.debug(f"Error during Telegram update poll: {e}")
                time.sleep(2)
            time.sleep(0.5)

    def _get_updates(self):
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {"offset": self.last_update_id, "timeout": 15}

        if requests is not None:
            for verify in [True, False]:
                try:
                    resp = requests.get(url, params=params, timeout=25, verify=verify)
                    if resp.status_code == 200:
                        return resp.json().get("result", [])
                except Exception:
                    if not verify:
                        break
        return []

    def _handle_update(self, update: Dict[str, Any]):
        # 1. Handle incoming text message
        if "message" in update:
            msg = update["message"]
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if chat_id != self.target_chat_id:
                return  # Ignore messages from strangers

            text = msg.get("text", "").strip()
            if not text:
                return

            # Check if waiting for duration
            if self.state == "WAITING_DURATION":
                self._handle_duration_reply(text)

        # 2. Handle callback queries from inline buttons
        elif "callback_query" in update:
            query = update["callback_query"]
            chat_id = str(query.get("message", {}).get("chat", {}).get("id", ""))
            if chat_id != self.target_chat_id:
                return

            query_id = query.get("id")
            data = query.get("data", "")
            self._answer_callback_query(query_id)
            self._handle_callback_data(data, query.get("message", {}).get("message_id"))

    def _handle_duration_reply(self, text: str):
        """User replied with duration/rounds."""
        self.duration_desc = text
        # Extract number of rounds if possible
        numbers = re.findall(r"\d+", text)
        if numbers:
            rounds = int(numbers[0])
            self.total_rounds = max(1, min(rounds, 10))  # Bound between 1 and 10
        else:
            self.total_rounds = 3  # Default 3 rounds

        self.state = "INTERACTIVE_QUESTIONS"
        self.current_round = 1

        self._send_chat_action("typing")
        self._send_message(
            f"✅ 收到！已规划 <b>{self.total_rounds} 轮</b>深入测评。\n"
            f"正在针对主题【{html.escape(self.topic)}】设计第 1 轮高精度事实性测试问题..."
        )

        # Generate Round 1 question
        q_text = self.generator.generate_question(
            topic=self.topic,
            current_round=1,
            total_rounds=self.total_rounds,
            duration_desc=self.duration_desc,
            history_questions=[],
            is_alternative=False,
        )
        self.history_questions.append(q_text)
        self._deliver_question_card(1, q_text)

    def _deliver_question_card(self, round_num: int, q_text: str):
        """Pushes question card with inline interactive buttons."""
        msg = (
            f"💡 <b>【第 {round_num}/{self.total_rounds} 轮测试问题】</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{html.escape(q_text)}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<i>👉 复制上方问题向两个 AI 模型提问；问完后可点击下方按钮推进：</i>"
        )

        keyboard = [
            [
                {"text": "🔄 换一个提问内容", "callback_data": "action_change_q"},
            ]
        ]

        if round_num < self.total_rounds:
            keyboard[0].append({"text": "➡️ 确认，获取下一轮", "callback_data": "action_next_round"})
        else:
            keyboard[0].append({"text": "✅ 完成，准备对比", "callback_data": "action_finish"})

        reply_markup = {"inline_keyboard": keyboard}
        self._send_message(msg, reply_markup=reply_markup)

    def _handle_callback_data(self, data: str, message_id: Optional[int]):
        """Handles inline button clicks."""
        if data == "action_change_q":
            self._send_chat_action("typing")
            self._send_message(f"🔄 正在为您换一个全新角度的第 {self.current_round} 轮提问...")

            # Generate alternative question
            new_q = self.generator.generate_question(
                topic=self.topic,
                current_round=self.current_round,
                total_rounds=self.total_rounds,
                duration_desc=self.duration_desc,
                history_questions=self.history_questions,
                is_alternative=True,
            )
            if self.history_questions:
                self.history_questions[-1] = new_q
            else:
                self.history_questions.append(new_q)

            self._deliver_question_card(self.current_round, new_q)

        elif data == "action_next_round":
            if self.current_round < self.total_rounds:
                self.current_round += 1
                self._send_chat_action("typing")
                self._send_message(f"✨ 正在顺承上下文生成第 {self.current_round}/{self.total_rounds} 轮递进问题...")

                next_q = self.generator.generate_question(
                    topic=self.topic,
                    current_round=self.current_round,
                    total_rounds=self.total_rounds,
                    duration_desc=self.duration_desc,
                    history_questions=self.history_questions,
                    is_alternative=False,
                )
                self.history_questions.append(next_q)
                self._deliver_question_card(self.current_round, next_q)
            else:
                self._finish_flow()

        elif data == "action_finish":
            self._finish_flow()

    def _finish_flow(self):
        self.state = "IDLE"
        finish_msg = (
            f"🎉 <b>全部 {self.total_rounds} 轮测评题目设计完毕！</b>\n\n"
            f"请将两个 AI 模型的对话记录分别复制到电脑桌面软件的：\n"
            f"• <b>【粘贴1】</b>\n"
            f"• <b>【粘贴2】</b>\n\n"
            f"点击底部的<b>【发送到手机】</b>，我将立即为您深度分析并推送纯英文事实性对比报告（包含 Verdict、每轮错误与 Ground Truth）！"
        )
        self._send_message(finish_msg)

    def _send_message(self, text: str, reply_markup: Optional[dict] = None) -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.target_chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        if requests is not None:
            for verify in [True, False]:
                try:
                    resp = requests.post(url, json=payload, timeout=15, verify=verify)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    if not verify:
                        break
        return False

    def _send_chat_action(self, action: str = "typing"):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendChatAction"
        payload = {"chat_id": self.target_chat_id, "action": action}
        if requests is not None:
            try:
                requests.post(url, json=payload, timeout=5, verify=False)
            except Exception:
                pass

    def _answer_callback_query(self, query_id: str, text: str = "处理中..."):
        url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
        payload = {"callback_query_id": query_id, "text": text}
        if requests is not None:
            try:
                requests.post(url, json=payload, timeout=5, verify=False)
            except Exception:
                pass
