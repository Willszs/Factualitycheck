"""
Interactive Telegram Bot Service for Factuality Check Tool.
Handles bidirectional communication:
1. Multi-turn question design and regeneration (Topic -> Rounds -> Questions with inline buttons).
2. Remote Human-like Typing Simulator (Copy text on phone -> Bot asks confirmation -> Types on Mac like human).
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
from typer import HumanTyper

logger = logging.getLogger("factuality.telegram_bot")


class TelegramBotService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        tg_cfg = config.get("telegram", {})
        self.bot_token = tg_cfg.get("bot_token", "").strip()
        self.target_chat_id = str(tg_cfg.get("chat_id", "")).strip()

        self.generator = QuestionGenerator(config)

        # Topic & Benchmark Session state
        self.state = "IDLE"  # IDLE, WAITING_DURATION, INTERACTIVE_QUESTIONS
        self.topic = ""
        self.total_rounds = 3
        self.current_round = 1
        self.duration_desc = ""
        self.history_questions = []

        # Typing simulation buffer
        self.pending_typing_text = ""
        self.current_factuality_report = ""
        self.is_typing_active = False

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

    def deliver_factuality_report(self, title: str, summary: str) -> bool:
        """
        Delivers the factuality check report (structured into two dimensions with an empty line),
        and offers interactive options:
        1. 准备好了，直接打字 (Ready, direct typing)
        2. 我需要修改 (Need to edit before typing)
        """
        # Preserve the empty line between the two evaluation dimensions
        clean_summary = re.sub(r"\r\n", "\n", summary).strip()
        clean_summary = re.sub(r"\n{3,}", "\n\n", clean_summary)

        self.current_factuality_report = clean_summary
        self.pending_typing_text = clean_summary
        self.state = "IDLE"

        escaped_title = html.escape(title)
        escaped_body = html.escape(clean_summary)

        msg = (
            f"📊 <b>{escaped_title}</b>\n\n"
            f"{escaped_body}\n\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<i>🤖 是否需要在电脑当前光标处自动打出此报告？</i>"
        )

        keyboard = [
            [
                {"text": "⌨️ 准备好了，直接打字", "callback_data": "action_ready_direct_type"},
                {"text": "✏️ 我需要修改", "callback_data": "action_need_edit_report"},
            ]
        ]
        return self._send_message(msg, reply_markup={"inline_keyboard": keyboard})

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

            # Check states
            if self.state == "WAITING_REPORT_EDIT":
                self.pending_typing_text = text
                self.state = "IDLE"
                preview = text if len(text) <= 120 else text[:115] + "..."
                confirm_msg = (
                    f"⌨️ <b>已接收修改后的新内容（共 {len(text)} 字）：</b>\n"
                    f"<blockquote>{html.escape(preview)}</blockquote>\n\n"
                    f"<b>准备好了吗？</b>\n"
                    f"请把电脑鼠标光标点进目标输入框，点击下方按钮开始（3 秒倒计时准备）："
                )
                keyboard = [
                    [
                        {"text": "▶️ 准备好了，开始打字", "callback_data": "action_start_typing"},
                        {"text": "❌ 取消", "callback_data": "action_cancel_typing"},
                    ]
                ]
                self._send_message(confirm_msg, reply_markup={"inline_keyboard": keyboard})
            elif self.state == "WAITING_DURATION":
                self._handle_duration_reply(text)
            else:
                self._send_message(
                    "💡 提示：自动打字功能专属于【事实排查报告】推送场景。\n"
                    "请在电脑端将两个 AI 模型的对话分别填入【粘贴1】和【粘贴2】，点击【发送到手机】即可在此接收报告并使用自动打字。"
                )

        # 2. Handle callback queries from inline buttons
        elif "callback_query" in update:
            query = update["callback_query"]
            chat_id = str(query.get("message", {}).get("chat", {}).get("id", ""))
            if chat_id != self.target_chat_id:
                return

            query_id = query.get("id")
            data = query.get("data", "")
            message_id = query.get("message", {}).get("message_id")
            self._answer_callback_query(query_id)
            self._handle_callback_data(data, message_id)

    def _handle_duration_reply(self, text: str):
        """User replied with duration/rounds."""
        text_clean = text.strip()
        cn_to_num = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
            "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
        }

        # Check if duration mentions minutes (分钟/分)
        is_minutes = "分" in text_clean or "min" in text_clean.lower()

        num_val = None
        digits = re.findall(r"\d+", text_clean)
        if digits:
            num_val = int(digits[0])
        else:
            for cn_str, val in sorted(cn_to_num.items(), key=lambda x: len(x[0]), reverse=True):
                if cn_str in text_clean:
                    num_val = val
                    break

        if is_minutes:
            minutes = num_val or 10
            # 1 minute per turn pacing: 10 mins -> 10 rounds, 15 mins -> 15 rounds
            self.total_rounds = max(1, min(minutes, 30))
            self.duration_desc = f"{minutes}分钟真人嘴巴口语交流（每轮发问30-65字，5-10秒念完）"
        elif num_val:
            # User explicitly requested N rounds (e.g. 15轮) -> Give exactly N rounds!
            self.total_rounds = max(1, min(num_val, 30))
            self.duration_desc = f"{self.total_rounds}轮真人口语交流（每轮发问30-65字，5-10秒念完）"
        else:
            self.total_rounds = 5
            self.duration_desc = "真人嘴巴口语交流（每轮发问30-65字，5-10秒念完）"

        self.state = "INTERACTIVE_QUESTIONS"
        self.current_round = 1

        self._send_chat_action("typing")
        self._send_message(
            f"✅ 收到！针对 <b>{self.duration_desc}</b> 场景，已规划 <b>{self.total_rounds} 轮</b>地道口语提问（每轮仅需 5-10 秒随口读出）。\n"
            f"正在针对【{html.escape(self.topic)}】设计第 1 轮口语提问..."
        )

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
                {"text": "🔄 换一个提问", "callback_data": "action_change_q"},
            ]
        ]

        if round_num < self.total_rounds:
            keyboard.append([{"text": "➡️ 确认，获取下一轮", "callback_data": "action_next_round"}])
        else:
            keyboard.append([{"text": "✅ 完成测试，准备对比", "callback_data": "action_finish"}])

        keyboard.append([{"text": "⏹️ 结束本轮对话", "callback_data": "action_end_dialogue"}])

        reply_markup = {"inline_keyboard": keyboard}
        self._send_message(msg, reply_markup=reply_markup)

    def _handle_callback_data(self, data: str, message_id: Optional[int]):
        """Handles inline button clicks."""
        # --- Factuality Check Direct Typing & Editing ---
        if data == "action_ready_direct_type":
            if not self.pending_typing_text:
                self.pending_typing_text = self.current_factuality_report
            if not self.pending_typing_text:
                self._send_message("⚠️ 暂无事实排查内容可输入。")
                return

            preview = self.pending_typing_text if len(self.pending_typing_text) <= 120 else self.pending_typing_text[:115] + "..."
            confirm_msg = (
                f"⌨️ <b>准备向电脑输入事实排查报告（共 {len(self.pending_typing_text)} 字）：</b>\n"
                f"<blockquote>{html.escape(preview)}</blockquote>\n\n"
                f"<b>准备好了吗？</b>\n"
                f"请把电脑鼠标光标点进目标输入框，点击下方按钮开始（3 秒倒计时准备）："
            )
            keyboard = [
                [
                    {"text": "▶️ 准备好了，开始打字", "callback_data": "action_start_typing"},
                    {"text": "❌ 取消", "callback_data": "action_cancel_typing"},
                ]
            ]
            self._send_message(confirm_msg, reply_markup={"inline_keyboard": keyboard})
            return

        elif data == "action_need_edit_report":
            self.state = "WAITING_REPORT_EDIT"
            edit_msg = (
                f"✏️ <b>我需要修改：</b>\n\n"
                f"请直接在 Telegram 中回复您<b>修改后的新内容</b>：\n"
                f"<i>（提示：您可以复制上方报告内容，在手机输入框修改完成后直接发给我）</i>"
            )
            keyboard = [
                [
                    {"text": "❌ 取消修改", "callback_data": "action_cancel_edit"},
                ]
            ]
            self._send_message(edit_msg, reply_markup={"inline_keyboard": keyboard})
            return

        elif data == "action_cancel_edit":
            self.state = "IDLE"
            self.pending_typing_text = self.current_factuality_report
            self._send_message("已取消修改，保留原始报告内容。")
            return

        # --- Typing Simulation Actions ---
        elif data == "action_start_typing":
            if not self.pending_typing_text:
                self._send_message("⚠️ 没有待输入的文本，请先发送一段文本给我。")
                return

            if self.is_typing_active:
                self._send_message("⚠️ 当前正在进行打字任务，请稍候...")
                return

            text_to_type = self.pending_typing_text
            self.pending_typing_text = ""

            countdown_msg = (
                f"⏳ <b>倒计时 3 秒后开始在电脑上打字！</b>\n"
                f"请把鼠标光标定位在目标输入框中，不要切换窗口..."
            )
            if message_id:
                self._edit_message_text(message_id, countdown_msg)
            else:
                self._send_message(countdown_msg)

            threading.Thread(
                target=self._execute_typing_task,
                args=(text_to_type, message_id),
                daemon=True,
            ).start()

        elif data == "action_cancel_typing":
            self.pending_typing_text = ""
            cancel_msg = "❌ <b>已取消打字任务。</b>"
            if message_id:
                self._edit_message_text(message_id, cancel_msg)
            else:
                self._send_message(cancel_msg)

        # --- Question Design Actions ---
        elif data == "action_change_q":
            if self.state != "INTERACTIVE_QUESTIONS" or not self.topic:
                self._send_message("⚠️ 当前没有进行中的评测。程序处于 Standby 状态，请在电脑桌面端提交新主题。")
                return

            self._send_chat_action("typing")
            self._send_message(f"🔄 正在为您换一个全新角度的第 {self.current_round} 轮提问...")

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
            if self.state != "INTERACTIVE_QUESTIONS" or not self.topic:
                self._send_message("⚠️ 当前没有进行中的评测。程序处于 Standby 状态，请在电脑桌面端提交新主题。")
                return

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

        elif data == "action_end_dialogue":
            self.state = "IDLE"
            self.topic = ""
            self.current_round = 1
            self.total_rounds = 0
            self.history_questions = []
            self.duration_desc = ""
            standby_msg = (
                "⏹️ <b>本轮对话已结束。</b>\n\n"
                "程序已进入 <b>Standby（待命）</b> 状态，等待下一轮新题目。\n"
                "随时在电脑桌面端输入并提交新主题即可开始新测评！"
            )
            self._send_message(standby_msg)

    def _execute_typing_task(self, text: str, message_id: Optional[int]):
        """Runs typing simulation in background and notifies user on completion."""
        self.is_typing_active = True
        try:
            success, err_msg = HumanTyper.type_like_human(text, countdown_secs=3)
            if success:
                done_msg = f"🎉 <b>模拟打字已完成！</b>（共输入 {len(text)} 字）"
                self._send_message(done_msg)
            else:
                self._send_message(f"⚠️ {err_msg}")
        except Exception as e:
            logger.error(f"Typing execution failed: {e}")
            self._send_message(f"⚠️ 模拟打字出错: {e}")
        finally:
            self.is_typing_active = False

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
        if len(text) > 4000:
            text = text[:3900] + "\n\n...[内容过长已截断]..."

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
                    # If HTML parsing failed (400 Bad Request), fallback to plain text
                    if resp.status_code == 400:
                        plain_payload = dict(payload)
                        plain_payload.pop("parse_mode", None)
                        plain_payload["text"] = re.sub(r"<[^>]+>", "", text)
                        fallback_resp = requests.post(url, json=plain_payload, timeout=15, verify=verify)
                        if fallback_resp.status_code == 200:
                            return True
                except Exception:
                    if not verify:
                        break
        return False

    def _edit_message_text(self, message_id: int, text: str, reply_markup: Optional[dict] = None) -> bool:
        if len(text) > 4000:
            text = text[:3900] + "\n\n...[内容过长已截断]..."

        url = f"https://api.telegram.org/bot{self.bot_token}/editMessageText"
        payload = {
            "chat_id": self.target_chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        if requests is not None:
            for verify in [True, False]:
                try:
                    resp = requests.post(url, json=payload, timeout=10, verify=verify)
                    if resp.status_code == 200:
                        return True
                    if resp.status_code == 400:
                        plain_payload = dict(payload)
                        plain_payload.pop("parse_mode", None)
                        plain_payload["text"] = re.sub(r"<[^>]+>", "", text)
                        fallback_resp = requests.post(url, json=plain_payload, timeout=10, verify=verify)
                        if fallback_resp.status_code == 200:
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
