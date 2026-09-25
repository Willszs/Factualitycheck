"""
Verification script for Factuality Comparison Tool components.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from evaluator import FactualityEvaluator, SYSTEM_PROMPT
from notifier import Notifier
from question_generator import QuestionGenerator
from telegram_bot import TelegramBotService


class TestFactualityComponents(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            "gemini_api_key": "dummy_key",
            "gemini_model": "gemini-3.6-flash",
            "push_channel": "telegram",
            "telegram": {
                "bot_token": "123456:ABC-DEF",
                "chat_id": "987654321",
            },
            "ntfy": {
                "topic": "test-topic",
            },
        }

    def test_evaluator_unconfigured_key(self):
        evaluator = FactualityEvaluator({"gemini_api_key": "YOUR_GEMINI_API_KEY_HERE"})
        res = evaluator.evaluate("Round 1: Hello", "Round 1: World")
        self.assertIn("⚠️ Evaluation Failed", res)
        self.assertIn("Gemini API Key", res)

    def test_prompt_structure_contains_required_sections(self):
        self.assertIn("Verdict", SYSTEM_PROMPT)
        self.assertIn("Model A", SYSTEM_PROMPT)
        self.assertIn("Model B", SYSTEM_PROMPT)
        self.assertIn("Ground Truth", SYSTEM_PROMPT)
        self.assertIn("Turn [X]", SYSTEM_PROMPT)
        self.assertIn("Flawless", SYSTEM_PROMPT)
        self.assertIn("NO PARAGRAPH BREAKS", SYSTEM_PROMPT)

    def test_clean_single_paragraph(self):
        multiline_text = "### The Verdict\nModel A was better.\n\n### Model A's Flaws\n* Turn 1: Mistake. Ground Truth: Fact.\n\n### Model B's Flaws\nFlawless - No flaws detected."
        cleaned = FactualityEvaluator.clean_single_paragraph(multiline_text)
        self.assertNotIn("\n", cleaned)
        self.assertIn("Verdict: Model A was better.", cleaned)
        self.assertIn("Turn 1: Mistake", cleaned)
        self.assertIn("Ground Truth: Fact", cleaned)

    @patch("requests.post")
    def test_telegram_notifier_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        notifier = Notifier(self.sample_config)
        success = notifier.send("Test Title", "Test Message Body")
        self.assertTrue(success)

    @patch("requests.post")
    def test_ntfy_notifier_success(self, mock_post):
        config = dict(self.sample_config)
        config["push_channel"] = "ntfy"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notifier = Notifier(config)
        success = notifier.send("Test Title", "Test Message Body")
        self.assertTrue(success)

    def test_question_generator_instantiation(self):
        gen = QuestionGenerator(self.sample_config)
        self.assertEqual(gen.primary_model, "gemini-3.6-flash")

    def test_telegram_bot_state_flow(self):
        bot = TelegramBotService(self.sample_config)
        self.assertEqual(bot.state, "IDLE")
        bot._send_message = MagicMock(return_value=True)

        bot.start_topic("测试主题：明朝历史")
        self.assertEqual(bot.state, "WAITING_DURATION")
        self.assertEqual(bot.topic, "测试主题：明朝历史")
        bot._send_message.assert_called_once()

    def test_factuality_report_delivery_and_edit_flow(self):
        bot = TelegramBotService(self.sample_config)
        bot._send_message = MagicMock(return_value=True)

        # 1. Deliver factuality report
        success = bot.deliver_factuality_report("Factuality Report", "Verdict: Model A is better. Model A: Flawless. Model B: Flawless.")
        self.assertTrue(success)
        self.assertEqual(bot.current_factuality_report, "Verdict: Model A is better. Model A: Flawless. Model B: Flawless.")
        # Check that inline keyboard has direct typing and edit options
        last_call_kwargs = bot._send_message.call_args[1]
        buttons = last_call_kwargs["reply_markup"]["inline_keyboard"][0]
        callback_datas = [b["callback_data"] for b in buttons]
        self.assertIn("action_ready_direct_type", callback_datas)
        self.assertIn("action_need_edit_report", callback_datas)

        # 2. User clicks "我需要修改"
        bot._handle_callback_data("action_need_edit_report", message_id=123)
        self.assertEqual(bot.state, "WAITING_REPORT_EDIT")

        # 3. User replies with revised text in Telegram
        bot._handle_update({
            "message": {
                "chat": {"id": 987654321},
                "text": "修改后的报告内容：Model A 逻辑严谨，Model B 有一处瑕疵。"
            }
        })
        self.assertEqual(bot.state, "IDLE")
        self.assertEqual(bot.pending_typing_text, "修改后的报告内容：Model A 逻辑严谨，Model B 有一处瑕疵。")
        # Check confirmation message asked "准备好了吗？"
        last_msg = bot._send_message.call_args[0][0]
        self.assertIn("准备好了吗", last_msg)

    def test_question_card_has_no_typing_button(self):
        bot = TelegramBotService(self.sample_config)
        bot._send_message = MagicMock(return_value=True)
        bot.total_rounds = 3
        bot._deliver_question_card(1, "【提问内容】：测试问题\n【测试关注点】：无")

        last_call_kwargs = bot._send_message.call_args[1]
        all_buttons = [b["callback_data"] for row in last_call_kwargs["reply_markup"]["inline_keyboard"] for b in row]
        self.assertNotIn("action_type_current_q", all_buttons)
        self.assertIn("action_change_q", all_buttons)
        self.assertIn("action_next_round", all_buttons)

    def test_app_import_and_initialization(self):
        import app
        self.assertTrue(hasattr(app, "FactualityApp"))
        self.assertTrue(hasattr(app, "main"))


if __name__ == "__main__":
    unittest.main()
