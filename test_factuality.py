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

    def test_app_import_and_initialization(self):
        import app
        self.assertTrue(hasattr(app, "FactualityApp"))
        self.assertTrue(hasattr(app, "main"))


if __name__ == "__main__":
    unittest.main()
