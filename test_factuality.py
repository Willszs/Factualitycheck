"""
Verification script for Factuality Comparison Tool components.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from evaluator import FactualityEvaluator, SYSTEM_PROMPT
from notifier import Notifier


class TestFactualityComponents(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            "gemini_api_key": "dummy_key",
            "gemini_model": "gemini-2.5-flash",
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
        self.assertIn("### The Verdict", SYSTEM_PROMPT)
        self.assertIn("### Model A's Flaws", SYSTEM_PROMPT)
        self.assertIn("### Model B's Flaws", SYSTEM_PROMPT)
        self.assertIn("Ground Truth", SYSTEM_PROMPT)
        self.assertIn("Round [X]", SYSTEM_PROMPT)
        self.assertIn("Flawless", SYSTEM_PROMPT)

    @patch("urllib.request.urlopen")
    def test_telegram_notifier_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true, "result": {}}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        notifier = Notifier(self.sample_config)
        success = notifier.send("Test Title", "Test Message Body")
        self.assertTrue(success)

    @patch("urllib.request.urlopen")
    def test_ntfy_notifier_success(self, mock_urlopen):
        config = dict(self.sample_config)
        config["push_channel"] = "ntfy"
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        notifier = Notifier(config)
        success = notifier.send("Test Title", "Test Message Body")
        self.assertTrue(success)

    def test_app_import_and_initialization(self):
        import app
        self.assertTrue(hasattr(app, "FactualityApp"))
        self.assertTrue(hasattr(app, "main"))


if __name__ == "__main__":
    unittest.main()
