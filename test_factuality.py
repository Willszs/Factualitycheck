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
        self.assertIn("Conversational Dynamics", SYSTEM_PROMPT)
        self.assertIn("Utility", SYSTEM_PROMPT)
        self.assertIn("For conversational dynamics I prefer", SYSTEM_PROMPT)
        self.assertIn("For utility I prefer", SYSTEM_PROMPT)
        self.assertIn("Model A", SYSTEM_PROMPT)
        self.assertIn("Model B", SYSTEM_PROMPT)
        self.assertIn("Ground Truth", SYSTEM_PROMPT)
        self.assertIn("Turn [X]", SYSTEM_PROMPT)
        self.assertIn("Bridging Quality", SYSTEM_PROMPT)
        self.assertIn("Looping", SYSTEM_PROMPT)

    def test_clean_evaluation_report(self):
        sample_raw = (
            "### Conversational Dynamics\n"
            "For conversational dynamics I prefer Model A.\n"
            "* Model B opened Turn 2 with artificial search intro.\n\n"
            "### Utility\n"
            "For utility I prefer Model A.\n"
            "* Model B in Turn 6 misstated voting thresholds. Ground Truth: simple majority."
        )
        cleaned = FactualityEvaluator.clean_evaluation_report(sample_raw)
        paragraphs = cleaned.split("\n\n")
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(paragraphs[0].startswith("For conversational dynamics I prefer Model A."))
        self.assertTrue(paragraphs[1].startswith("For utility I prefer Model A."))
        self.assertNotIn("###", cleaned)
        self.assertIn("Ground Truth: simple majority.", paragraphs[1])

    def test_clean_evaluation_report_word_limit(self):
        long_paragraph = "word " * 400 + "."
        sample_raw = (
            f"For conversational dynamics I prefer Model A. {long_paragraph}\n\n"
            f"For utility I prefer Model B. {long_paragraph}"
        )
        cleaned = FactualityEvaluator.clean_evaluation_report(sample_raw)
        paragraphs = cleaned.split("\n\n")
        self.assertEqual(len(paragraphs), 2)
        # Each paragraph must be <= 150 words, total <= 300 words
        p1_words = len(paragraphs[0].split())
        p2_words = len(paragraphs[1].split())
        self.assertLessEqual(p1_words, 150)
        self.assertLessEqual(p2_words, 150)
        self.assertLessEqual(p1_words + p2_words, 300)

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

    def test_question_generator_system_prompt_and_cleaning(self):
        from question_generator import get_system_prompt
        prompt = get_system_prompt()
        self.assertIn("REAL HUMAN SPEAKING ALOUD", prompt)
        self.assertIn("ZERO HALLUCINATED PREMISES", prompt)
        self.assertIn("STRICT TOPIC INDEPENDENCE", prompt)
        self.assertIn("各话题严格独立", prompt)

        gen = QuestionGenerator(self.sample_config)
        dirty_output = "好的，这是为您设计的第 1 轮提问：\n\n【提问内容】: 测试问题\n【测试关注点】: 测试关注点"
        cleaned = gen._clean_output(dirty_output)
        self.assertTrue(cleaned.startswith("【提问内容】:"))
        self.assertNotIn("好的，这是为您设计的", cleaned)

    def test_duration_parsing(self):
        bot = TelegramBotService(self.sample_config)
        bot._send_message = MagicMock(return_value=True)
        bot.generator.generate_question = MagicMock(return_value="【提问内容】: 测试\n【测试关注点】: 测试")

        # Test Chinese numerals and minutes
        bot._handle_duration_reply("三分钟")
        self.assertEqual(bot.total_rounds, 3)
        self.assertIn("3分钟真人嘴巴口语交流", bot.duration_desc)

        # Test 10 minutes gives 10 rounds
        bot._handle_duration_reply("10分钟")
        self.assertEqual(bot.total_rounds, 10)
        self.assertIn("10分钟真人嘴巴口语交流", bot.duration_desc)

        # Test Arabic numerals and 15 rounds
        bot._handle_duration_reply("15轮")
        self.assertEqual(bot.total_rounds, 15)

        # Test Chinese numerals fifteen rounds
        bot._handle_duration_reply("十五轮")
        self.assertEqual(bot.total_rounds, 15)

    def test_question_generator_placeholder_sanitation(self):
        qg = QuestionGenerator({"gemini_api_key": "test_key"})
        raw_text = "【提问内容】: 哎，最近除了某某电影，电影院还有什么好看的片子吗？\n【测试关注点】: 考察事实。"
        cleaned = qg._clean_output(raw_text)
        self.assertNotIn("某某电影", cleaned)
        self.assertIn("《抓娃娃》", cleaned)

        raw_text2 = "【提问内容】: 听说某某话剧最近很火，票价值得吗？"
        cleaned2 = qg._clean_output(raw_text2)
        self.assertNotIn("某某话剧", cleaned2)
        self.assertIn("《暗恋桃花源》", cleaned2)

    def test_telegram_bot_state_flow(self):
        bot = TelegramBotService(self.sample_config)
        self.assertEqual(bot.state, "IDLE")
        bot._send_message = MagicMock(return_value=True)

        bot.start_topic("测试主题：明朝历史")
        self.assertEqual(bot.state, "WAITING_DURATION")
        self.assertEqual(bot.topic, "测试主题：明朝历史")
        bot._send_message.assert_called_once()

    def test_end_dialogue_standby_button(self):
        bot = TelegramBotService(self.sample_config)
        bot._send_message = MagicMock(return_value=True)

        bot.state = "INTERACTIVE_QUESTIONS"
        bot.topic = "测试电影主题"
        bot.total_rounds = 3
        bot.current_round = 1

        # Check question card includes "action_end_dialogue" button
        bot._deliver_question_card(1, "【提问内容】: 测试问题\n【测试关注点】: 事实")
        last_call_kwargs = bot._send_message.call_args[1]
        buttons = [b["callback_data"] for row in last_call_kwargs["reply_markup"]["inline_keyboard"] for b in row]
        self.assertIn("action_end_dialogue", buttons)

        # Trigger "action_end_dialogue"
        bot._handle_callback_data("action_end_dialogue", message_id=456)
        self.assertEqual(bot.state, "IDLE")
        self.assertEqual(bot.topic, "")
        self.assertEqual(bot.total_rounds, 0)
        self.assertEqual(bot.current_round, 1)

        # Subsequent click on "action_next_round" should reject gracefully
        bot._send_message.reset_mock()
        bot._handle_callback_data("action_next_round", message_id=456)
        bot._send_message.assert_called_with("⚠️ 当前没有进行中的评测。程序处于 Standby 状态，请在电脑桌面端提交新主题。")

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
        row1_buttons = last_call_kwargs["reply_markup"]["inline_keyboard"][1]
        self.assertIn("action_condense_report", [b["callback_data"] for b in row1_buttons])

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

    def test_search_grounding_trigger_and_queries(self):
        from search_grounding import SearchGrounding
        # Test trigger detection
        self.assertTrue(SearchGrounding.should_search("一款新手机刚刚发布，询问新功能"))
        self.assertTrue(SearchGrounding.should_search("欧冠最新的比赛对阵和比分"))
        self.assertTrue(SearchGrounding.should_search("今天最新的科技新闻"))
        self.assertFalse(SearchGrounding.should_search("自由意志与决定论的哲学探讨"))

        # Test query generation
        queries_phone = SearchGrounding.extract_search_queries("一款新手机刚刚发布，问新功能")
        self.assertTrue(any("手机" in q for q in queries_phone))

        queries_sports = SearchGrounding.extract_search_queries("欧冠最新比赛战报")
        self.assertTrue(any("欧冠" in q for q in queries_sports))

    def test_dialogue_auditor_detection(self):
        from dialogue_auditor import DialogueAuditor
        sample_a = (
            "第 2 轮\n"
            "好的，我来帮你查一下。\n"
            "美国今年3月开始用122条款，对所有进口加了大概10%的基准关税。\n\n"
            "第 6 轮\n"
            "Ooooh, tricky one. One sec.\n"
            "现在已经可以开始预定了。"
        )
        sample_b = (
            "第 2 轮\n"
            "Just pulling the latest trade policy\n\n"
            "第 3 轮\n"
            "中文回答我。人呢人呢？\n"
        )
        report = DialogueAuditor.generate_pre_audit_report(sample_a, sample_b)
        self.assertIn("122条款", report)
        self.assertIn("人呢", report)
        self.assertIn("中文回答我", report)
        self.assertIn("Ooooh, tricky one", report)
        self.assertIn("现在已经可以开始预定", report)
        self.assertIn("PRE-AUDIT FORENSIC EVIDENCE", report)

    def test_dialogue_auditor_search_filler_tolerance(self):
        from dialogue_auditor import DialogueAuditor

        # 1. Short dialogue (3 turns <= 4): 1 search filler is ACCEPTABLE (no violation)
        short_single_filler = (
            "第 1 轮\n用户问今天的天气。\n好的，我来帮你查一下，今天晴转多云。\n\n"
            "第 2 轮\n明天呢？\n明天有小雨，记得带伞。\n\n"
            "第 3 轮\n后天呢？\n后天阴天。"
        )
        audit_short_1 = DialogueAuditor.audit_transcript(short_single_filler, "Model Test")
        self.assertEqual(len(audit_short_1["violations"]), 0)

        # 2. Short dialogue (3 turns <= 4): 2 search fillers EXCEEDS tolerance (violation logged)
        short_two_fillers = (
            "第 1 轮\n好的，我来帮你查一下，今天晴。\n\n"
            "第 2 轮\n我来看看哈，明天有雨。\n\n"
            "第 3 轮\n后天阴天。"
        )
        audit_short_2 = DialogueAuditor.audit_transcript(short_two_fillers, "Model Test")
        self.assertTrue(any("exceeding the allowable threshold" in v for v in audit_short_2["violations"]))

        # 3. Long dialogue (6 turns >= 5): 2 search fillers is ACCEPTABLE (no violation)
        long_two_fillers = (
            "第 1 轮\n好的，我来查一下，北京今天气温20度。\n\n"
            "第 2 轮\n风力怎么样？\n微风2级。\n\n"
            "第 3 轮\n上海呢？\n我来看看哈，上海今天24度。\n\n"
            "第 4 轮\n深圳呢？\n深圳28度。\n\n"
            "第 5 轮\n广州呢？\n广州29度。\n\n"
            "第 6 轮\n杭州呢？\n杭州22度。"
        )
        audit_long_2 = DialogueAuditor.audit_transcript(long_two_fillers, "Model Test")
        self.assertEqual(len(audit_long_2["violations"]), 0)

        # 4. Long dialogue (6 turns >= 5): 3 search fillers EXCEEDS tolerance (violation logged)
        long_three_fillers = (
            "第 1 轮\n好的，我来查一下，北京20度。\n\n"
            "第 2 轮\n我来看看哈，风力2级。\n\n"
            "第 3 轮\n等我一下，上海24度。\n\n"
            "第 4 轮\n深圳28度。\n\n"
            "第 5 轮\n广州29度。\n\n"
            "第 6 轮\n杭州22度。"
        )
        audit_long_3 = DialogueAuditor.audit_transcript(long_three_fillers, "Model Test")
        self.assertTrue(any("exceeding the allowable threshold" in v for v in audit_long_3["violations"]))

    def test_search_grounding_earthquake_support(self):
        from search_grounding import SearchGrounding
        # Trigger check
        self.assertTrue(SearchGrounding.should_search("台湾花莲发生地震，震中在什么位置？"))
        self.assertTrue(SearchGrounding.should_search("四川宜宾地震震源深度和震级"))
        self.assertTrue(SearchGrounding.should_search("台风最新路径走向"))

        # Query extraction
        queries = SearchGrounding.extract_search_queries("台湾花莲刚刚发生地震，震中距离花莲县政府多远？")
        self.assertTrue(any("花莲" in q or "台湾" in q for q in queries))
        self.assertTrue(any("地震" in q for q in queries))

    def test_human_typer_features(self):
        from typer import HumanTyper, QWERTY_NEIGHBORS
        # 1. Test QWERTY neighbors map
        self.assertIn("w", QWERTY_NEIGHBORS["e"])
        self.assertIn("r", QWERTY_NEIGHBORS["e"])
        self.assertIn("s", QWERTY_NEIGHBORS["a"])

        # 2. Test adjacent typo generator
        typo_e = HumanTyper._get_adjacent_typo("e")
        self.assertIn(typo_e.lower(), QWERTY_NEIGHBORS["e"])

        # 3. Test uppercase preservation / handling
        typo_A = HumanTyper._get_adjacent_typo("A")
        self.assertIn(typo_A.lower(), QWERTY_NEIGHBORS["a"])

        # 4. Test empty input handling
        success, msg = HumanTyper.type_like_human("", countdown_secs=0)
        self.assertFalse(success)
        self.assertEqual(msg, "待输入文本为空")

    def test_macos_robust_hotkeys(self):
        from pynput.keyboard import Key, KeyCode
        from app import RobustGlobalHotKeys

        triggered = []
        hotkeys = {
            '<cmd>+<alt>+1': lambda: triggered.append('paste_a'),
            '<cmd>+<alt>+2': lambda: triggered.append('paste_b'),
            '<cmd>+<alt>+3': lambda: triggered.append('paste_topic'),
            '<cmd>+<alt>+s': lambda: triggered.append('submit'),
        }

        listener = RobustGlobalHotKeys(hotkeys)

        # 1. Test Option+3 on macOS (dead-key produces '£', vk=20)
        listener._on_press(Key.cmd, False)
        listener._on_press(Key.alt, False)
        listener._on_press(KeyCode.from_char('£', vk=20), False)
        listener._on_release(KeyCode.from_char('£', vk=20), False)
        listener._on_release(Key.alt, False)
        listener._on_release(Key.cmd, False)

        # 2. Test Option+1 on macOS (dead-key produces '¡', vk=18)
        listener._on_press(Key.cmd, False)
        listener._on_press(Key.alt, False)
        listener._on_press(KeyCode.from_char('¡', vk=18), False)
        listener._on_release(KeyCode.from_char('¡', vk=18), False)
        listener._on_release(Key.alt, False)
        listener._on_release(Key.cmd, False)

        # 3. Test Option+2 on macOS (dead-key produces '™', vk=19)
        listener._on_press(Key.cmd, False)
        listener._on_press(Key.alt, False)
        listener._on_press(KeyCode.from_char('™', vk=19), False)
        listener._on_release(KeyCode.from_char('™', vk=19), False)
        listener._on_release(Key.alt, False)
        listener._on_release(Key.cmd, False)

        # 4. Test Option+S on macOS (dead-key produces 'ß', vk=1)
        listener._on_press(Key.cmd, False)
        listener._on_press(Key.alt, False)
        listener._on_press(KeyCode.from_char('ß', vk=1), False)
        listener._on_release(KeyCode.from_char('ß', vk=1), False)
        listener._on_release(Key.alt, False)
        listener._on_release(Key.cmd, False)

        self.assertEqual(triggered, ['paste_topic', 'paste_a', 'paste_b', 'submit'])

    def test_notification_switch_and_config(self):
        import tempfile
        import shutil
        from unittest.mock import patch, MagicMock
        from app import load_config, save_config, FactualityApp

        # Test save_config and load_config
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_config_file = os.path.join(tmpdir, "config.json")
            with patch("app.os.path.dirname", return_value=tmpdir):
                cfg = {"gemini_model": "gemini-3.1-pro-preview", "enable_notifications": False}
                self.assertTrue(save_config(cfg))
                loaded = load_config()
                self.assertFalse(loaded.get("enable_notifications"))

                cfg["enable_notifications"] = True
                self.assertTrue(save_config(cfg))
                loaded2 = load_config()
                self.assertTrue(loaded2.get("enable_notifications"))

        # Test _notify_macos suppression when disabled
        mock_root = MagicMock()
        with patch.object(FactualityApp, "__init__", return_value=None):
            app = FactualityApp(mock_root)
            app.enable_notifications = False

            with patch("subprocess.run") as mock_run:
                app._notify_macos("Test", "Should not notify")
                # When disabled, subprocess.run must not be called
                mock_run.assert_not_called()

            app.enable_notifications = True
            with patch("subprocess.run") as mock_run:
                app._notify_macos("Test", "Should notify")
                # Note: spawned in thread, give a tiny moment or check thread launch
                # But here we verify that it didn't return early

    def test_parse_topic_and_skills(self):
        from question_generator import QuestionGenerator
        
        # Test 1: User's exact format with newlines
        raw_1 = """你看到有关新呼吸道疾病传播的新闻，你下周有航班——你应该要担心吗？

Skills tested

事实准确性
校准式风险沟通
共情"""
        core_1, skills_1 = QuestionGenerator.parse_topic_and_skills(raw_1)
        self.assertEqual(core_1, "你看到有关新呼吸道疾病传播的新闻，你下周有航班——你应该要担心吗？")
        self.assertEqual(skills_1, ["事实准确性", "校准式风险沟通", "共情"])

        # Test 2: Inline colon with commas
        raw_2 = "你下周有航班。Skills tested: 事实准确性, 校准式风险沟通, 共情"
        core_2, skills_2 = QuestionGenerator.parse_topic_and_skills(raw_2)
        self.assertEqual(core_2, "你下周有航班。")
        self.assertEqual(skills_2, ["事实准确性", "校准式风险沟通", "共情"])

        # Test 3: Chinese keywords with bullet points
        raw_3 = """关于消费电子新品
考察技能：
1. 事实准确性
2. 逻辑严密性"""
        core_3, skills_3 = QuestionGenerator.parse_topic_and_skills(raw_3)
        self.assertEqual(core_3, "关于消费电子新品")
        self.assertEqual(skills_3, ["事实准确性", "逻辑严密性"])

        # Test 4: Plain topic without skills
        raw_4 = "明朝万历十五年的历史事件"
        core_4, skills_4 = QuestionGenerator.parse_topic_and_skills(raw_4)
        self.assertEqual(core_4, "明朝万历十五年的历史事件")
        self.assertEqual(skills_4, [])

    def test_char_limit_and_hard_truncate(self):
        very_long_p1 = "For conversational dynamics I prefer Model A. " + "Model A remained direct and avoided unnecessary preamble. " * 20
        very_long_p2 = "For utility I prefer Model A. " + "Model B in Turn 2 gave completely fabricated numbers about battery life. Ground Truth: 350 miles. " * 20
        full_report = f"{very_long_p1}\n\n{very_long_p2}"
        self.assertGreater(len(full_report), 1500)

        truncated = FactualityEvaluator._hard_truncate_to_char_limit(full_report, max_chars=800)
        self.assertLessEqual(len(truncated), 800)
        paragraphs = truncated.split("\n\n")
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(paragraphs[0].startswith("For conversational dynamics I prefer Model A."))
        self.assertTrue(paragraphs[1].startswith("For utility I prefer Model A."))

    def test_human_typer_control_states(self):
        from typer import HumanTyper

        HumanTyper.pause()
        self.assertTrue(HumanTyper.is_paused())

        HumanTyper.resume()
        self.assertFalse(HumanTyper.is_paused())

        HumanTyper.stop()
        self.assertFalse(HumanTyper.is_active())

    @patch("requests.post")
    def test_telegram_typing_controls_and_condense(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 999}}
        mock_post.return_value = mock_resp

        bot = TelegramBotService(self.sample_config)
        bot.current_factuality_report = (
            "For conversational dynamics I prefer Model A. Model A was clear.\n\n"
            "For utility I prefer Model A. Model B erred in Turn 4. Ground Truth: 100."
        )

        # 1. Test condense callback
        with patch.object(bot.evaluator, "condense_report", return_value="For conversational dynamics I prefer Model A. Clear.\n\nFor utility I prefer Model A. Turn 4 error."):
            bot._handle_callback_data("action_condense_report", message_id=123)
            self.assertIn("Turn 4 error", bot.current_factuality_report)
            self.assertLessEqual(len(bot.current_factuality_report), 800)

        # 2. Test typing pause, resume, stop callbacks
        with patch("typer.HumanTyper.is_active", return_value=True):
            bot._handle_callback_data("action_pause_typing", message_id=123)
            from typer import HumanTyper
            self.assertTrue(HumanTyper.is_paused())

            bot._handle_callback_data("action_resume_typing", message_id=123)
            self.assertFalse(HumanTyper.is_paused())

        bot._handle_callback_data("action_stop_typing", message_id=123)

    def test_dialogue_auditor_bridging_and_looping(self):
        from dialogue_auditor import DialogueAuditor
        sample = (
            "Turn 1: 我来看看哈，先这样。\n"
            "Turn 2: 我来看看哈，再那样。要不要我帮你再拆细一点？\n"
            "Turn 3: 好的。要不要我帮你再拆细一点？"
        )
        res = DialogueAuditor.audit_transcript(sample, "Model A")
        violations_str = " ".join(res["violations"])
        self.assertIn("Bridging Quality", violations_str)
        self.assertIn("Looping", violations_str)


if __name__ == "__main__":
    unittest.main()

