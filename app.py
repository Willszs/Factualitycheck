#!/usr/bin/env python3
"""
AI Factuality Comparison Tool - Desktop GUI
A minimalist desktop app to compare multi-turn dialogue factuality between Model A & Model B,
and interactively design multi-turn benchmarking questions via Telegram.
"""

import os
import sys
import json
import queue
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

from evaluator import FactualityEvaluator
from notifier import Notifier
from telegram_bot import TelegramBotService

# Configure file logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factuality.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("factuality.app")


def load_config() -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.json")
    example_path = os.path.join(base_dir, "config.example.json")

    if not os.path.exists(config_path) and os.path.exists(example_path):
        import shutil
        try:
            shutil.copyfile(example_path, config_path)
            logger.info("Created config.json from template config.example.json")
        except Exception as e:
            logger.warning(f"Failed to copy config.example.json: {e}")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config.json: {e}")
    return {}


class FactualityApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("发送至手机")
        self.root.geometry("980x700")
        self.root.minsize(820, 560)

        # Load configurations & initialize backend engines
        self.config = load_config()
        self.evaluator = FactualityEvaluator(self.config)
        self.notifier = Notifier(self.config)

        # Initialize and start background interactive Telegram bot
        self.tg_service = TelegramBotService(self.config)
        self.tg_service.start()

        # Background processing queue & worker for factuality evaluation
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        self._apply_styles()
        self._build_ui()

        # Clean shutdown handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_styles(self):
        style = ttk.Style(self.root)
        if "aqua" in style.theme_names():
            style.theme_use("aqua")
        else:
            style.theme_use("clam")

        self.bg_color = "#f7f8fa"
        self.card_bg = "#ffffff"
        self.text_color = "#202124"
        self.border_color = "#dadce0"
        self.accent_color = "#1a73e8"
        self.green_accent = "#34a853"

        self.root.configure(bg=self.bg_color)

    def _build_ui(self):
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        # --- Top Section: 3rd Input Box for Topic (对话主题) ---
        topic_frame = tk.Frame(main_container, bg=self.bg_color)
        topic_frame.pack(fill=tk.X, pady=(0, 14))

        label_topic = tk.Label(
            topic_frame,
            text="对话主题",
            font=("SF Pro Text", 13, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
        )
        label_topic.pack(side=tk.LEFT, padx=(0, 10))

        self.txt_topic = tk.Entry(
            topic_frame,
            font=("SF Pro Text", 13),
            bg=self.card_bg,
            fg=self.text_color,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.accent_color,
        )
        self.txt_topic.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=6)
        self.txt_topic.bind("<Return>", lambda event: self.on_submit_topic())

        self.btn_topic = tk.Button(
            topic_frame,
            text="发起主题",
            font=("SF Pro Text", 12, "bold"),
            bg=self.green_accent,
            fg="#ffffff",
            activebackground="#2d9249",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=6,
            cursor="pointinghand",
            command=self.on_submit_topic,
        )
        self.btn_topic.pack(side=tk.RIGHT)

        # --- Middle Section: Two Side-by-Side Text Inputs (粘贴1 & 粘贴2) ---
        input_panes = tk.Frame(main_container, bg=self.bg_color)
        input_panes.pack(fill=tk.BOTH, expand=True)

        input_panes.columnconfigure(0, weight=1)
        input_panes.columnconfigure(1, weight=1)
        input_panes.rowconfigure(0, weight=1)

        # --- Box 1: 粘贴1 ---
        frame_a = tk.Frame(input_panes, bg=self.bg_color)
        frame_a.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        label_a = tk.Label(
            frame_a,
            text="粘贴1",
            font=("SF Pro Text", 13, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            anchor="w",
        )
        label_a.pack(fill=tk.X, pady=(0, 6))

        self.txt_a = tk.Text(
            frame_a,
            wrap=tk.WORD,
            font=("Menlo", 12),
            bg=self.card_bg,
            fg=self.text_color,
            insertbackground="#1a73e8",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.accent_color,
            padx=12,
            pady=10,
        )
        self.txt_a.pack(fill=tk.BOTH, expand=True)

        # --- Box 2: 粘贴2 ---
        frame_b = tk.Frame(input_panes, bg=self.bg_color)
        frame_b.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        label_b = tk.Label(
            frame_b,
            text="粘贴2",
            font=("SF Pro Text", 13, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            anchor="w",
        )
        label_b.pack(fill=tk.X, pady=(0, 6))

        self.txt_b = tk.Text(
            frame_b,
            wrap=tk.WORD,
            font=("Menlo", 12),
            bg=self.card_bg,
            fg=self.text_color,
            insertbackground="#1a73e8",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.accent_color,
            padx=12,
            pady=10,
        )
        self.txt_b.pack(fill=tk.BOTH, expand=True)

        # --- Bottom Section: Action Button & Subtle Status ---
        bottom_bar = tk.Frame(main_container, bg=self.bg_color)
        bottom_bar.pack(fill=tk.X, pady=(16, 0))

        self.btn_submit = tk.Button(
            bottom_bar,
            text="发送到手机",
            font=("SF Pro Text", 14, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            activebackground="#1557b0",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=32,
            pady=10,
            cursor="pointinghand",
            command=self.on_submit_eval,
        )
        self.btn_submit.pack(side=tk.LEFT)

        # Status indicator (CRITICAL: zero evaluation result on UI)
        self.lbl_status = tk.Label(
            bottom_bar,
            text="就绪",
            font=("SF Pro Text", 11),
            bg=self.bg_color,
            fg="#5f6368",
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # Keyboard shortcuts (Cmd+Return on macOS, Ctrl+Return on Windows/Linux)
        self.root.bind("<Command-Return>", lambda event: self.on_submit_eval())
        self.root.bind("<Control-Return>", lambda event: self.on_submit_eval())

        # Set initial focus to Topic input
        self.txt_topic.focus_set()

    def on_submit_topic(self):
        """Immediately captures topic, clears input, and asks duration on Telegram."""
        topic = self.txt_topic.get().strip()
        if not topic:
            self._set_status_temp("请先输入对话主题", duration_ms=2500, fg="#d93025")
            return

        # 1. Immediately wipe topic input
        self.txt_topic.delete(0, tk.END)

        # 2. Update status and inform user to check Telegram
        self._set_status_temp("主题已发起！请在手机 Telegram 回复轮数...", duration_ms=4000, fg="#1e8e3e")

        # 3. Trigger topic flow in background thread
        threading.Thread(target=self.tg_service.start_topic, args=(topic,), daemon=True).start()
        logger.info(f"Topic submitted: {topic}")

    def on_submit_eval(self):
        """Immediately captures text, clears UI inputs, and dispatches to background evaluation."""
        content_a = self.txt_a.get("1.0", tk.END).strip()
        content_b = self.txt_b.get("1.0", tk.END).strip()

        if not content_a and not content_b:
            self._set_status_temp("请至少输入内容", duration_ms=2500, fg="#d93025")
            return

        # 1. Immediately wipe both text inputs
        self.txt_a.delete("1.0", tk.END)
        self.txt_b.delete("1.0", tk.END)

        # 2. Reset cursor back to Box 1
        self.txt_a.focus_set()

        # 3. Inform user that task was handed off to background
        self._set_status_temp("已发送，请留意手机...", duration_ms=3000, fg="#1e8e3e")

        # 4. Enqueue task for background evaluation
        task = {
            "content_a": content_a,
            "content_b": content_b,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        self.task_queue.put(task)
        logger.info(f"Evaluation task queued at {task['time']} (A: {len(content_a)} chars, B: {len(content_b)} chars)")

    def _set_status_temp(self, text: str, duration_ms: int = 2500, fg: str = "#5f6368"):
        self.lbl_status.config(text=text, fg=fg)
        self.root.after(duration_ms, lambda: self.lbl_status.config(text="就绪", fg="#5f6368"))

    def _worker_loop(self):
        """Background daemon processing evaluation tasks sequentially without freezing UI."""
        while True:
            try:
                task = self.task_queue.get()
                logger.info(f"Starting background evaluation for task queued at {task['time']}...")

                # 1. Call Gemini LLM to evaluate factuality
                summary = self.evaluator.evaluate(task["content_a"], task["content_b"])

                # 2. Push result strictly to mobile notification
                title = f"Factuality Assessment Report [{task['time']}]"
                success = self.notifier.send(title, summary)

                if success:
                    logger.info("Assessment pushed successfully to mobile device.")
                else:
                    logger.warning("Push notification could not be dispatched (check config).")

                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"Error in background evaluation worker: {e}", exc_info=True)

    def on_close(self):
        """Clean shutdown of background threads."""
        logger.info("Application shutting down...")
        self.tg_service.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FactualityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
