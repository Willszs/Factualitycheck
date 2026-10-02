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
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Optional

from evaluator import FactualityEvaluator
from notifier import Notifier
from telegram_bot import TelegramBotService

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


class RobustGlobalHotKeys(keyboard.GlobalHotKeys if keyboard else object):
    """Subclass of pynput GlobalHotKeys handling macOS Option/Alt dead-key translation and virtual keycodes.
    On macOS, pressing Option + 1/2/3/s produces unicode characters '¡', '™', '£', 'ß'.
    This normalizes those events back to their canonical base keys so hotkeys like <cmd>+<alt>+3 work reliably.
    """
    MACOS_VK_MAP = {
        18: keyboard.KeyCode.from_char('1') if keyboard else None,
        19: keyboard.KeyCode.from_char('2') if keyboard else None,
        20: keyboard.KeyCode.from_char('3') if keyboard else None,
        21: keyboard.KeyCode.from_char('4') if keyboard else None,
        23: keyboard.KeyCode.from_char('5') if keyboard else None,
        22: keyboard.KeyCode.from_char('6') if keyboard else None,
        26: keyboard.KeyCode.from_char('7') if keyboard else None,
        28: keyboard.KeyCode.from_char('8') if keyboard else None,
        25: keyboard.KeyCode.from_char('9') if keyboard else None,
        29: keyboard.KeyCode.from_char('0') if keyboard else None,
        1: keyboard.KeyCode.from_char('s') if keyboard else None,
        36: keyboard.Key.enter if keyboard else None,
    }

    MACOS_CHAR_MAP = {
        '¡': keyboard.KeyCode.from_char('1') if keyboard else None,
        '™': keyboard.KeyCode.from_char('2') if keyboard else None,
        '£': keyboard.KeyCode.from_char('3') if keyboard else None,
        '¢': keyboard.KeyCode.from_char('4') if keyboard else None,
        '∞': keyboard.KeyCode.from_char('5') if keyboard else None,
        '§': keyboard.KeyCode.from_char('6') if keyboard else None,
        '¶': keyboard.KeyCode.from_char('7') if keyboard else None,
        '•': keyboard.KeyCode.from_char('8') if keyboard else None,
        'ª': keyboard.KeyCode.from_char('9') if keyboard else None,
        'º': keyboard.KeyCode.from_char('0') if keyboard else None,
        'ß': keyboard.KeyCode.from_char('s') if keyboard else None,
    }

    def canonical(self, key):
        if hasattr(key, 'vk') and key.vk in self.MACOS_VK_MAP and self.MACOS_VK_MAP[key.vk]:
            return self.MACOS_VK_MAP[key.vk]
        if hasattr(key, 'char') and key.char in self.MACOS_CHAR_MAP and self.MACOS_CHAR_MAP[key.char]:
            return self.MACOS_CHAR_MAP[key.char]
        return super().canonical(key)

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


def save_config(config: dict) -> bool:
    """Persists updated configuration to config.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save config.json: {e}")
        return False


class FactualityApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("发送至手机")
        self.root.geometry("980x700")
        self.root.minsize(820, 560)

        # Load configurations & initialize backend engines
        self.config = load_config()
        self.enable_notifications = bool(self.config.get("enable_notifications", True))
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

        # Setup global hotkeys for hands-free background pasting & sending
        self.hotkey_listener = None
        self._setup_global_hotkeys()

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

        # --- Top Section: 3rd Input Box for Topic (粘贴3) ---
        topic_frame = tk.Frame(main_container, bg=self.bg_color)
        topic_frame.pack(fill=tk.X, pady=(0, 14))

        label_topic = tk.Label(
            topic_frame,
            text="粘贴3  (⌥⌘3)",
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
            text="📋 粘贴并发送 (⌥⌘3)",
            font=("SF Pro Text", 12, "bold"),
            bg=self.green_accent,
            fg="#ffffff",
            activebackground="#2d9249",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="pointinghand",
            command=self._hotkey_paste_topic,
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

        header_a = tk.Frame(frame_a, bg=self.bg_color)
        header_a.pack(fill=tk.X, pady=(0, 6))

        label_a = tk.Label(
            header_a,
            text="粘贴1 (Model A)",
            font=("SF Pro Text", 13, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
        )
        label_a.pack(side=tk.LEFT)

        btn_paste_a = tk.Button(
            header_a,
            text="📋 粘贴到此处 (⌥⌘1)",
            font=("SF Pro Text", 11),
            bg=self.card_bg,
            fg=self.accent_color,
            activebackground=self.border_color,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="pointinghand",
            command=self._hotkey_paste_a,
        )
        btn_paste_a.pack(side=tk.RIGHT)

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

        header_b = tk.Frame(frame_b, bg=self.bg_color)
        header_b.pack(fill=tk.X, pady=(0, 6))

        label_b = tk.Label(
            header_b,
            text="粘贴2 (Model B)",
            font=("SF Pro Text", 13, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
        )
        label_b.pack(side=tk.LEFT)

        btn_paste_b = tk.Button(
            header_b,
            text="📋 粘贴到此处 (⌥⌘2)",
            font=("SF Pro Text", 11),
            bg=self.card_bg,
            fg=self.accent_color,
            activebackground=self.border_color,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="pointinghand",
            command=self._hotkey_paste_b,
        )
        btn_paste_b.pack(side=tk.RIGHT)

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

        # --- Bottom Section: Auto-Status Indicator & Manual Fallback Button ---
        bottom_bar = tk.Frame(main_container, bg=self.bg_color)
        bottom_bar.pack(fill=tk.X, pady=(16, 0))

        # Status indicator
        self.lbl_status = tk.Label(
            bottom_bar,
            text="⚡ 自动流程已就绪：粘贴问题即发，两模型填满即发",
            font=("SF Pro Text", 12, "bold"),
            bg=self.bg_color,
            fg="#188038",
        )
        self.lbl_status.pack(side=tk.LEFT)

        # Standard send button as backup
        self.btn_submit = tk.Button(
            bottom_bar,
            text="📤 手动重发 (⌥⌘↩)",
            font=("SF Pro Text", 11),
            bg=self.card_bg,
            fg=self.accent_color,
            activebackground=self.border_color,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="pointinghand",
            command=self.on_submit_eval,
        )
        self.btn_submit.pack(side=tk.RIGHT)

        # Toggle switch for desktop popup notifications
        self.var_notify = tk.BooleanVar(value=self.enable_notifications)
        self.chk_notify = tk.Checkbutton(
            bottom_bar,
            text="🔔 桌面弹窗提醒",
            variable=self.var_notify,
            command=self._on_toggle_notifications,
            font=("SF Pro Text", 11),
            bg=self.bg_color,
            fg=self.text_color,
            activebackground=self.bg_color,
            selectcolor=self.card_bg,
            cursor="pointinghand",
            padx=6,
        )
        self.chk_notify.pack(side=tk.RIGHT, padx=(0, 16))

        # Keyboard shortcuts (dual-layer: both Cmd+Return and Option+Command bindings)
        self.root.bind("<Command-Return>", lambda event: self.on_submit_eval())
        self.root.bind("<Control-Return>", lambda event: self.on_submit_eval())
        self.root.bind("<Option-Command-Return>", lambda event: self.on_submit_eval())
        self.root.bind("<Alt-Command-Return>", lambda event: self.on_submit_eval())
        self.root.bind("<Option-Command-1>", lambda event: self._hotkey_paste_a())
        self.root.bind("<Alt-Command-1>", lambda event: self._hotkey_paste_a())
        self.root.bind("<Option-Command-2>", lambda event: self._hotkey_paste_b())
        self.root.bind("<Alt-Command-2>", lambda event: self._hotkey_paste_b())
        self.root.bind("<Option-Command-3>", lambda event: self._hotkey_paste_topic())
        self.root.bind("<Alt-Command-3>", lambda event: self._hotkey_paste_topic())
        self.root.bind("<Option-Command-s>", lambda event: self._hotkey_paste_b_and_submit())
        self.root.bind("<Option-Command-S>", lambda event: self._hotkey_paste_b_and_submit())
        self.root.bind("<Alt-Command-s>", lambda event: self._hotkey_paste_b_and_submit())
        self.root.bind("<Alt-Command-S>", lambda event: self._hotkey_paste_b_and_submit())

        # Set initial focus to Topic input
        self.txt_topic.focus_set()

        # Enhance inputs with native shortcuts & context menus
        self._setup_text_enhancements()

    def _setup_text_enhancements(self):
        """Adds macOS Cmd+A select all and native right-click context menu."""
        widgets = [self.txt_topic, self.txt_a, self.txt_b]
        for w in widgets:
            self._attach_context_menu(w)

        # macOS Cmd+A and Ctrl+A binding
        self.txt_topic.bind("<Command-a>", lambda e: (self.txt_topic.select_range(0, tk.END), self.txt_topic.icursor(tk.END), "break")[2])
        self.txt_topic.bind("<Control-a>", lambda e: (self.txt_topic.select_range(0, tk.END), self.txt_topic.icursor(tk.END), "break")[2])
        self.txt_a.bind("<Command-a>", lambda e: (self.txt_a.tag_add("sel", "1.0", "end"), "break")[1])
        self.txt_a.bind("<Control-a>", lambda e: (self.txt_a.tag_add("sel", "1.0", "end"), "break")[1])
        self.txt_b.bind("<Command-a>", lambda e: (self.txt_b.tag_add("sel", "1.0", "end"), "break")[1])
        self.txt_b.bind("<Control-a>", lambda e: (self.txt_b.tag_add("sel", "1.0", "end"), "break")[1])

        # Auto-send on paste events inside input widgets (Zero-click workflow)
        self.txt_topic.bind("<<Paste>>", lambda e: self.root.after(100, self._check_and_auto_send_topic))
        self.txt_a.bind("<<Paste>>", lambda e: self.root.after(100, self._check_and_auto_send_eval))
        self.txt_b.bind("<<Paste>>", lambda e: self.root.after(100, self._check_and_auto_send_eval))

    def _check_and_auto_send_topic(self):
        clip = self._get_clipboard_text().strip()
        # If clipboard contains multi-line content (e.g. Skills tested), preserve it over Entry truncation
        topic = clip if (clip and ("\n" in clip or "skills" in clip.lower() or "技能" in clip)) else self.txt_topic.get().strip()
        if topic:
            self._set_status_temp("已存入主题，自动发送中...", duration_ms=2500, fg="#1e8e3e")
            self._play_feedback_sound("Glass")
            self._notify_macos("Factuality 自动发送", "检测到已粘贴主题，已自动发送至 Telegram 机器人！")
            self.on_submit_topic(custom_topic=topic)

    def _check_and_auto_send_eval(self):
        content_a = self.txt_a.get("1.0", tk.END).strip()
        content_b = self.txt_b.get("1.0", tk.END).strip()
        if content_a and content_b:
            self._set_status_temp("两边模型内容齐全，正在自动测评发送...", duration_ms=2500, fg="#1e8e3e")
            self._play_feedback_sound("Hero")
            self._notify_macos("Factuality 自动发送", "检测到两模型内容已齐全，正在自动评测并发送至手机！")
            self.on_submit_eval()
        elif content_a:
            self._set_status_temp(f"已存入【粘贴1】({len(content_a)}字)，等待模型二...", duration_ms=2500, fg="#1a73e8")
        elif content_b:
            self._set_status_temp(f"已存入【粘贴2】({len(content_b)}字)，等待模型一...", duration_ms=2500, fg="#1a73e8")

    def _attach_context_menu(self, widget):
        """Attaches right-click Cut, Copy, Paste, Select All, Clear menu."""
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="剪切 (Cut)", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="复制 (Copy)", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="粘贴 (Paste)", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_separator()
        if isinstance(widget, tk.Entry):
            menu.add_command(label="全选 (Select All)", command=lambda: [widget.select_range(0, tk.END), widget.icursor(tk.END)])
            menu.add_command(label="清空 (Clear)", command=lambda: widget.delete(0, tk.END))
        else:
            menu.add_command(label="全选 (Select All)", command=lambda: widget.tag_add("sel", "1.0", "end"))
            menu.add_command(label="清空 (Clear)", command=lambda: widget.delete("1.0", tk.END))

        def show_menu(event):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        widget.bind("<Button-2>", show_menu)
        widget.bind("<Button-3>", show_menu)
        widget.bind("<Control-Button-1>", show_menu)

    def on_submit_topic(self, custom_topic: Optional[str] = None):
        """Immediately captures text in 粘贴3, clears input, and asks duration on Telegram."""
        topic = (custom_topic or self.txt_topic.get()).strip()
        if not topic:
            self._set_status_temp("请先在【粘贴3】输入内容", duration_ms=2500, fg="#d93025")
            return

        # 1. Immediately wipe topic input
        self.txt_topic.delete(0, tk.END)

        # 2. Update status and inform user to check Telegram
        self._set_status_temp("已发送，请留意手机...", duration_ms=3000, fg="#1e8e3e")

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
                logger.info(
                    f"Starting background evaluation for task queued at {task['time']}...\n"
                    f"=== TRANSCRIPT A ({len(task['content_a'])} chars) ===\n{task['content_a']}\n"
                    f"=== TRANSCRIPT B ({len(task['content_b'])} chars) ===\n{task['content_b']}"
                )

                # 1. Call Gemini LLM to evaluate factuality
                summary = self.evaluator.evaluate(task["content_a"], task["content_b"])
                logger.info(f"=== EVALUATION REPORT [{task['time']}] ===\n{summary}")

                # 2. Push result strictly to mobile notification
                title = f"Factuality Assessment Report [{task['time']}]"
                delivered = False
                if self.tg_service and self.tg_service.is_running and self.tg_service.bot_token:
                    delivered = self.tg_service.deliver_factuality_report(title, summary)
                    if delivered:
                        logger.info("Assessment pushed to Telegram bot with interactive typing actions.")

                if not delivered or self.notifier.channel != "telegram":
                    success = self.notifier.send(title, summary)
                    if success:
                        logger.info(f"Assessment pushed successfully via {self.notifier.channel}.")
                    else:
                        logger.warning("Push notification could not be dispatched (check config).")

                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"Error in background evaluation worker: {e}", exc_info=True)

    # =========================================================================
    # Global Hotkeys (Hands-Free Background Pasting & Sending)
    # =========================================================================
    def _setup_global_hotkeys(self):
        """Initializes system-wide global hotkeys for hands-free background pasting & sending."""
        if not keyboard:
            logger.warning("pynput is not installed; global hotkeys disabled.")
            return

        try:
            hotkey_map = {}
            a_keys = [
                "<cmd>+<alt>+1", "<ctrl>+<alt>+1", "<cmd>+<ctrl>+1", "<cmd>+<shift>+1", "<ctrl>+<shift>+1"
            ]
            b_keys = [
                "<cmd>+<alt>+2", "<ctrl>+<alt>+2", "<cmd>+<ctrl>+2", "<cmd>+<shift>+2", "<ctrl>+<shift>+2"
            ]
            topic_keys = [
                "<cmd>+<alt>+3", "<ctrl>+<alt>+3", "<cmd>+<ctrl>+3", "<ctrl>+<shift>+3"
            ]
            submit_keys = [
                "<cmd>+<alt>+<enter>", "<ctrl>+<alt>+<enter>", "<cmd>+<alt>+s", "<ctrl>+<alt>+s"
            ]

            for k in a_keys:
                hotkey_map[k] = self._hotkey_paste_a
            for k in b_keys:
                hotkey_map[k] = self._hotkey_paste_b
            for k in topic_keys:
                hotkey_map[k] = self._hotkey_paste_topic
            for k in submit_keys:
                hotkey_map[k] = self._hotkey_submit_eval

            self.hotkey_listener = RobustGlobalHotKeys(hotkey_map)
            self.hotkey_listener.start()
            logger.info("Robust global hotkeys started: ⌥⌘1 (Model A), ⌥⌘2 (Model B), ⌥⌘3 (Topic), ⌥⌘S/↩ (Send).")
        except Exception as e:
            logger.warning(f"Could not initialize global hotkeys: {e}")
            self.hotkey_listener = None

    def _get_clipboard_text(self) -> str:
        """Reads system clipboard safely on macOS via pbpaste."""
        try:
            res = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0 and res.stdout:
                return res.stdout
        except Exception:
            pass
        return ""

    def _play_feedback_sound(self, sound_name: str = "Tink"):
        """Plays subtle macOS feedback sound in background."""
        def _play():
            try:
                subprocess.run(
                    ["afplay", f"/System/Library/Sounds/{sound_name}.aiff"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()

    def _on_toggle_notifications(self):
        enabled = self.var_notify.get()
        self.enable_notifications = enabled
        self.config["enable_notifications"] = enabled
        save_config(self.config)
        status_text = "🔔 桌面弹窗提醒已开启" if enabled else "🔕 桌面弹窗提醒已关闭（静音免打扰）"
        self._set_status_temp(status_text, duration_ms=2500, fg="#1e8e3e" if enabled else "#5f6368")
        logger.info(f"Desktop popup notifications toggled: {enabled}")

    def _notify_macos(self, title: str, message: str):
        """Dispatches a lightweight macOS system notification if enabled."""
        if not getattr(self, "enable_notifications", True):
            return
        def _notify():
            try:
                clean_title = title.replace('"', '\\"')
                clean_msg = message.replace('"', '\\"').replace("\n", " ")
                script = f'display notification "{clean_msg}" with title "{clean_title}"'
                subprocess.run(["osascript", "-e", script], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        threading.Thread(target=_notify, daemon=True).start()

    def _hotkey_paste_a(self):
        logger.info("Hotkey triggered: Paste A (⌥⌘1)")
        clip = self._get_clipboard_text().strip()
        self.root.after(0, lambda: self._apply_paste_a_main_thread(clip))

    def _apply_paste_a_main_thread(self, clip: str):
        existing = self.txt_a.get("1.0", tk.END).strip()
        target_text = clip or existing
        if not target_text:
            self._set_status_temp("⚠️ 剪贴板和输入框均为空，请先复制内容", duration_ms=2500, fg="#d93025")
            self._notify_macos("Factuality 提示", "剪贴板为空，请先在其他应用复制内容！")
            return

        self.txt_a.delete("1.0", tk.END)
        self.txt_a.insert("1.0", target_text)
        content_a = target_text.strip()
        content_b = self.txt_b.get("1.0", tk.END).strip()

        if content_a and content_b:
            self._set_status_temp("两边模型内容齐全，自动测评发送中...", duration_ms=2500, fg="#1e8e3e")
            self._play_feedback_sound("Hero")
            self._notify_macos("Factuality 自动发送", f"【粘贴1】已存入（{len(content_a)}字），两模型齐全，自动测评发送中！")
            self.on_submit_eval()
        else:
            self._set_status_temp(f"已存入【粘贴1】({len(content_a)}字)，等待模型二...", duration_ms=2500, fg="#1a73e8")
            self._play_feedback_sound("Pop")
            self._notify_macos("Factuality 快捷键", f"已从剪贴板存入【粘贴1】（{len(content_a)} 字），等待模型二...")

    def _hotkey_paste_b(self):
        logger.info("Hotkey triggered: Paste B (⌥⌘2)")
        clip = self._get_clipboard_text().strip()
        self.root.after(0, lambda: self._apply_paste_b_main_thread(clip))

    def _apply_paste_b_main_thread(self, clip: str):
        existing = self.txt_b.get("1.0", tk.END).strip()
        target_text = clip or existing
        if not target_text:
            self._set_status_temp("⚠️ 剪贴板和输入框均为空，请先复制内容", duration_ms=2500, fg="#d93025")
            self._notify_macos("Factuality 提示", "剪贴板为空，请先在其他应用复制内容！")
            return

        self.txt_b.delete("1.0", tk.END)
        self.txt_b.insert("1.0", target_text)
        content_b = target_text.strip()
        content_a = self.txt_a.get("1.0", tk.END).strip()

        if content_a and content_b:
            self._set_status_temp("两边模型内容齐全，自动测评发送中...", duration_ms=2500, fg="#1e8e3e")
            self._play_feedback_sound("Hero")
            self._notify_macos("Factuality 自动发送", f"【粘贴2】已存入（{len(content_b)}字），两模型齐全，自动测评发送中！")
            self.on_submit_eval()
        else:
            self._set_status_temp(f"已存入【粘贴2】({len(content_b)}字)，等待模型一...", duration_ms=2500, fg="#1a73e8")
            self._play_feedback_sound("Pop")
            self._notify_macos("Factuality 快捷键", f"已从剪贴板存入【粘贴2】（{len(content_b)} 字），等待模型一...")

    def _hotkey_paste_topic(self):
        logger.info("Hotkey triggered: Paste Topic (⌥⌘3)")
        clip = self._get_clipboard_text().strip()
        self.root.after(0, lambda: self._apply_paste_topic_main_thread(clip))

    def _apply_paste_topic_main_thread(self, clip: str):
        existing = self.txt_topic.get().strip()
        target_text = clip or existing
        if not target_text:
            self._set_status_temp("⚠️ 剪贴板和输入框均为空，请先复制内容", duration_ms=2500, fg="#d93025")
            self._notify_macos("Factuality 提示", "剪贴板与输入框均为空，请先在其他应用复制内容！")
            return

        clean_topic = target_text.strip()
        self.txt_topic.delete(0, tk.END)
        self.txt_topic.insert(0, clean_topic)
        self._set_status_temp("已存入主题，自动发送中...", duration_ms=2500, fg="#1e8e3e")
        self._play_feedback_sound("Glass")
        self._notify_macos("Factuality 自动发送", f"已自动获取主题并发送至 Telegram：\n{clean_topic[:40]}...")
        self.on_submit_topic(custom_topic=clean_topic)

    def _hotkey_submit_eval(self):
        logger.info("Hotkey triggered: Submit Eval (⌥⌘S / ⌥⌘↩)")
        self.root.after(0, self._apply_submit_eval)

    def _apply_submit_eval(self):
        self.on_submit_eval()
        self._play_feedback_sound("Hero")
        self._notify_macos("Factuality 快捷键", "已触发测评并发送至手机！")

    def on_close(self):
        """Clean shutdown of background threads."""
        logger.info("Application shutting down...")
        if hasattr(self, "hotkey_listener") and self.hotkey_listener:
            try:
                self.hotkey_listener.stop()
            except Exception:
                pass
        self.tg_service.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FactualityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
