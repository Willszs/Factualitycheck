"""
Human Typing Simulator for macOS.
Simulates natural, realistic human typing cadence with punctuation pauses
and seamless Unicode/Chinese/Emoji support across any active application.
Uses native CoreGraphics for high-speed, zero-process Unicode event posting.
"""

import time
import random
import logging
import subprocess
import ctypes
import ctypes.util
from typing import Tuple

logger = logging.getLogger("factuality.typer")


class HumanTyper:
    _cg = None
    _cf = None
    _app_services = None
    _init_done = False

    @classmethod
    def _init_native(cls):
        if cls._init_done:
            return
        try:
            cls._cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cls._cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            cls._app_services = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")

            cls._cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
            cls._cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p

            # UniChar is a 16-bit unsigned integer (UTF-16 code unit), NOT a 32-bit wchar_t
            cls._cg.CGEventKeyboardSetUnicodeString.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint16),
            ]
            cls._cg.CGEventKeyboardSetUnicodeString.restype = None

            cls._cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
            cls._cg.CGEventPost.restype = None

            cls._cf.CFRelease.argtypes = [ctypes.c_void_p]
        except Exception as e:
            logger.warning(f"Failed to load native CoreGraphics libraries: {e}")
        cls._init_done = True

    @classmethod
    def is_accessibility_trusted(cls) -> bool:
        cls._init_native()
        try:
            if cls._app_services and hasattr(cls._app_services, "AXIsProcessTrusted"):
                return bool(cls._app_services.AXIsProcessTrusted())
        except Exception:
            pass
        return True

    @staticmethod
    def get_clipboard() -> str:
        try:
            return subprocess.check_output(["pbpaste"]).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    @staticmethod
    def set_clipboard(text: str):
        try:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(text.encode("utf-8"))
        except Exception as e:
            logger.warning(f"pbcopy error: {e}")

    @classmethod
    def _post_unicode_char(cls, char: str) -> bool:
        """
        Sends a single character to the active window using native CoreGraphics.
        Uses 16-bit UTF-16 UniChar code units and posts ONLY to kCGHIDEventTap (0) once.
        Unicode string is set strictly on KeyDown to prevent duplicate character insertions.
        """
        cls._init_native()
        if not cls._cg or not cls._cf:
            return False

        try:
            # Special handling for newline / Return key (macOS virtual keycode 36)
            if char == "\n" or char == "\r":
                evt_down = cls._cg.CGEventCreateKeyboardEvent(None, 36, True)
                cls._cg.CGEventPost(0, evt_down)
                cls._cf.CFRelease(evt_down)

                evt_up = cls._cg.CGEventCreateKeyboardEvent(None, 36, False)
                cls._cg.CGEventPost(0, evt_up)
                cls._cf.CFRelease(evt_up)
                return True

            # Convert to UTF-16 code units (UniChar = uint16)
            utf16_bytes = char.encode("utf-16le")
            unichar_count = len(utf16_bytes) // 2
            UniCharArray = ctypes.c_uint16 * unichar_count
            unichars = UniCharArray.from_buffer_copy(utf16_bytes)

            # 1. KeyDown WITH Unicode string
            evt_down = cls._cg.CGEventCreateKeyboardEvent(None, 0, True)
            cls._cg.CGEventKeyboardSetUnicodeString(evt_down, unichar_count, unichars)
            cls._cg.CGEventPost(0, evt_down)  # kCGHIDEventTap = 0 ONLY (never loop taps)
            cls._cf.CFRelease(evt_down)

            # 2. KeyUp WITHOUT Unicode string (pure key release, prevents 2x typing)
            evt_up = cls._cg.CGEventCreateKeyboardEvent(None, 0, False)
            cls._cg.CGEventPost(0, evt_up)  # kCGHIDEventTap = 0 ONLY
            cls._cf.CFRelease(evt_up)
            return True
        except Exception as e:
            logger.debug(f"Native post error: {e}")
            return False

    @classmethod
    def type_like_human(cls, text: str, countdown_secs: int = 3) -> Tuple[bool, str]:
        """
        Simulates natural human typing into whatever application/window currently has focus.
        Always backs up full text into clipboard so user can also Cmd+V.
        """
        if not text:
            return False, "待输入文本为空"

        # Pre-load full text into clipboard as convenience backup
        cls.set_clipboard(text)

        # Check accessibility permission
        if not cls.is_accessibility_trusted():
            err_msg = (
                "<b>需要开启 Mac【辅助功能】权限：</b>\n"
                "macOS 拦截了自动按键模拟。请在 Mac 打开：\n"
                "<b>系统设置 -> 隐私与安全性 -> 辅助功能</b>\n"
                "将 <b>终端（Terminal）</b> 或 <b>Python</b> 勾选允许。\n\n"
                "<i>📋 已自动为您将内容复制到电脑剪贴板，您可以在目标输入框直接按 <b>Cmd+V</b> 粘贴！</i>"
            )
            logger.warning("Accessibility permission is not trusted.")
            return False, err_msg

        logger.info(f"Starting native human typing simulation in {countdown_secs}s for {len(text)} characters...")

        if countdown_secs > 0:
            time.sleep(countdown_secs)

        punctuation_set = set("，。！？；：、,.!?;:\n\r\t")

        try:
            for char in text:
                success = cls._post_unicode_char(char)
                if not success:
                    # Fallback to AppleScript keystroke if native fails
                    subprocess.run(
                        ["osascript", "-e", f'tell application "System Events" to keystroke "{char}"'],
                        capture_output=True,
                        timeout=5,
                    )

                # 1. Base typing cadence: 2x slower (放慢一倍)
                if char in "\n\r":
                    time.sleep(random.uniform(0.90, 1.80))
                elif char in punctuation_set:
                    time.sleep(random.uniform(0.50, 0.95))
                elif char in " \t":
                    time.sleep(random.uniform(0.18, 0.36))
                else:
                    time.sleep(random.uniform(0.12, 0.22))

                # 2. Irregular human pauses & hesitations (没有规律的自然停顿)
                rand_val = random.random()
                if rand_val < 0.008:
                    # Deep reading / proofreading hesitation (0.8% chance)
                    time.sleep(random.uniform(2.0, 3.5))
                elif rand_val < 0.033:
                    # Mid-sentence thought reorganization (2.5% chance)
                    time.sleep(random.uniform(0.90, 1.80))
                elif rand_val < 0.095:
                    # Word-boundary / cognitive thinking pause (6.2% chance)
                    time.sleep(random.uniform(0.40, 0.85))
                elif char.isupper() or char in '()"“”':
                    # Occasional reach for Shift / quote key
                    if random.random() < 0.25:
                        time.sleep(random.uniform(0.12, 0.25))

            logger.info("Human typing simulation finished successfully.")
            return True, ""
        except Exception as e:
            logger.error(f"Error during typing simulation: {e}")
            return False, f"打字模拟中断: {e}（已将全文复制到剪贴板，可按 Cmd+V 粘贴）"
