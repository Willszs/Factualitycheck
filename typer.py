"""
Human Typing Simulator for macOS.
Simulates natural, realistic human typing cadence with punctuation pauses,
irregular hesitations, cognitive thinking intervals, and human-like typos
with realistic backspace corrections across any active application.
Uses native CoreGraphics for high-speed, zero-process Unicode and key event posting.
"""

import time
import random
import logging
import subprocess
import threading
import ctypes
import ctypes.util
from typing import Tuple, List, Optional

logger = logging.getLogger("factuality.typer")


# QWERTY adjacent key map for realistic human finger slips
QWERTY_NEIGHBORS = {
    'q': ['w', 'a', '1', '2'],
    'w': ['q', 'e', 's', 'a', '2', '3'],
    'e': ['w', 'r', 'd', 's', '3', '4'],
    'r': ['e', 't', 'f', 'd', '4', '5'],
    't': ['r', 'y', 'g', 'f', '5', '6'],
    'y': ['t', 'u', 'h', 'g', '6', '7'],
    'u': ['y', 'i', 'j', 'h', '7', '8'],
    'i': ['u', 'o', 'k', 'j', '8', '9'],
    'o': ['i', 'p', 'l', 'k', '9', '0'],
    'p': ['o', '[', ';', 'l', '0', '-'],
    'a': ['s', 'q', 'z', 'w'],
    's': ['a', 'd', 'w', 'x', 'z', 'e'],
    'd': ['s', 'f', 'e', 'c', 'x', 'r'],
    'f': ['d', 'g', 'r', 'v', 'c', 't'],
    'g': ['f', 'h', 't', 'b', 'v', 'y'],
    'h': ['g', 'j', 'y', 'n', 'b', 'u'],
    'j': ['h', 'k', 'u', 'm', 'n', 'i'],
    'k': ['j', 'l', 'i', ',', 'm', 'o'],
    'l': ['k', ';', 'o', '.', ',', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 'c', 's', 'd'],
    'c': ['x', 'v', 'd', 'f'],
    'v': ['c', 'b', 'f', 'g'],
    'b': ['v', 'n', 'g', 'h'],
    'n': ['b', 'm', 'h', 'j'],
    'm': ['n', ',', 'j', 'k'],
    '1': ['2', 'q'],
    '2': ['1', '3', 'w', 'q'],
    '3': ['2', '4', 'e', 'w'],
    '4': ['3', '5', 'r', 'e'],
    '5': ['4', '6', 't', 'r'],
    '6': ['5', '7', 'y', 't'],
    '7': ['6', '8', 'u', 'y'],
    '8': ['7', '9', 'i', 'u'],
    '9': ['8', '0', 'o', 'i'],
    '0': ['9', '-', 'p', 'o'],
}


class HumanTyper:
    _cg = None
    _cf = None
    _app_services = None
    _init_done = False

    # Threading control for pause, resume, and abort
    _pause_event = threading.Event()
    _pause_event.set()
    _stop_event = threading.Event()
    _is_running = False
    _is_paused = False
    _current_index = 0
    _total_chars = 0

    @classmethod
    def pause(cls):
        """Pauses the active typing simulation immediately."""
        cls._is_paused = True
        cls._pause_event.clear()
        logger.info("HumanTyper: pause signal triggered.")

    @classmethod
    def resume(cls):
        """Resumes a paused typing simulation."""
        cls._is_paused = False
        cls._pause_event.set()
        logger.info("HumanTyper: resume signal triggered.")

    @classmethod
    def stop(cls):
        """Aborts the active typing simulation immediately."""
        cls._stop_event.set()
        cls._pause_event.set()  # Unblock if currently paused
        cls._is_running = False
        cls._is_paused = False
        logger.info("HumanTyper: stop signal triggered.")

    @classmethod
    def is_paused(cls) -> bool:
        return cls._is_paused

    @classmethod
    def is_active(cls) -> bool:
        return cls._is_running

    @classmethod
    def get_progress(cls) -> Tuple[int, int]:
        return cls._current_index, cls._total_chars

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

            # UniChar is a 16-bit unsigned integer (UTF-16 code unit)
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
    def _post_key_code(cls, keycode: int) -> bool:
        """Posts a raw virtual keycode event (e.g., 51 for Backspace, 36 for Return)."""
        cls._init_native()
        if not cls._cg or not cls._cf:
            return False
        try:
            evt_down = cls._cg.CGEventCreateKeyboardEvent(None, keycode, True)
            cls._cg.CGEventPost(0, evt_down)
            cls._cf.CFRelease(evt_down)

            time.sleep(random.uniform(0.03, 0.07))

            evt_up = cls._cg.CGEventCreateKeyboardEvent(None, keycode, False)
            cls._cg.CGEventPost(0, evt_up)
            cls._cf.CFRelease(evt_up)
            return True
        except Exception as e:
            logger.debug(f"Native keycode post error: {e}")
            return False

    @classmethod
    def _post_backspace(cls, count: int = 1) -> bool:
        """Simulates human pressing Backspace (macOS virtual keycode 51)."""
        for i in range(count):
            success = cls._post_key_code(51)
            if not success:
                try:
                    subprocess.run(
                        ["osascript", "-e", 'tell application "System Events" to key code 51'],
                        capture_output=True,
                        timeout=2,
                    )
                except Exception as e:
                    logger.debug(f"AppleScript backspace error: {e}")
            if count > 1 and i < count - 1:
                # Natural delay between consecutive backspaces (80ms - 150ms)
                time.sleep(random.uniform(0.08, 0.15))
        return True

    @classmethod
    def _post_unicode_char(cls, char: str) -> bool:
        """
        Sends a single character to the active window using native CoreGraphics.
        Uses 16-bit UTF-16 UniChar code units and posts strictly to kCGHIDEventTap (0).
        """
        cls._init_native()
        if not cls._cg or not cls._cf:
            return False

        try:
            # Special handling for newline / Return key (macOS virtual keycode 36)
            if char == "\n" or char == "\r":
                return cls._post_key_code(36)

            # Convert to UTF-16 code units (UniChar = uint16)
            utf16_bytes = char.encode("utf-16le")
            unichar_count = len(utf16_bytes) // 2
            UniCharArray = ctypes.c_uint16 * unichar_count
            unichars = UniCharArray.from_buffer_copy(utf16_bytes)

            # 1. KeyDown WITH Unicode string
            evt_down = cls._cg.CGEventCreateKeyboardEvent(None, 0, True)
            cls._cg.CGEventKeyboardSetUnicodeString(evt_down, unichar_count, unichars)
            cls._cg.CGEventPost(0, evt_down)
            cls._cf.CFRelease(evt_down)

            # 2. KeyUp WITHOUT Unicode string (prevents 2x duplicate typing)
            evt_up = cls._cg.CGEventCreateKeyboardEvent(None, 0, False)
            cls._cg.CGEventPost(0, evt_up)
            cls._cf.CFRelease(evt_up)
            return True
        except Exception as e:
            logger.debug(f"Native post error: {e}")
            return False

    @classmethod
    def _type_character_safe(cls, char: str) -> bool:
        """Types a character with native CoreGraphics and AppleScript fallback."""
        success = cls._post_unicode_char(char)
        if not success:
            try:
                subprocess.run(
                    ["osascript", "-e", f'tell application "System Events" to keystroke "{char}"'],
                    capture_output=True,
                    timeout=5,
                )
                return True
            except Exception:
                return False
        return True

    @classmethod
    def _get_adjacent_typo(cls, char: str) -> Optional[str]:
        """Returns a plausible adjacent key slip on a QWERTY keyboard."""
        is_upper = char.isupper()
        low = char.lower()
        neighbors = QWERTY_NEIGHBORS.get(low)
        if not neighbors:
            return None
        typo = random.choice(neighbors)
        # If original was uppercase, typo might be upper or accidental lower (Shift slip)
        if is_upper:
            return typo.upper() if random.random() < 0.65 else typo.lower()
        return typo

    @classmethod
    def type_like_human(cls, text: str, countdown_secs: int = 3) -> Tuple[bool, str]:
        """
        Simulates advanced, hyper-realistic human typing into whatever application has focus:
        1. Natural typing bursts (bursts of 3-7 characters, then micro-pause).
        2. QWERTY neighbor typos, latency hesitation, backspace deletion, and self-correction.
        3. Irregular cognitive pauses (reading transcripts, sentence boundary, comma cadence).
        4. Paragraph transitions and shift key delays.
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
        # Reset control states
        cls._pause_event.set()
        cls._stop_event.clear()
        cls._is_running = True
        cls._is_paused = False
        cls._current_index = 0
        cls._total_chars = len(text)

        logger.info(f"Starting advanced human typing simulation in {countdown_secs}s for {len(text)} characters...")

        if countdown_secs > 0:
            for _ in range(int(countdown_secs * 10)):
                if cls._stop_event.is_set():
                    cls._is_running = False
                    return False, "打字已由用户手动停止"
                time.sleep(0.1)

        punctuation_set = set("，。！？；：、,.!?;:\n\r\t")
        clause_breaks = set(",;，；")
        sentence_ends = set(".!?。！？")

        # Typing burst state
        burst_remaining = random.randint(3, 7)
        i = 0
        n = len(text)

        try:
            while i < n:
                if cls._stop_event.is_set():
                    logger.info("Typing simulation aborted by user stop event.")
                    return False, "打字已由用户手动停止"

                while not cls._pause_event.is_set():
                    if cls._stop_event.is_set():
                        return False, "打字已由用户手动停止"
                    time.sleep(0.1)

                cls._current_index = i + 1
                char = text[i]
                next_char = text[i + 1] if i + 1 < n else ""

                # --- ADVANCED FEATURE: REALISTIC HUMAN TYPO & BACKSPACE CORRECTION ---
                # Trigger realistic typo on English letters/numbers (~3.0% chance)
                # Avoid typos on punctuation, whitespace, or immediate first character of sentences
                is_eligible_for_typo = (char.isalnum() and char not in "\n\r " and i > 2 and text[i - 1] not in "\n\r")
                should_make_typo = is_eligible_for_typo and (random.random() < 0.030)

                if should_make_typo:
                    typo_char = cls._get_adjacent_typo(char)
                    if typo_char and typo_char != char:
                        typo_flavor = random.random()

                        if typo_flavor < 0.75:
                            # 1. Single character slip & immediate backspace correction (75% of typos)
                            cls._type_character_safe(typo_char)
                            # Human reaction latency: noticing typo on screen
                            time.sleep(random.uniform(0.18, 0.38))
                            # Delete the wrong character
                            cls._post_backspace(1)
                            # Micro-hesitation before typing correct key
                            time.sleep(random.uniform(0.10, 0.22))
                            # Proceed to type correct char below

                        elif typo_flavor < 0.95 and next_char and next_char.isalnum():
                            # 2. Runaway typo (2 characters before brain catches it, 20% of typos)
                            cls._type_character_safe(typo_char)
                            time.sleep(random.uniform(0.06, 0.12))
                            # Finger accidentally strikes next letter or neighboring key
                            runaway_char = next_char if random.random() < 0.6 else typo_char
                            cls._type_character_safe(runaway_char)
                            # Brain stops and notices
                            time.sleep(random.uniform(0.24, 0.44))
                            # Delete both characters
                            cls._post_backspace(2)
                            time.sleep(random.uniform(0.12, 0.25))
                            # Proceed to type correct char below

                        else:
                            # 3. Double key bounce slip (5% of typos: e.g. "aa" instead of "a")
                            cls._type_character_safe(char)
                            time.sleep(random.uniform(0.04, 0.08))
                            cls._type_character_safe(char)
                            # Notice repeated letter
                            time.sleep(random.uniform(0.18, 0.35))
                            cls._post_backspace(1)
                            time.sleep(random.uniform(0.08, 0.18))
                            # Already typed correct once, advance and continue!
                            i += 1
                            continue

                # Type the genuine intended character
                cls._type_character_safe(char)

                # --- ADVANCED CADENCE & IRREGULAR HUMAN PAUSES ---
                if char in "\n\r":
                    # Paragraph / line break: context switch pause (1.2s - 2.5s)
                    time.sleep(random.uniform(1.20, 2.50))
                    burst_remaining = random.randint(3, 7)
                elif char in sentence_ends:
                    # Sentence completion pause: review what was just composed (0.75s - 1.60s)
                    time.sleep(random.uniform(0.75, 1.60))
                    burst_remaining = random.randint(3, 7)
                elif char in clause_breaks:
                    # Clause breath pause (0.45s - 0.85s)
                    time.sleep(random.uniform(0.45, 0.85))
                    burst_remaining = random.randint(2, 5)
                elif char in " \t":
                    # Word boundary: check for word-level thinking pause
                    if random.random() < 0.07:
                        # Thinking of next phrasing (0.45s - 0.95s)
                        time.sleep(random.uniform(0.45, 0.95))
                    else:
                        time.sleep(random.uniform(0.14, 0.28))
                    burst_remaining = random.randint(3, 8)
                else:
                    # Regular letter keystroke inside a word
                    burst_remaining -= 1
                    if burst_remaining <= 0:
                        # Syllable / rhythm boundary micro-pause
                        time.sleep(random.uniform(0.16, 0.30))
                        burst_remaining = random.randint(3, 7)
                    else:
                        # Flowing stroke within a typing burst (fast & natural)
                        time.sleep(random.uniform(0.07, 0.15))

                # Occasional deep cognitive hesitation (unpredictable human thinking)
                rand_pause = random.random()
                if rand_pause < 0.009:
                    # Looking back at original transcripts / deep proofreading (2.0s - 3.8s)
                    time.sleep(random.uniform(2.0, 3.8))
                elif rand_pause < 0.032:
                    # Re-organizing clause structure in head (0.8s - 1.6s)
                    time.sleep(random.uniform(0.8, 1.6))

                # Extra physical reach delay for Shift / Capitalization / Brackets
                if char.isupper() or char in '()"“”':
                    if random.random() < 0.30:
                        time.sleep(random.uniform(0.10, 0.22))

                i += 1

            logger.info("Advanced human typing simulation finished successfully.")
            return True, ""
        except Exception as e:
            logger.error(f"Error during typing simulation: {e}")
            return False, f"打字模拟中断: {e}（已将全文复制到剪贴板，可按 Cmd+V 粘贴）"
        finally:
            cls._is_running = False
            cls._is_paused = False
