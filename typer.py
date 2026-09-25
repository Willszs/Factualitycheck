"""
Human Typing Simulator for macOS.
Simulates natural, realistic human typing cadence with punctuation pauses
and seamless Unicode/Chinese/Emoji support across any active application.
"""

import time
import random
import logging
import subprocess

logger = logging.getLogger("factuality.typer")


class HumanTyper:
    @staticmethod
    def get_clipboard() -> str:
        try:
            return subprocess.check_output(["pbpaste"]).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    @staticmethod
    def set_clipboard(text: str):
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        p.communicate(text.encode("utf-8"))

    @classmethod
    def type_like_human(cls, text: str, countdown_secs: int = 3, progress_callback=None):
        """
        Simulates natural human typing into whatever application/window currently has focus.
        """
        if not text:
            return

        logger.info(f"Starting human typing simulation in {countdown_secs}s for {len(text)} characters...")

        # Countdown delay allowing user to position cursor
        if countdown_secs > 0:
            time.sleep(countdown_secs)

        # Backup original clipboard to restore after typing
        original_clipboard = cls.get_clipboard()

        punctuation_set = set("，。！？；：、,.!?;:\n\r\t")

        try:
            for i, char in enumerate(text):
                # Copy single character to clipboard and simulate Cmd+V
                cls.set_clipboard(char)
                subprocess.run(
                    ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'],
                    capture_output=True,
                    timeout=2,
                )

                # Realistic cadence delays
                if char in punctuation_set:
                    # Longer natural pause at punctuation / newlines
                    time.sleep(random.uniform(0.18, 0.35))
                elif char in " \t":
                    time.sleep(random.uniform(0.05, 0.12))
                else:
                    # Standard keystroke typing rhythm (around 12-18 chars/sec)
                    time.sleep(random.uniform(0.025, 0.075))

            logger.info("Human typing simulation finished successfully.")
        except Exception as e:
            logger.error(f"Error during typing simulation: {e}")
        finally:
            # Restore original clipboard content
            time.sleep(0.2)
            cls.set_clipboard(original_clipboard)
