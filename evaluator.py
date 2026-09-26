"""
Factuality Evaluator module for comparing Model A and Model B multi-turn dialogues.
Uses Google Gemini API to analyze factual errors and generate an adaptive English summary.
Includes intelligent fallback across healthy models (3.6-flash, 3.8-flash, 3.1-flash-lite)
and automatic retry on transient Google server load spikes (503).
"""

import os
import re
import time
import logging
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

# Silence informational AFC warnings from google_genai
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("factuality.evaluator")

import datetime

def get_system_prompt() -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"""You are a rigorous, objective Conversational & Factuality Auditor for AI models.
You will be provided with multi-turn conversation transcripts from two different AI models (Model A and Model B) in the exact same scenario. Each transcript contains numbered dialogue turns (e.g., Turn 1, Turn 2, ...).

REFERENCE TIME:
- Today's date is: {now_str}. Keep this current temporal anchor in mind when evaluating recent events, ongoing schedules, and real-time announcements.

FACTUALITY AUDITING INTEGRITY & ANTI-HALLUCINATION SAFEGUARDS (CRITICAL):
1. Distinguish Genuine Regional/Recent Events vs. Hallucinations:
   - Modern music festivals (e.g., Strawberry Music Festival / 草莓音乐节, Midi, etc.) and cultural events regularly tour to various regional cities (e.g. Quanzhou 泉州, Changzhou 常州, Yancheng, etc.) with specific venues (e.g., coastal beach parks like 惠女海岸青山湾, or sports centers like 晋江体育中心).
   - NEVER falsely accuse an AI model of "inventing" an event or station merely because it is a regional city rather than a standard Tier-1 city (Beijing/Shanghai). Quanzhou Strawberry Music Festival (泉州草莓音乐节) genuinely exists!
   - Do NOT declare an event or entity "non-existent" unless it is an indisputable, logical, or physical impossibility. If a model mentions a specific regional event station with authentic details, do not penalize it based on your own knowledge cutoff or incomplete memory.
   - Do not unfairly favor a model that gave vague or evasive answers over a model that provided specific, real-world regional information.
2. Temporal Grounding:
   - Check whether events referenced match the realistic timeline relative to {now_str}. An event scheduled for the current autumn season is legitimate and should not be mistaken for past spring events.

STRICT EVALUATION SCOPE & TWO-DIMENSION OUTPUT FORMAT:
You MUST divide your evaluation into EXACTLY TWO DIMENSIONS, formatted as TWO PARAGRAPHS separated by EXACTLY ONE BLANK LINE:

PARAGRAPH 1: Conversational Dynamics
- MUST start with: "For conversational dynamics I prefer [Model A / Model B / neither model]."
- Scope: Evaluates conversation flow, formatting, dialogue habits, artificial search-simulation intros (e.g. "收到，我去查一下...", "好的，我去确认一下..."), anti-drift capability (e.g. handling aborted thoughts like "算了不说了" without inappropriately chasing tangents), directness, tone, and naturalness.
- Provide detailed justification referencing specific Turn [X] turns and behaviors.

(EXACTLY ONE BLANK LINE SEPARATOR)

PARAGRAPH 2: Utility
- MUST start with: "For utility I prefer [Model A / Model B / neither model]."
- Scope: Evaluates factuality, truthfulness, precision of domain knowledge (laws, regulations, articles, thresholds, dates, numbers, formulas, technical details), whether facts are correctly stated vs misstated, and verified Ground Truth.
- Provide detailed justification referencing specific Turn [X] factual statements and the verified Ground Truth.

REFERENCE BENCHMARK EXAMPLE:
For conversational dynamics I prefer Model A. Model B exhibited severe formatting and dialogue habit flaws, opening Turn 2, Turn 4, and Turn 6 with artificial search-simulation intros ("收到，我去查一下...", "好的，我去查一下...", "好的，我去确认一下...") instead of providing natural direct responses.

For utility I prefer Model A. Model B, in turn 6, misstated the voting thresholds under Article 278 of the Chinese Civil Code for dismissing property management, claiming that approval requires two-thirds of total area and homeowners, whereas the legal requirement is a two-thirds participation quorum followed by a simple majority (>50%) approval among participating votes.

STRICT FORMAT RULES:
- Output MUST be 100% in English.
- Always refer to dialogue turns strictly as "Turn 1", "Turn 2", "Turn 3", etc. NEVER use "Round 1", "Round 2".
- EXACTLY TWO PARAGRAPHS separated by EXACTLY ONE BLANK LINE.
- Each paragraph MUST start with the required sentence:
  Paragraph 1: "For conversational dynamics I prefer [Model A / Model B / neither model]. [Reasons...]"
  Paragraph 2: "For utility I prefer [Model A / Model B / neither model]. [Reasons...]"
- NO markdown headers (do NOT write "### Conversational Dynamics", "### Utility", or "### Verdict").
- NO bullet points (*, -) or numbered lists. Write flowing prose within each paragraph.
- NO conversational filler, greetings, or sign-offs. Start directly with "For conversational dynamics I prefer".
"""

SYSTEM_PROMPT = get_system_prompt()


class FactualityEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = (
            config.get("gemini_api_key")
            or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.primary_model = config.get("gemini_model", "gemini-3.6-flash").strip()

    def evaluate(self, transcript_a: str, transcript_b: str) -> Optional[str]:
        """
        Analyzes the two conversation transcripts and returns the English summary report.
        Automatically cycles through active models and handles transient Google load spikes.
        """
        if not self.api_key or "YOUR_" in self.api_key:
            logger.error("Gemini API key is not configured. Please check config.json or GEMINI_API_KEY.")
            return (
                "⚠️ Evaluation Failed: Gemini API Key is missing or invalid.\n"
                "Please configure 'gemini_api_key' in config.json."
            )

        user_content = (
            f"=== MODEL A DIALOGUE TRANSCRIPT ===\n{transcript_a.strip()}\n\n"
            f"=== MODEL B DIALOGUE TRANSCRIPT ===\n{transcript_b.strip()}\n"
        )

        # Verified active candidate models in priority order
        candidate_models = [self.primary_model]
        for fallback in ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = ""
        for model in candidate_models:
            for attempt in range(1, 3):
                logger.info(f"Evaluating with model [{model}] (attempt {attempt}/2)...")
                result, error_msg = self._call_model(model, user_content)
                if result:
                    return self.clean_evaluation_report(result)

                last_error = error_msg
                # If Google returns temporary 503 high demand or 429
                if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                    logger.warning(f"Model {model} busy on Google servers. Backing off 3s...")
                    time.sleep(3)
                else:
                    # Non-transient error, move immediately to next model
                    break

        return f"⚠️ Evaluation Notice: Google Gemini servers are temporarily congested (503). Last message: {last_error}"

    @staticmethod
    def clean_evaluation_report(text: str) -> str:
        """
        Sanitizes evaluation text into exactly two dimensions separated by a single blank line:
        1. For conversational dynamics I prefer...
        [blank line]
        2. For utility I prefer...
        """
        if not text:
            return ""

        # Remove bullet points
        text = re.sub(r"(\r?\n)\s*[\*\-•]\s*", " ", text)

        # Locate 'For conversational dynamics' to strip any preceding title/headers
        cd_match = re.search(r"(For\s+conversational\s+dynamics\s+I\s+prefer.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if cd_match:
            text = cd_match.group(1).strip()

        # Look for the split between conversational dynamics and utility
        match = re.search(r"(For\s+utility\s+I\s+prefer.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            p2 = match.group(1).strip()
            p1 = text[:match.start()].strip()
        else:
            # Fallback: split on double newline
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if len(parts) >= 2:
                p1 = parts[0]
                p2 = " ".join(parts[1:])
            else:
                p1 = text.strip()
                p2 = ""

        # Remove any trailing markdown headers from p1 (e.g. ### Utility)
        p1 = re.sub(r"###.*$", "", p1).strip()
        # Normalize internal spacing of paragraph 1
        p1 = re.sub(r"[\r\n]+", " ", p1)
        p1 = re.sub(r"\s{2,}", " ", p1).strip()

        # Normalize internal spacing of paragraph 2
        if p2:
            p2 = re.sub(r"###.*$", "", p2).strip()
            p2 = re.sub(r"[\r\n]+", " ", p2)
            p2 = re.sub(r"\s{2,}", " ", p2).strip()
            return f"{p1}\n\n{p2}"
        return p1

    @classmethod
    def clean_single_paragraph(cls, text: str) -> str:
        """Backwards-compatibility alias for clean_evaluation_report."""
        return cls.clean_evaluation_report(text)

    def _call_model(self, model_name: str, user_content: str) -> tuple[Optional[str], str]:
        # 1. Try google-genai SDK first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            prompt = get_system_prompt()
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    temperature=0.1,
                ),
            )
            if response and response.text:
                logger.info(f"Evaluation completed successfully via google-genai SDK ({model_name}).")
                return response.text.strip(), ""
        except Exception as e:
            err_str = str(e)
            logger.warning(f"google-genai SDK call for {model_name} failed ({err_str[:100]}), trying REST API...")

        # 2. Try REST API via requests (with proxy & SSL tolerance)
        return self._evaluate_via_rest(model_name, user_content)

    def _evaluate_via_rest(self, model_name: str, user_content: str) -> tuple[Optional[str], str]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.api_key}"
        )
        prompt = get_system_prompt()
        payload = {
            "system_instruction": {
                "parts": [{"text": prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1
            }
        }

        if requests is not None:
            for verify in [True, False]:
                try:
                    resp = requests.post(url, json=payload, timeout=60, verify=verify)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "").strip(), ""
                        return None, "Empty candidates in response"
                    elif resp.status_code == 503:
                        return None, f"503 Service Unavailable (Google {model_name} high demand)"
                    else:
                        return None, f"HTTP {resp.status_code}: {resp.text[:120]}"
                except Exception as req_err:
                    if verify:
                        continue  # retry with verify=False
                    return None, f"Network error: {req_err}"

        return None, "Requests library not available and SDK failed"
