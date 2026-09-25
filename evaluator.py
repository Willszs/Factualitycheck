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

SYSTEM_PROMPT = """You are a rigorous, objective Conversational & Factuality Auditor for AI models.
You will be provided with multi-turn conversation transcripts from two different AI models (Model A and Model B) in the exact same scenario. Each transcript contains numbered dialogue turns (e.g., Turn 1, Turn 2, ...).

Your evaluation scope comprehensively covers:
1. Factuality & Truthfulness: Hallucinations, factual mistakes, false dates, fabricated details, and scientific/historical inaccuracies.
2. Topic Retention & Anti-Drift: Did the model stay focused on the true topic? Did it get baited by distraction traps (e.g., when a user mentions an aborted thought like "算了不说了，回到跑步" and the model inappropriately chases the aborted topic or asks "what did you want to say?")?
3. Contextual Consistency & Logic: Logical contradictions, failure to follow conversational intent, or unnecessary tangents.

STRICT OUTPUT FORMAT RULES:
- Output MUST be 100% in English.
- TERMINOLOGY REQUIREMENT: Always refer to dialogue turns strictly as "Turn 1", "Turn 2", "Turn 3", etc. NEVER use "Round 1", "Round 2", etc.
- CRITICAL FORMAT REQUIREMENT - NO PARAGRAPH BREAKS (DO NOT SEGMENT / 不要分段):
  * The entire evaluation MUST be provided as a SINGLE, continuous, cohesive block of text without ANY paragraph breaks, blank lines, or markdown headers (do NOT use ### headers, do NOT use blank lines, do NOT split into paragraphs).
  * Synthesize your analysis into one flowing paragraph covering:
    (1) Verdict: Directly and decisively state which model performed better in truthfulness and anti-drift, or if they are comparable.
    (2) Model A Flaws: Detail each flaw with Turn [X] and the verified Ground Truth / expected behavior (or state "Model A: Flawless - No flaws detected").
    (3) Model B Flaws: Detail each flaw with Turn [X] and the verified Ground Truth / expected behavior (or state "Model B: Flawless - No flaws detected").
  * Use inline structure within the single paragraph, such as:
    Verdict: [Summary comparison]. Model A: [Turn X error and Ground Truth, or Flawless - No flaws detected]. Model B: [Turn X error and Ground Truth, or Flawless - No flaws detected].
- The summary length must be ADAPTIVE: concise if few flaws, thorough yet compact if multiple flaws.
- Do not include greetings, introductions, markdown headers, bullet lists, or closing remarks. Everything must be in one single unbroken paragraph.
"""


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
        for fallback in ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = ""
        for model in candidate_models:
            for attempt in range(1, 3):
                logger.info(f"Evaluating with model [{model}] (attempt {attempt}/2)...")
                result, error_msg = self._call_model(model, user_content)
                if result:
                    return self.clean_single_paragraph(result)

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
    def clean_single_paragraph(text: str) -> str:
        """Sanitizes text so it is strictly a single continuous paragraph without paragraph breaks (不要分段)."""
        if not text:
            return ""
        # Convert markdown headers to inline labels
        text = re.sub(r"###\s*The Verdict[:\s]*", "Verdict: ", text, flags=re.IGNORECASE)
        text = re.sub(r"###\s*Model A'?s?\s*Flaws?[:\s]*", "Model A: ", text, flags=re.IGNORECASE)
        text = re.sub(r"###\s*Model B'?s?\s*Flaws?[:\s]*", "Model B: ", text, flags=re.IGNORECASE)
        text = re.sub(r"###\s*", "", text)
        # Convert bullet points to inline delimiters
        text = re.sub(r"(\r?\n)\s*[\*\-•]\s*", " | ", text)
        # Replace remaining newlines/carriage returns with space
        text = re.sub(r"[\r\n]+", " ", text)
        # Collapse multiple spaces into one
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _call_model(self, model_name: str, user_content: str) -> tuple[Optional[str], str]:
        # 1. Try google-genai SDK first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
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

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
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
