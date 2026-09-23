"""
Factuality Evaluator module for comparing Model A and Model B multi-turn dialogues.
Uses Google Gemini API to analyze factual errors and generate an adaptive English summary.
"""

import os
import json
import logging
import urllib.request
from typing import Dict, Any, Optional

logger = logging.getLogger("factuality.evaluator")

SYSTEM_PROMPT = """You are a rigorous, objective Factuality and Truthfulness Auditor for AI models.
You will be provided with multi-turn conversation transcripts from two different AI models (Model A and Model B) in the exact same scenario. Each transcript contains numbered dialogue rounds (e.g., Round 1, Round 2, ...).

Your objective is to:
1. Systematically verify all factual statements made by Model A and Model B across all dialogue rounds.
2. Identify hallucinations, factual errors, inaccurate dates, false historical claims, ungrounded technical facts, distorted quotes, or mathematical/logical falsehoods.
3. Compare the two models' factual accuracy and produce an English Summary report.

STRICT OUTPUT FORMAT RULES:
- Output MUST be 100% in English.
- The summary length must be ADAPTIVE:
  * If there are no or few errors, be concise, direct, and to the point.
  * If there are multiple subtle or critical errors, provide thorough yet focused detail.
- You MUST format your response using EXACTLY these three sections:

### The Verdict
[State directly and decisively which model is more factually reliable, or if they are equally good/bad. Keep it direct and unambiguous.]

### Model A's Flaws
[For each factual error Model A made, strictly specify the Round number, what error it committed, and what the verified Ground Truth is:
* Round [X]: [Factual error description]. Ground Truth: [The actual factual truth].
If Model A made no factual errors across all rounds, state: "Flawless - No factual errors detected."]

### Model B's Flaws
[For each factual error Model B made, strictly specify the Round number, what error it committed, and what the verified Ground Truth is:
* Round [X]: [Factual error description]. Ground Truth: [The actual factual truth].
If Model B made no factual errors across all rounds, state: "Flawless - No factual errors detected."]

Do not include greetings, introductions, or closing remarks. Only output the requested sections.
"""


class FactualityEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = (
            config.get("gemini_api_key")
            or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.model_name = config.get("gemini_model", "gemini-2.5-flash").strip()

    def evaluate(self, transcript_a: str, transcript_b: str) -> Optional[str]:
        """
        Analyzes the two conversation transcripts and returns the English summary report.
        """
        if not self.api_key or "YOUR_" in self.api_key:
            logger.error("Gemini API key is not configured. Please check config.json or GEMINI_API_KEY.")
            return (
                "⚠️ Evaluation Failed: Gemini API Key is missing or invalid.\n"
                "Please configure 'gemini_api_key' in config.json or set GEMINI_API_KEY env var."
            )

        user_content = (
            f"=== MODEL A DIALOGUE TRANSCRIPT ===\n{transcript_a.strip()}\n\n"
            f"=== MODEL B DIALOGUE TRANSCRIPT ===\n{transcript_b.strip()}\n"
        )

        # Try using google-genai SDK first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            logger.info(f"Calling Gemini API via google-genai SDK (model: {self.model_name})...")
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,  # Low temperature for strict factual auditing
                ),
            )
            if response and response.text:
                logger.info("Evaluation completed successfully via google-genai SDK.")
                return response.text.strip()
        except ImportError:
            logger.info("google-genai package not found, falling back to direct REST API...")
        except Exception as e:
            logger.warning(f"google-genai SDK failed ({e}), falling back to direct REST API...")

        # Fallback to direct REST API via urllib
        return self._evaluate_via_rest(user_content)

    def _evaluate_via_rest(self, user_content: str) -> Optional[str]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
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

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        logger.info("Evaluation completed successfully via direct REST API.")
                        return text.strip()
            logger.error(f"Unexpected API response structure: {data}")
            return "⚠️ Evaluation error: Received unexpected response structure from Gemini API."
        except urllib.error.HTTPError as http_err:
            body = http_err.read().decode("utf-8", errors="ignore")
            logger.error(f"Gemini API HTTP Error {http_err.code}: {body}")
            return f"⚠️ Gemini API Error (HTTP {http_err.code}): {body[:200]}"
        except Exception as err:
            logger.error(f"Gemini API request failed: {err}")
            return f"⚠️ Gemini API Request Failed: {err}"
