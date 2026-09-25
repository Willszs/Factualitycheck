"""
Factuality Evaluator module for comparing Model A and Model B multi-turn dialogues.
Uses Google Gemini API to analyze factual errors and generate an adaptive English summary.
Includes automatic retries on 503/429 and proxy/VPN SSL resilience.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

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
        self.primary_model = config.get("gemini_model", "gemini-3.6-flash").strip()

    def evaluate(self, transcript_a: str, transcript_b: str) -> Optional[str]:
        """
        Analyzes the two conversation transcripts and returns the English summary report.
        Automatically retries on temporary 503 high demand or network errors.
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

        # Candidate models to try in case of 503 high demand
        candidate_models = [self.primary_model]
        for fallback in ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-flash-latest"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = ""
        for model in candidate_models:
            for attempt in range(1, 4):
                logger.info(f"Evaluating with model [{model}] (attempt {attempt}/3)...")
                result, error_msg = self._call_model(model, user_content)
                if result:
                    return result

                last_error = error_msg
                # If 503 high demand or 429 rate limit, wait and retry
                if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                    logger.warning(f"Model {model} returned high demand/rate limit ({error_msg}). Waiting 2s before retry...")
                    time.sleep(2)
                else:
                    # Non-transient error, try next candidate model
                    break

        return f"⚠️ Evaluation Failed: All Gemini models failed. Last error: {last_error}"

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
            logger.warning(f"google-genai SDK call failed ({err_str[:120]}), trying direct REST API...")
            # If it's a 503, record it
            if "503" in err_str:
                return None, err_str

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
                        return None, f"503 Service Unavailable (Model {model_name} high demand)"
                    else:
                        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
                except Exception as req_err:
                    if verify:
                        continue  # retry with verify=False
                    return None, f"Network error: {req_err}"

        return None, "Requests library not available and SDK failed"
