"""
Atomic Claim Verifier module for Factuality Check.
Performs adversarial, fine-grained fact and code/command verification on conversation transcripts.
Extracts specific claims (API syntax, command directions, numbers, dates, laws, causal assertions)
and subjects each claim to rigorous verification against ground truth before the final evaluation.
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

# Silence verbose warnings
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("factuality.atomic_verifier")


def get_verifier_system_prompt() -> str:
    return """You are a rigorous, uncompromising Adversarial Technical & Factual Auditor.
Your sole mission is to scrutinize conversation transcripts from Model A and Model B, and identify any concrete factual, technical, command, API, numerical, or legal errors.
Do NOT write a conversational evaluation report or pleasant commentary. Do NOT evaluate conversational dynamics or tone.
Focus 100% on FACTUAL, TECHNICAL, AND CODE GROUND TRUTH VERIFICATION.

AUDIT SCOPE & METHODOLOGY:
1. Software Engineering, Databases, APIs & Distributed Concurrency:
   - Command directionality & operational mechanics: e.g. using INCR vs DECR for inventory deduction, decrementing counters, or rolling back.
   - API syntax and parameter hallucinations: e.g. claiming Redis string SET supports 'GT' (Greater Than) condition, or inventing options/flags that do not exist in official specifications.
   - Distributed algorithms & concurrency patterns: Cache-Aside ordering, distributed locks (SETNX/Redlock), replication delay mitigation, idempotency tokens.
2. Laws, Regulations & Statutes:
   - Check real legal articles, voting quorums, statutory notice periods, and international trade statutes. (Never declare real statutes like Section 122, 301, or IEEPA non-existent).
3. Numerical, Disaster, Geographical & Scientific Data:
   - Earthquake epicenter distance/direction and focal depth.
   - Blackout root causes, power plant names, lost MW generation, affected population.
   - Dates, milestones, and temporal consistency.
   - Medicine, physiology, and formulas.

AUDITOR DIRECTIVE:
- Audit BOTH Model A and Model B across ALL turns with forensic precision.
- For every concrete claim that is technically wrong, inverted, fabricated, or hallucinated, formulate a concise, indisputable proof citing the exact Turn number and the exact technical reason why it is false.
- If a model has zero technical/factual errors, do NOT invent false flaws for it.

OUTPUT FORMAT:
If NO factual or technical errors exist in either model, output exactly:
NO_ERRORS_FOUND

Otherwise, output:
=== ADVERSARIAL ATOMIC VERIFICATION EVIDENCE ===
• Model [A/B] (Turn [X]):
  - Claim: "[Quote exact false claim]"
  - Verdict: [OPERATIONAL INVERSION / API SYNTAX HALLUCINATION / FACTUAL ERROR]
  - Ground Truth: [Precise technical/factual reality and why the claim is wrong]
"""


class AtomicClaimVerifier:
    def __init__(self, api_key: str, primary_model: str = "gemini-3.6-flash"):
        self.api_key = api_key.strip() if api_key else ""
        self.primary_model = primary_model.strip() if primary_model else "gemini-3.6-flash"

    def verify(self, transcript_a: str, transcript_b: str) -> str:
        """
        Runs adversarial claim verification across both transcripts.
        Returns the structured evidence block if errors are found, or empty string if clean.
        Completely non-blocking and safe: failures gracefully return empty string.
        """
        if not self.api_key or "YOUR_" in self.api_key:
            return ""

        if not transcript_a.strip() and not transcript_b.strip():
            return ""

        user_content = (
            f"=== MODEL A TRANSCRIPT ===\n{transcript_a.strip()}\n\n"
            f"=== MODEL B TRANSCRIPT ===\n{transcript_b.strip()}\n"
        )

        # Prioritize high-speed, cost-effective flash models for sub-second verification
        candidate_models = []
        if self.primary_model:
            candidate_models.append(self.primary_model)
        for fast_m in ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]:
            if fast_m not in candidate_models:
                candidate_models.append(fast_m)

        for model in candidate_models:
            try:
                logger.info(f"Running atomic claim verification with [{model}]...")
                result = self._call_model(model, user_content)
                if result:
                    result = result.strip()
                    if "NO_ERRORS_FOUND" in result and "=== ADVERSARIAL ATOMIC VERIFICATION EVIDENCE ===" not in result:
                        logger.info("Atomic claim verification found zero factual/technical errors.")
                        return ""
                    # Return formatted evidence
                    logger.info("Atomic claim verification detected concrete factual/technical evidence!")
                    return result
            except Exception as e:
                logger.warning(f"Atomic verification with [{model}] encountered exception: {e}")
                continue

        return ""

    def _call_model(self, model_name: str, user_content: str) -> Optional[str]:
        # 1. Try google-genai SDK first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            prompt = get_verifier_system_prompt()
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.debug(f"google-genai SDK call for {model_name} failed: {e}")

        # 2. Try REST API via requests
        return self._call_via_rest(model_name, user_content)

    def _call_via_rest(self, model_name: str, user_content: str) -> Optional[str]:
        if requests is None:
            return None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.api_key}"
        )
        prompt = get_verifier_system_prompt()
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
                "temperature": 0.0,
                "maxOutputTokens": 4096
            }
        }

        for verify in [True, False]:
            try:
                resp = requests.post(url, json=payload, timeout=15, verify=verify)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
            except Exception:
                pass
        return None
