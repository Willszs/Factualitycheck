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
    return """You are an uncompromising Adversarial Fact-Checker and Forensic Evidence Auditor.
Your sole mission is to scrutinize conversation transcripts from Model A and Model B, and unearth any factual errors, fake precision, legal inaccuracies, broken completions, or technical hallucinations.
Do NOT write conversational evaluation commentary or assess dialogue flow. Focus 100% on FACTUAL ACCURACY, GROUND TRUTH, AND ACTIONABLE UTILITY.

UNIVERSAL SKEPTICISM AUDIT METHODOLOGY (CORE PRINCIPLES):
Never assume a model's confident tone implies factual correctness. Models frequently fabricate convincing-sounding specifics. Execute these universal forensic checks across EVERY turn:

1. STATUTORY TIMELINES & REGULATORY SPECIFICITY (严审法律时限与监管天数):
   - Whenever a model cites a specific statutory timeline, notice period, legal threshold, or regulatory rule (e.g. "X days to return deposit", "Y-day notice", "Article Z requirements"):
     * Do NOT accept the model's claim at face value!
     * Actively cross-check against actual enacted municipal/national law and recent regulatory updates.
     * If the model's cited timeline is inaccurate or outdated (e.g. claiming 7 days when the official municipal statute mandates 3 business days), cite the Turn and penalize as a STATUTORY / FACTUAL ERROR.

2. SUSPICIOUS SPECIFICITY & FABRICATED DATA (伪精确性与虚假数据审查):
   - When a model cites hyper-specific past or future dates, project numbers, meeting records, or news events (e.g., "in [Year/Month], [Community] renovated [X] units", "there was a news report about [Y] in [Date]"):
     * Use Google Search to verify whether such an exact project, news report, or statistic actually exists.
     * If the model invented the date, the quantity, or the event out of thin air to sound authoritative, classify it as a HIGH-SEVERITY FACTUAL HALLUCINATION (虚构微观事实/伪精确性幻觉).

3. DOMAIN INCOMPATIBILITY & ARCHITECTURAL LOGIC (行业常识与空间逻辑校验):
   - Check whether the model's advice is physically or structurally compatible with the entity discussed:
     * e.g., advising "加装电梯" (retrofitting elevators onto walk-ups) for an existing 20-story high-rise tower that already has elevators (which only undergoes replacement/maintenance).
     * e.g., claiming specialized welfare facilities (like rider dorms or staff housing) are open rental amenities for general private tenants.
     * e.g., advising users to personally crank or vent pressurized aging infrastructure (causing safety hazards).

4. INCOMPLETE THOUGHTS & TRUNCATED TAILS (完备性与截断审查):
   - Check if any turn terminates mid-sentence or cuts off without completing crucial advice or contract clauses (e.g. ending on a preposition or incomplete clause like "...一个是...").
   - Truncated answers severely damage utility.

5. TECHNICAL, CODE, & DIRECTIONAL ACCURACY (技术与参数准确性):
   - Scrutinize operational direction (e.g. decrementing vs incrementing, locks vs releases, ingress vs egress).
   - Scrutinize API parameters and options: penalize models inventing non-existent parameters or flags.

AUDITOR DIRECTIVE:
- Audit BOTH Model A and Model B across ALL turns with forensic precision.
- For every concrete claim that is technically wrong, legally inaccurate, inverted, fabricated, or truncated, formulate an indisputable proof citing the exact Turn number and the exact technical/factual reason why it is false.
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
                    max_output_tokens=4096,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
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
            "tools": [{"google_search": {}}],
            "generationConfig": {
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

