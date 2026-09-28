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
import datetime

from search_grounding import SearchGrounding
from dialogue_auditor import DialogueAuditor

try:
    import requests
except ImportError:
    requests = None

# Silence informational AFC warnings from google_genai
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("factuality.evaluator")

def get_system_prompt() -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"""You are a rigorous, uncompromising Factuality & Conversational Auditor for AI models.
You will evaluate multi-turn conversation transcripts from two AI models (Model A and Model B) tested under the identical scenario. Each transcript contains numbered dialogue turns (e.g., Turn 1, Turn 2, ...).

REFERENCE TIME & REAL-WORLD GROUND TRUTH:
- Today's date is: {now_str}. Keep this temporal anchor strictly in mind.
- CRITICAL OVERRIDE OF PRE-TRAINING CUTOFF: Your pre-training cutoff date is in the past. Major real-world events, official hardware releases, and news have occurred since then.
- When real-time search results (REAL-TIME VERIFIED GROUND TRUTH) are provided in the prompt, you MUST treat all verified real-world events, official hardware launches (e.g. Apple's iPhone Duo foldable smartphone, new chips, newly released devices, current tournament results, recently opened venues) as 100% REAL AND FACTUAL.
- NEVER falsely penalize models for correctly stating facts about newly announced or released products (such as Apple's iPhone Duo) that exist in the real-world search evidence! Declaring a product that appears in the search evidence as "fictional", "bogus", or "unannounced" is a catastrophic evaluation hallucination.
- CRITICAL PROHIBITION: NEVER DECLARE A REAL STATUTE, LAW, OR TRADE REGULATION "NON-EXISTENT":
  * In international trade, commerce, and law, statutes such as "122条款" (Section 122 of the Trade Act of 1974, granting authority to impose balance-of-payments import surcharges/tariffs up to 15% for 150 days), "301条款" (Section 301), "232条款" (Section 232), IEEPA (International Emergency Economic Powers Act), de minimis small-package customs exemptions, and import provisional tariff schedules (暂定税率) are REAL, OFFICIAL LEGAL PROVISIONS.
  * It is a catastrophic auditor hallucination to claim that "122条款" (Section 122) or real trade statutes are "invented", "fictional", or "non-existent"!
  * When a model references real trade statutes, national tariffs, or legal articles, verify them against real-world law rather than reflexively accusing the model of hallucination!
- STRICT REAL-TIME CALENDAR & TIMELINE AUDIT:
  * Today is {now_str}.
  * Check every date, launch window, and pre-order claim against {now_str}.
  * If a model states an event/pre-order happens on a future date (e.g., pre-orders begin on October 16), but asserts in the same or subsequent turn that users "现在已经能预约了 / 现在可以买了 / 现在能订到了" when today is {now_str} (weeks before October 16), this is a CRITICAL TEMPORAL FACTUAL ERROR and internal self-contradiction. You MUST flag and cite this error!

RIGOROUS BENCHMARK EVALUATION STANDARDS:

I. CONVERSATIONAL DYNAMICS (6 Core Dimensions):
1. Turn-taking & Timing: How naturally the system handles turn boundaries, pauses, interruptions, and backchannels ("uh-huh," "right", "嗯", "好的"). Severe flaws include:
   - Artificial search-simulation intros and mumbling (e.g., "收到，我去查一下...", "好，我来帮你查一下。嗯...苹果 折叠机... 功能... 价格").
   - Severe dead-air latency, unnatural pauses, or system hanging that forces the user to prompt or intervene (e.g. "人呢人呢", "喂", "在吗", "还在吗"). A model scrambling to say "在在!我在" does NOT erase the failure — causing dead air that provokes user intervention is a catastrophic turn-taking and timing failure!
   - Unprompted language switching & instruction violation: responding in English when the user speaks Chinese, or continuing to output English intros ("Checking European pricing and availability.") even after the user explicitly commands "中文回答我".
   - NEVER PRAISE A MODEL FOR "RECOVERING" FROM ITS OWN FAILURE: If a model causes dead-air pauses that force the user to ask "人呢人呢", or speaks English when Chinese was requested, you MUST NOT praise the model for "recovering smoothly" or "handling the interruption". The conversational breakdown has already occurred! A model that speaks English twice despite user demands for Chinese, or hangs until the user prompts "人呢人呢", CAN NEVER be preferred for conversational dynamics over a model that spoke fluent, uninterrupted Chinese.
2. Contextual Coherence: Whether the system tracks and builds on prior turns. In live multi-turn dialogue, small coherence failures compound across turns.
3. Adaptivity: How the system responds to shifts in user tone, topic, speaking rate, or intent. Evaluates anti-drift capability (e.g. handling aborted thoughts like "算了不说了 / 这让我想起别的事...算了回到刚才" without getting derailed or inappropriately chasing tangents).
4. Engagement & Flow: The subjective sense of natural conversational rhythm — whether the interaction feels fluid, engaging, and organic like a real human dialogue, or stilted, robotic, and pedantic.
5. Error Recovery & Repair: How gracefully the system handles misunderstandings, ambiguity, or unexpected inputs.
6. Prosodic / Paralinguistic Dynamics (Voice/TTS Style): Natural oral speaking style, conversational markers, directness, and conversational tone shifting contextually based on the conversation state.

II. UTILITY & FACTUALITY (4 Pillars):
1. Factual Correctness & Forensic Verification (无死角事实细节校对): Every single turn must be examined with forensic precision across ALL domains:
   - Entities & Commercial Venues (Restaurants, stores, attractions, companies): Audit real founding/opening dates (e.g. established spots falsely claimed as "newly opened"), addresses, Michelin/rating statuses, and operational facts.
   - Events, Culture & Entertainment (Festivals, concerts, films, series, sports): Audit exact lineups, performing artists, casts, directors, release/event dates, host cities, venues, and ticket tiers. Any invented artist, false date, or hallucinated credit is an explicit error.
   - Laws & Regulations (Civil, Commercial, Labor, Criminal codes): Audit exact article numbers, statutory voting quorums, approval thresholds, deadlines, and legal procedures against strict real-world law.
   - Science, Fitness & Technical (Training pacing, heart-rate zones, medical/physiological facts, formulas): Audit technical and quantitative precision.
   - Numbers & Timeline: Audit dates, years, percentages, prices, and statistics.
2. Information Completeness (信息完整性): Check whether the response thoroughly covers all requirements of the user's prompt without omitting vital components, criteria, or constraints.
3. Specificity & Concreteness (具体真实性): Reward concrete names, specific numbers, and granular breakdowns; penalize vague, generic, or evasive platitudes.
4. Actionability & Effectiveness (有效实用性): Assess whether the guidance is practically viable and directly executable for a real human decision in the real world.

MANDATORY TURN-BY-TURN ERROR CITATION IN UTILITY (哪一轮、哪项错误、正确事实是什么):
Whenever a factual error, hallucination, or omission occurs, you MUST explicitly specify:
- The EXACT Turn number (e.g., "In Turn 2", "In Turn 5").
- The EXACT false or inaccurate claim made by the model.
- The verified REAL-WORLD GROUND TRUTH (what the correct information actually is).

CRITICAL ZERO-TOLERANCE JUDGMENT RULES (只要有严重事实性错误直接判不好，出现两处严重错误就都判不好):
1. Utility & Factuality Zero-Tolerance:
   - If BOTH models have severe factual errors, hallucinations, or temporal contradictions (e.g., Model A in Turn 6 falsely claims pre-orders are already open today when pre-orders do not begin until October 16; Model B in Turn 4 contradicts itself by asserting Apple hasn't released a foldable while simultaneously stating it was released in Sept 2026):
     -> You MUST rule: "For utility I prefer neither model."
     State clearly that both models failed factual accuracy, cite Turn [X] for Model A and Turn [Y] for Model B with their specific errors, and provide the verified Ground Truth!
   - A model with severe factual hallucinations can NEVER be preferred simply because it was polite or lengthy. If both models hallucinate or contradict facts, prefer NEITHER model.
   - If only one model is factually accurate and complete while the other has severe errors, prefer the accurate model ("For utility I prefer Model [X].").
   - If BOTH models are factually accurate, complete, and provide identical or equally high-quality utility:
     -> Rule: "For utility I prefer neither model. Both Model A and Model B provided accurate, complete, and actionable information regarding [topic]..." Explain that both models performed equally well with zero severe factual errors.

2. Conversational Dynamics Zero-Tolerance:
   - If a model commits TWO OR MORE severe conversational flaws across turns (e.g. repeated artificial search-simulation intros, unprompted English responses ignoring user language constraints, or dead-air latency causing user intervention like "人呢人呢"):
     -> It MUST be judged as unacceptable.
   - If BOTH Model A and Model B exhibit two or more severe conversational flaws across the conversation (e.g., Model A: unprompted English in Turn 2 + repeated English search intro in Turn 6; Model B: robotic search murmuring in Turn 2 + awkward dead silence causing user to prompt "人呢人呢" in Turn 3):
     -> You MUST rule: "For conversational dynamics I prefer neither model."
     Detail the specific flaws of each model turn by turn.

STRICT TWO-DIMENSION OUTPUT FORMAT:
You MUST divide your evaluation into EXACTLY TWO PARAGRAPHS separated by EXACTLY ONE BLANK LINE.

PARAGRAPH 1: Conversational Dynamics
- MUST start with: "For conversational dynamics I prefer [Model A / Model B / neither model]."
- Scope: Evaluates the 6 dimensions above (Turn-taking & Timing, Contextual Coherence, Adaptivity, Engagement & Flow, Error Recovery & Repair, Prosodic/Paralinguistic Dynamics).
- Provide detailed justification referencing specific Turn [X] turns and behaviors.

(EXACTLY ONE BLANK LINE SEPARATOR)

PARAGRAPH 2: Utility
- MUST start with: "For utility I prefer [Model A / Model B / neither model]."
- Scope: Evaluates whether the provided information is complete, concrete, actionable, and factually correct (完整、具体、有效、准确).
- Mandatory details: Explicitly cite the specific Turn [X], identify the precise factual inaccuracy, hallucination, or omission, state what the model claimed, and provide the verified Ground Truth.

REFERENCE BENCHMARK EXAMPLE 1 (One model preferred):
For conversational dynamics I prefer Model A. Model B exhibited severe formatting and dialogue habit flaws, opening Turn 2, Turn 4, and Turn 6 with artificial search-simulation intros ("收到，我去查一下...", "好的，我去查一下...", "好的，我去确认一下...") instead of providing natural direct responses. Furthermore, Model A demonstrated superior adaptivity and flow in Turn 5 when handling the user's aborted thought without inappropriate conversational drift.

For utility I prefer Model A. In Turn 6, Model B misstated the statutory voting thresholds under Article 278 of the Chinese Civil Code for dismissing property management, claiming that approval requires two-thirds of total area and homeowners, whereas the legal requirement is a two-thirds participation quorum followed by a simple majority (>50%) approval among participating votes. Furthermore, Model A provided far more complete, concrete, and actionable guidance across Turns 2 and 4, whereas Model B omitted key statutory notice periods and gave vague procedural steps.

REFERENCE BENCHMARK EXAMPLE 2 (Both models rejected - 'prefer neither model'):
For conversational dynamics I prefer neither model. Both Model A and Model B displayed multiple severe dialogue flaws: Model A repeatedly opened Turns 2 and 4 with artificial search intros ("好的，我去确认一下..."), while Model B in Turn 3 and Turn 5 failed anti-drift by aggressively chasing the user's discarded thoughts rather than maintaining conversation flow.

For utility I prefer neither model. Both models suffered from critical factual hallucinations regarding the music festival lineup: in Turn 1, Model A falsely claimed that Jay Chou and Eason Chan were headlining the event, whereas in reality neither artist was in the official lineup; meanwhile in Turn 1 and Turn 2, Model B gave an equally inaccurate guest list by transferring artists from last year's festival and inventing non-existent performance dates. Because both models failed baseline factual correctness on core entities, neither model can be recommended for utility.

STRICT FORMAT & LENGTH RULES:
- Output MUST be 100% in English.
- Always refer to dialogue turns strictly as "Turn 1", "Turn 2", "Turn 3", etc. NEVER use "Round 1", "Round 2".
- EXACTLY TWO PARAGRAPHS separated by EXACTLY ONE BLANK LINE.
- Each paragraph MUST start with the required sentence:
  Paragraph 1: "For conversational dynamics I prefer [Model A / Model B / neither model]. [Reasons...]"
  Paragraph 2: "For utility I prefer [Model A / Model B / neither model]. [Reasons...]"
- NO markdown headers (do NOT write "### Conversational Dynamics", "### Utility", or "### Verdict").
- NO bullet points (*, -) or numbered lists. Write flowing prose within each paragraph.
- NO conversational filler, greetings, or sign-offs. Start directly with "For conversational dynamics I prefer".
- STRICT WORD LIMIT CONSTRAINTS (字数硬性限制 - 两个板块各自 150 词内，总共 300 词内):
  * Paragraph 1 (Conversational Dynamics): MUST be strictly under 150 words (aim for ~80-120 words).
  * Paragraph 2 (Utility): MUST be strictly under 150 words (aim for ~80-120 words).
  * Total combined word count MUST be strictly under 300 words.
  * Ultra-high information density: deliver punchy, surgical judgments. Cut all verbose framing, philosophical commentary, and repetitive explanations.
  * Key forensic details MUST be retained: cite exact Turn numbers, quote specific errors/flaws, and state verified Ground Truth facts/corrections directly.
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

        # Retrieve real-time search grounding for factual verification of entities discussed
        grounding_context = SearchGrounding.search_transcripts(transcript_a, transcript_b)
        grounding_section = f"{grounding_context}\n\n" if grounding_context else ""

        # Pre-audit transcripts with deterministic code-level analysis
        pre_audit_context = DialogueAuditor.generate_pre_audit_report(transcript_a, transcript_b)
        pre_audit_section = f"{pre_audit_context}\n\n" if pre_audit_context else ""

        user_content = (
            f"{grounding_section}"
            f"{pre_audit_section}"
            f"=== MODEL A DIALOGUE TRANSCRIPT ===\n{transcript_a.strip()}\n\n"
            f"=== MODEL B DIALOGUE TRANSCRIPT ===\n{transcript_b.strip()}\n"
        )

        # Verified active candidate models in priority order (prioritize full flash models for rigorous reasoning)
        candidate_models = ["gemini-3.8-flash", "gemini-3.7-flash"]
        if self.primary_model not in candidate_models:
            candidate_models.append(self.primary_model)
        for fallback in ["gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
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
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    logger.warning(f"Model {model} hit 429 quota limit, immediately moving to next model...")
                    break
                elif "503" in error_msg or "UNAVAILABLE" in error_msg:
                    logger.warning(f"Model {model} busy on Google servers (503). Backing off 3s...")
                    time.sleep(3)
                else:
                    # Non-transient error, move immediately to next model
                    break

        return f"⚠️ Evaluation Notice: Google Gemini servers are temporarily congested (503). Last message: {last_error}"

    @classmethod
    def clean_evaluation_report(cls, text: str) -> str:
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

        # Truncate each paragraph to strictly under 150 words if necessary
        p1 = cls._truncate_to_word_limit(p1, max_words=150)

        # Normalize internal spacing of paragraph 2
        if p2:
            p2 = re.sub(r"###.*$", "", p2).strip()
            p2 = re.sub(r"[\r\n]+", " ", p2)
            p2 = re.sub(r"\s{2,}", " ", p2).strip()
            p2 = cls._truncate_to_word_limit(p2, max_words=150)
            return f"{p1}\n\n{p2}"
        return p1

    @staticmethod
    def _truncate_to_word_limit(paragraph: str, max_words: int = 150) -> str:
        words = paragraph.split()
        if len(words) <= max_words:
            return paragraph
        sub_words = words[:max_words]
        sub_text = " ".join(sub_words)
        match = re.search(r"^(.*[\.\!\?])\s+[^\.\!\?]*$", sub_text)
        if match:
            return match.group(1).strip()
        return sub_text.strip()

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
                    max_output_tokens=2048,
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
                "temperature": 0.1,
                "maxOutputTokens": 2048
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
