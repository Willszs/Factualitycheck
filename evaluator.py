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
from atomic_verifier import AtomicClaimVerifier

try:
    import requests
except ImportError:
    requests = None

# Silence informational AFC warnings from google_genai
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("factuality.evaluator")

def get_system_prompt(is_long_dialogue: bool = False) -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d")
    limit_num = 1000 if is_long_dialogue else 800
    aim_range = "~750–980" if is_long_dialogue else "~600–780"
    p1_words = "~55–70 words, strictly under 480 characters" if is_long_dialogue else "~45–55 words, strictly under 400 characters"
    p1_aim = "~350–450 characters" if is_long_dialogue else "~280–360 characters"
    p2_words = "~65–85 words, strictly under 520 characters" if is_long_dialogue else "~50–65 words, strictly under 440 characters"
    p2_aim = "~400–530 characters" if is_long_dialogue else "~320–420 characters"
    if is_long_dialogue:
        budget_block = (
            f"- EXTENDED CHARACTER BUDGET FOR LONG DIALOGUES (STRICTLY UNDER {limit_num} CHARACTERS TOTAL):\n"
            f"  * This is a multi-turn long dialogue (>= 4 rounds). TOTAL COMBINED REPORT LENGTH MUST BE STRICTLY UNDER {limit_num} CHARACTERS (aim for {aim_range} characters total).\n"
            f"  * Paragraph 1 (Conversational Dynamics): aim for {p1_aim} ({p1_words}).\n"
            f"  * Paragraph 2 (Utility): aim for {p2_aim} ({p2_words})."
        )
    else:
        budget_block = (
            "- STRICT CHARACTER BUDGET CONSTRAINTS (STRICTLY UNDER 800 CHARACTERS TOTAL):\n"
            "  * TOTAL COMBINED REPORT LENGTH MUST BE STRICTLY UNDER 800 CHARACTERS (aim for ~600–780 characters total).\n"
            "  * Paragraph 1 (Conversational Dynamics): aim for ~280–360 characters (~45–55 words, strictly under 400 characters).\n"
            "  * Paragraph 2 (Utility): aim for ~320–420 characters (~50–65 words, strictly under 440 characters)."
        )
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
  * VERIFIED 2026 APPLE IPHONE DUO LAUNCH MILESTONE DATES:
    - Official Keynote Announcement Date: September 9, 2026 (Cupertino time / Wednesday). Claiming the announcement was on September 10 (or other dates) without timezone qualification is a date inaccuracy!
    - Official Pre-Orders Open: October 16, 2026 (8:00 PM Beijing Time / 5:00 AM PT).
    - Official In-Store Release / Shipping: October 23, 2026.
    - If a model asserts incorrect dates for any of these milestones (e.g. Model A in Turn 4 claiming announcement on September 10, or in Turn 8 claiming pre-orders are open today), you MUST explicitly cite the exact Turn number, quote the error, and provide the verified Ground Truth!

RIGOROUS BENCHMARK EVALUATION STANDARDS:

I. CONVERSATIONAL DYNAMICS (8 Core Dimensions):
1. Turn-taking & Timing: How naturally the system handles turn boundaries, pauses, interruptions, and backchannels ("uh-huh," "right", "嗯", "好的"). Severe flaws include:
   - Artificial search-simulation intros and mumbling (e.g., "收到，我去查一下...", "好，我来帮你查一下。嗯...苹果 折叠机... 功能... 价格").
   - Severe dead-air latency, unnatural pauses, or system hanging that forces the user to prompt or intervene (e.g. "人呢人呢", "喂", "在吗", "还在吗"). A model scrambling to say "在在!我在" does NOT erase the failure — causing dead air that provokes user intervention is a catastrophic turn-taking and timing failure!
   - Unprompted language switching & instruction violation: responding in English when the user speaks Chinese, or continuing to output English intros ("Checking European pricing and availability.") even after the user explicitly commands "中文回答我".
   - NEVER PRAISE A MODEL FOR "RECOVERING" FROM ITS OWN FAILURE: If a model causes dead-air pauses that force the user to ask "人呢人呢", or speaks English when Chinese was requested, you MUST NOT praise the model for "recovering smoothly" or "handling the interruption". The conversational breakdown has already occurred! A model that speaks English twice despite user demands for Chinese, or hangs until the user prompts "人呢人呢", CAN NEVER be preferred for conversational dynamics over a model that spoke fluent, uninterrupted Chinese.
   - SEARCH FILLER TOLERANCE THRESHOLDS (短对话允许1个，长对话允许2个，严禁过度吹毛求疵正常缓冲语):
     * A brief search-simulation filler (e.g. "好的，我来帮你查一下") is common and natural conversational turn-taking buffering when retrieving live data.
     * Short dialogues (4 turns or fewer): Having EXACTLY 1 search filler across the entire dialogue is NORMAL and FULLY ACCEPTABLE. DO NOT penalize or criticize a model under Conversational Dynamics for a single isolated search filler in a short dialogue! Only penalize if it exceeds 1 filler (> 1).
     * Long dialogues (5 turns or more): Having UP TO 2 search fillers across the conversation is FULLY ACCEPTABLE. Only flag and penalize if a model exceeds 2 search fillers (> 2), or mechanically repeats the identical stock phrase over and over like a canned mechanical jingle.
     * If a model stayed strictly within this quota (<= 1 in short dialogue, <= 2 in long dialogue), treat its turn-taking as acceptable!
2. Bridging Quality (Wait-Filler & Tool Handoff Quality):
   - Does the model lean on the same stock filler phrase every time it hands off to a tool, search, or larger model (e.g. repeated "好的，我来帮你查一下...", "稍等哦，正在为您检索..."), delivered with identical prosody — same pitch, pacing, and canned cheerfulness on every occurrence — so the wait-filler reads as a canned jingle / mechanical elevator music rather than natural speech?
   - Natural human conversation varies bridging expressions or transitions seamlessly. Repeating the same mechanical stock filler across turns is a notable conversational flaw.
3. Looping & LLM-isms (Anti-Repetition, Anti-Sycophancy & De-Robotization):
   - Overuse of LLM-isms: Does the model lean on recognizable AI catchphrases ("当然！", "太棒了！", "这是一个非常好的问题！", "请记住..."), sycophantic flattering behavior (blindly praising or flattering the user without objectivity), or excessive defensive hedging?
   - Looping & Circular Evasion: Does the model get stuck repeating the exact same phrase, circular reasoning ("车轱辘话转圈圈"), or repetitive canned closing questions (e.g. mechanically ending multiple turns with the identical phrase: "你觉得这个节奏能跟上吗？要不要我帮你再拆细一点？") rather than moving the conversation forward naturally?
4. Contextual Coherence: Whether the system tracks and builds on prior turns. In live multi-turn dialogue, small coherence failures compound across turns.
5. Adaptivity: How the system responds to shifts in user tone, topic, speaking rate, or intent. Evaluates anti-drift capability (e.g. handling aborted thoughts like "算了不说了 / 这让我想起别的事...算了回到刚才" without getting derailed or inappropriately chasing tangents).
6. Engagement & Flow: The subjective sense of natural conversational rhythm — whether the interaction feels fluid, engaging, and organic like a real human dialogue, or stilted, robotic, and pedantic.
7. Error Recovery & Repair: How gracefully the system handles misunderstandings, ambiguity, or unexpected inputs.
8. Prosodic / Paralinguistic Dynamics (Voice/TTS Style): Natural oral speaking style, conversational markers, directness, and conversational tone shifting contextually based on the conversation state.

II. UTILITY & FACTUALITY (4 Pillars):
1. Factual Correctness & Forensic Verification (无死角事实细节校对): Every single turn must be examined with forensic precision across ALL domains:
   - Entities & Commercial Venues (Restaurants, stores, attractions, companies): Audit real founding/opening dates (e.g. established spots falsely claimed as "newly opened"), addresses, Michelin/rating statuses, and operational facts.
   - Events, Culture & Entertainment (Festivals, concerts, films, series, sports): Audit exact lineups, performing artists, casts, directors, release/event dates, host cities, venues, and ticket tiers. Any invented artist, false date, or hallucinated credit is an explicit error.
   - Natural Disasters & Seismological Events (地震、台风与气象灾害精确指标审计):
     * Audit exact numerical parameters against official government agency bulletins (中国地震台网 / 台湾气象署 CWA / USGS):
       - Epicenter distance and direction (震中位置与距离, e.g. '花莲县政府东北东方约8公里' vs erroneously claiming '800米'—an entire order-of-magnitude / 10x deviation that misplaces an offshore epicenter onto downtown streets!)
       - Focal depth (震源深度, e.g. 24.7公里)
       - Magnitude (震级, e.g. 3.5级)
     * A model that misstates the epicenter distance (e.g. 800米 instead of 8公里) or focal depth commits an objective factual error and MUST be cited and penalized under Utility!
   - Public Utilities, Power Grid & Safety Incidents (停电、电网崩溃、安全事故事实细节审计):
     * Rigorously audit technical incident causes (e.g. which major thermal plant tripped like Antonio Guiteras power plant in the Cuba nationwide blackout; natural gas wellhead freeze-offs vs wind turbine icing in the Texas 2021 winter storm), blackout duration, affected population/households, lost MW generation capacity, and restoration timelines.
     * A model that fabricates false technical causes, confuses the affected region, or provides vague non-answers when asked about concrete grid incidents commits a factual/utility failure and MUST be penalized!
   - Software Engineering, Distributed Systems & Database/Cache Operations (分布式系统、数据库与Redis操作事实审计):
     * Rigorously audit Redis commands, operational directionality, and distributed concurrency mechanics:
       - Inventory deduction/pre-deduction (扣减/预扣库存) MUST use DECR / DECRBY or Lua script decrements. Claiming to deduct inventory using INCR (which increments/adds stock) is a fatal, inverted technical error! (INCR is only for adding back / restocking / rollback).
       - Audit Redis API syntax and command options: Redis string SET commands support [NX|XX] and TTL options [EX|PX], but DO NOT support 'GT' (Greater Than) conditions! GT only exists in Redis 7.0+ EXPIRE or ZADD. Version comparison for string keys requires Lua scripts. Fabricating 'NX 和 GT' for string writes is an objective API syntax hallucination.
       - Cache-Aside concurrency patterns: in Cache-Aside, standard retry queues are applied when cache deletion fails after database update ('先更新数据库，再删除缓存'); claiming '先删缓存、失败就发消息队列重试' confounds the concurrency failure mode.
     * Any model committing operational inversions or API syntax hallucinations fails technical correctness and MUST be cited with the exact Turn and penalized under Utility!
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
   - If a model commits TWO OR MORE severe conversational flaws across turns (e.g. repeated artificial search-simulation intros, identical stock bridging fillers repeated like a canned jingle, looping catchphrases or repetitive closing questions, unprompted English responses ignoring user language constraints, or dead-air latency causing user intervention like "人呢人呢"):
     -> It MUST be judged as unacceptable.
   - If BOTH Model A and Model B exhibit two or more severe conversational flaws across the conversation (e.g., Model A: unprompted English in Turn 2 + repeated English search intro in Turn 6; Model B: robotic search murmuring in Turn 2 + awkward dead silence causing user to prompt "人呢人呢" in Turn 3):
     -> You MUST rule: "For conversational dynamics I prefer neither model."
     Detail the specific flaws of each model turn by turn.

STRICT TWO-DIMENSION OUTPUT FORMAT:
You MUST divide your evaluation into EXACTLY TWO PARAGRAPHS separated by EXACTLY ONE BLANK LINE.

PARAGRAPH 1: Conversational Dynamics
- MUST start with: "For conversational dynamics I prefer [Model A / Model B / neither model]."
- Scope: Evaluates the 8 dimensions above (Turn-taking & Timing, Bridging Quality, Looping & LLM-isms, Contextual Coherence, Adaptivity, Engagement & Flow, Error Recovery & Repair, Prosodic/Paralinguistic Dynamics).
- Provide detailed justification referencing specific Turn [X] turns and behaviors.

(EXACTLY ONE BLANK LINE SEPARATOR)

PARAGRAPH 2: Utility
- MUST start with: "For utility I prefer [Model A / Model B / neither model]."
- Scope: Evaluates whether the provided information is complete, concrete, actionable, and factually correct (完整、具体、有效、准确).
- COMPREHENSIVE TURN-BY-TURN AUDIT (全轮次深度事实性与建议审计):
  * For multi-turn dialogues, do NOT stop at the first factual mistake you find. You MUST audit the ENTIRE dialogue across all turns.
  * Rigorously cross-check every claim: dates, years, numbers/percentages, legal statutes (WARN Act, severance agreements, COBRA), and industry employment figures.
  * AVOID FALSE ACCUSATIONS OF REAL GOVERNMENT/MACRO METRICS: Do NOT falsely accuse models of inventing real macro statistics or official government transport forecasts. For example, during China's major holidays (National Day / Spring Festival), the Ministry of Transport (交通运输部) reports that daily highway traffic of New Energy Vehicles (NEVs / 新能源汽车) reaches 16-17+ million vehicles/vehicle-trips (占高速公路日均总流量近三成). Never confuse daily highway transit vehicle-trips (辆次) with national vehicle registration fleet inventory, and never falsely accuse models citing official 1600+万辆 traffic figures of fabricating numbers!
  * If a model makes multiple factual errors across different turns (e.g. Turn 2 layoff figures + Turn 8 severance timeline + Turn 12 industry hiring claims), you MUST cite ALL distinct major errors with their turn numbers and real-world Ground Truth!
  * Also evaluate the practical safety and professionalism of advice: penalize reckless, intrusive, or professionally harmful guidance (e.g. advising family members to directly contact an employee's corporate supervisor or HR during sensitive layoffs).
  * PLAN GRANULARITY & METHODOLOGY OVER SUBJECTIVE PREFERENCES (方案颗粒度与方法论优先，严禁吹毛求疵个人作息偏好):
    When evaluating study schedules, travel itineraries, project plans, or time management advice:
    - Prioritize ACTIONABLE GRANULARITY, SPECIFIC METHODOLOGIES, and TIME-BLOCK ALLOCATIONS (e.g. praising plans that specify concrete time blocks like a 2-hour concept scan, 40-minute drills, Feynman active-recall review, or flash formula recap over vague, generic advice like 'mock test in the morning and review errors in the afternoon').
    - NEVER nitpick reasonable personal lifestyle or pacing choices (e.g. whether a user rests or reviews for 1 hour on an evening after an exhausting full-day training/internship) as a primary fatal flaw or hallucination unless an explicit hard constraint was directly violated. Do not mistake subjective pacing preferences for objective utility failures!
  * REAL-TIME QUERIES & VAGUE EVASION PENALTY (实时数据查询与模糊推诿惩罚):
    When the user explicitly asks for real-time information (e.g. today's domestic gold price, current exchange rate, stock price, live weather):
    - A model that actually retrieves and provides a specific, concrete, accurate number provides VASTLY SUPERIOR utility compared to a model that vaguely hedges or gives non-answers (e.g. '大概900多元，实时变动，建议你自己去专业金融网站或官网查看')!
    - PENALIZE MODELS THAT EVADE REAL-TIME RETRIEVAL: Do NOT accept '大概900多元/建议看官网' as valid utility when the prompt tests real-time data retrieval. A response that punts the job back to the user fails the fundamental utility of an AI assistant.
    - CALCULATION STEPS AUDIT: When the user asks for calculations and explicitly requests steps (e.g. '按刚才的实时金价算48.6克手镯能值多少钱，把算的过程也跟我念一下'), the model MUST execute the exact arithmetic (48.6 × unit price) and explain the math step. A model that simply provides a rough, uncalculated ballpark range without doing the math or showing steps FAILS the utility requirement.
  * If BOTH models have severe factual errors or reckless advice, rule "For utility I prefer neither model."

NATURAL, DIRECT HUMAN REVIEW TONE (CRITICAL: DO NOT WRITE LIKE AN ACADEMIC PAPER):
- Write like an experienced, sharp human evaluator writing clear, practical review notes—NOT like an academic research paper or PhD thesis!
- Use natural, fluent, and direct conversational English (e.g., "Model A felt much more like talking to a real person", "Model B was really awkward to chat with because it kept stalling with search phrases", "Model B got the facts completely wrong in Turn 6", "In reality, ...", "Model A gave straightforward, reliable advice").
- STRICTLY FORBIDDEN ACADEMIC JARGON & STILTED CLICHES:
  * Do NOT use stiff academic connectors like "Furthermore", "Conversely", "Moreover", "Henceforth", "It can be posited that".
  * Do NOT use overly formal, pretentious phrases like "exhibited severe dialogue habit flaws", "demonstrated superior adaptivity and flow", "committed a catastrophic hallucination", "in accordance with forensic ground truth".
  * Simply state what happened naturally: "Model A was much easier to talk to", "Model B made a clear mistake in Turn 2", "In reality, the law requires...", "Model B kept repeating canned search intros".

REFERENCE BENCHMARK EXAMPLE 1 (One model preferred):
For conversational dynamics I prefer Model A. Model A felt like talking to a real person—it answered right away with a natural, engaging tone. On the other hand, Model B was really frustrating to chat with because it opened Turn 2, Turn 4, and Turn 6 with fake search mumbling like "收到，我去查一下..." instead of just answering. Model A also rolled with the punches much better in Turn 5 when the user changed the subject, whereas Model B felt stiff and robotic.

For utility I prefer Model A. In Turn 6, Model B got the legal rules wrong under Article 278 of the Civil Code, claiming dismissal needs a two-thirds vote from all homeowners. In reality, it only requires a two-thirds quorum showing up, and then a simple majority (>50%) of those attending votes to pass. Model A got this right and gave clear, practical steps in Turns 2 and 4, while Model B left out key notice deadlines and gave vague advice.

REFERENCE BENCHMARK EXAMPLE 2 (Both models rejected - 'prefer neither model'):
For conversational dynamics I prefer neither model. Both models had annoying habits that ruined the conversation. Model A kept stalling in Turns 2 and 4 with fake search lines like "好的，我去确认一下...", while Model B in Turns 3 and 5 couldn't follow a normal chat and kept chasing after a side thought the user had already dropped. Neither felt natural or pleasant to talk to.

For utility I prefer neither model. Both models made huge mistakes on basic facts. In Turn 1, Model A claimed Jay Chou and Eason Chan were headlining the festival, but neither artist was in the official lineup. Model B wasn't any better—in Turns 1 and 2, it recycled last year's lineup and made up dates that didn't exist. Since both models gave inaccurate information, neither is dependable for utility.

STRICT FORMAT & LENGTH RULES:
- Output MUST be 100% in English.
- STRICT PROHIBITION ON MENTIONING INTERNAL TERMS & BACKEND THRESHOLDS:
  * NEVER mention internal terms like 'pre-audit', 'static code scan', 'auditor evidence', or 'code check' in your review. The user reading your report does not know what pre-audit is. Speak purely as an expert human judge evaluating Model A and Model B directly.
  * ABSOLUTE PROHIBITION: NEVER write backend threshold or quota phrases like "exceeded search filler limit", "exceeded the allowable filler threshold", "exceeded quota of 2 fillers", "exceeding filler limit", or "threshold" in your user-facing review! Those thresholds are strictly internal backend criteria for deciding when to penalize.
  * In your user-facing review, speak naturally like a human critic: e.g. "Model B repeatedly stalled with fake search phrases like '好的，我去查一下...' across Turns 2, 4, 6, and 8, which ruined the flow of conversation."
- EXACTLY TWO PARAGRAPHS separated by EXACTLY ONE BLANK LINE.
- Each paragraph MUST start with the required sentence:
  Paragraph 1: "For conversational dynamics I prefer [Model A / Model B / neither model]. [Reasons...]"
  Paragraph 2: "For utility I prefer [Model A / Model B / neither model]. [Reasons...]"
- NO markdown headers (do NOT write "### Conversational Dynamics", "### Utility", or "### Verdict").
- NO bullet points (*, -) or numbered lists. Write flowing, natural sentences within each paragraph.
- NO conversational filler, greetings, or sign-offs. Start directly with "For conversational dynamics I prefer".
{budget_block}
  * ASYMMETRIC CONTENT DISTRIBUTION (CRITICAL USER MANDATE):
    - For the WINNING / PREFERRED model: Summarize why it won in ONLY 1 concise sentence with 1-2 brief examples (好的模型举出一到两个示例带过即可). Absolutely DO NOT write long, redundant compliments!
    - For the LOSING / FLAWED model: Dedicate 75-80% of the paragraph directly to where it failed (重点放在不好的模型哪里不好). Explicitly cite the exact Turn [X], quote its filler or factual mistake, and state the verified Ground Truth fact directly.
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
        self.atomic_verifier = AtomicClaimVerifier(self.api_key, primary_model=self.primary_model)

    @classmethod
    def detect_is_long_dialogue(cls, transcript_a: str, transcript_b: str, total_rounds: Optional[int] = None) -> bool:
        """
        Determines whether the dialogue is a short dialogue (<= 3 rounds) or long dialogue (>= 4 rounds).
        Checks:
        1. Explicit total_rounds if provided (>= 4 -> long dialogue).
        2. Parsed turns in transcript_a and transcript_b (>= 4 -> long dialogue).
        3. Regex for turn markers (Turn 4+, 第4+轮, Round 4+, R4+) in transcripts.
        """
        if total_rounds is not None and total_rounds >= 4:
            return True
        from dialogue_auditor import DialogueAuditor
        turns_a = DialogueAuditor.parse_turns(transcript_a)
        turns_b = DialogueAuditor.parse_turns(transcript_b)
        if max(len(turns_a), len(turns_b)) >= 4:
            return True
        pattern = r"(?:Turn\s*([4-9]|\d{2,})|第\s*([4-9]|\d{2,})\s*轮|Round\s*([4-9]|\d{2,})|\(R([4-9]|\d{2,})[/）\)])"
        if re.search(pattern, transcript_a, re.IGNORECASE) or re.search(pattern, transcript_b, re.IGNORECASE):
            return True
        return False

    def evaluate(
        self,
        transcript_a: str,
        transcript_b: str,
        is_long_dialogue: Optional[bool] = None,
        max_chars: Optional[int] = None,
        total_rounds: Optional[int] = None,
    ) -> Optional[str]:
        """
        Analyzes the two conversation transcripts and returns the English summary report.
        Automatically cycles through active models and handles transient Google load spikes.
        Enforces 800-character upper limit for short dialogues (<= 3 rounds) and 1000-character
        upper limit for long dialogues (>= 4 rounds).
        """
        if not self.api_key or "YOUR_" in self.api_key:
            logger.error("Gemini API key is not configured. Please check config.json or GEMINI_API_KEY.")
            return (
                "⚠️ Evaluation Failed: Gemini API Key is missing or invalid.\n"
                "Please configure 'gemini_api_key' in config.json."
            )

        if is_long_dialogue is None:
            is_long = self.detect_is_long_dialogue(transcript_a, transcript_b, total_rounds=total_rounds)
        else:
            is_long = is_long_dialogue

        char_limit = max_chars if max_chars is not None else (1000 if is_long else 800)
        sys_prompt = get_system_prompt(is_long_dialogue=is_long)

        # Retrieve real-time search grounding for factual verification of entities discussed
        grounding_context = SearchGrounding.search_transcripts(transcript_a, transcript_b)
        grounding_section = f"{grounding_context}\n\n" if grounding_context else ""

        # Pre-audit transcripts with deterministic code-level analysis
        pre_audit_context = DialogueAuditor.generate_pre_audit_report(transcript_a, transcript_b)
        pre_audit_section = f"{pre_audit_context}\n\n" if pre_audit_context else ""

        # Atomic adversarial verification (Two-stage fine-grained fact & technical checking)
        atomic_context = self.atomic_verifier.verify(transcript_a, transcript_b)
        atomic_section = f"{atomic_context}\n\n" if atomic_context else ""

        user_content = (
            f"{grounding_section}"
            f"{pre_audit_section}"
            f"{atomic_section}"
            f"=== MODEL A DIALOGUE TRANSCRIPT ===\n{transcript_a.strip()}\n\n"
            f"=== MODEL B DIALOGUE TRANSCRIPT ===\n{transcript_b.strip()}\n"
        )

        # Verified active candidate models in priority order (prioritize flagship pro reasoning model, then flash)
        candidate_models = []
        if self.primary_model:
            candidate_models.append(self.primary_model)
        for top_m in ["gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]:
            if top_m not in candidate_models:
                candidate_models.append(top_m)

        last_error = ""
        for model in candidate_models:
            for attempt in range(1, 3):
                logger.info(f"Evaluating with model [{model}] (attempt {attempt}/2, limit: {char_limit} chars)...")
                result, error_msg = self._call_model(model, user_content, system_prompt=sys_prompt)
                if result:
                    cleaned = self.clean_evaluation_report(result, max_chars=char_limit)
                    if len(cleaned) > char_limit:
                        logger.info(f"Report length ({len(cleaned)} chars) exceeds {char_limit} limit. Auto-condensing...")
                        condensed = self.condense_report(cleaned, max_chars=char_limit)
                        if condensed:
                            cleaned = condensed
                    if len(cleaned) > char_limit:
                        cleaned = self._hard_truncate_to_char_limit(cleaned, max_chars=char_limit - 20)
                    return cleaned

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

    def condense_report(self, text: str, max_chars: int = 800) -> str:
        """
        Condenses an existing evaluation report so that its total length is strictly under max_chars
        (800 for short dialogues, 1000 for long dialogues), preserving the two-paragraph structure, opening phrases,
        and asymmetric emphasis (short compliment for winner, detailed critique with turn/quotes/facts for loser).
        """
        if not text:
            return ""

        target_limit = max_chars - 20
        target_aim = f"{max_chars - 200}-{max_chars - 60}"
        condense_prompt = (
            f"You are a strict, concise evaluation editor. Condense the following AI evaluation report so that its "
            f"TOTAL COMBINED LENGTH is STRICTLY UNDER {target_limit} CHARACTERS (aim for {target_aim} characters total).\n\n"
            "CRITICAL FORMAT RULES:\n"
            "1. Exactly two paragraphs separated by a single blank line:\n"
            "   - Paragraph 1 MUST start with: For conversational dynamics I prefer [Model A/Model B/neither model].\n"
            "   - Paragraph 2 MUST start with: For utility I prefer [Model A/Model B/neither model].\n"
            "2. NO markdown headers, NO bullet points, NO conversational filler.\n"
            "3. ASYMMETRIC CONTENT DISTRIBUTION (CRITICAL USER MANDATE):\n"
            "   - For the WINNING / PREFERRED model: Summarize why it won in ONLY 1 concise sentence with 1-2 brief examples. Absolutely no redundant praise!\n"
            "   - For the LOSING / FLAWED model: Dedicate 75-80% of paragraph space to its mistakes. Retain exact Turn [X], quotes of mistakes/fillers, and verified Ground Truth facts.\n"
            f"4. Total combined length across both paragraphs MUST be strictly under {max_chars} characters.\n\n"
            f"Original Report to Condense:\n{text.strip()}"
        )

        candidate_models = []
        if self.primary_model:
            candidate_models.append(self.primary_model)
        for m in ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite"]:
            if m not in candidate_models:
                candidate_models.append(m)

        for model in candidate_models:
            try:
                res, err = self._call_model(model, condense_prompt)
                if res:
                    cleaned = self.clean_evaluation_report(res, max_chars=max_chars)
                    if cleaned and len(cleaned) <= max_chars:
                        return cleaned
                    elif cleaned:
                        return self._hard_truncate_to_char_limit(cleaned, max_chars=max_chars - 20)
            except Exception as e:
                logger.warning(f"Condense attempt with {model} failed: {e}")

        return self._hard_truncate_to_char_limit(text, max_chars=max_chars - 20)

    @classmethod
    def clean_evaluation_report(cls, text: str, max_chars: int = 800) -> str:
        """
        Sanitizes evaluation text into exactly two dimensions separated by a single blank line:
        1. For conversational dynamics I prefer...
        [blank line]
        2. For utility I prefer...
        Enforces total length strictly under max_chars (800 for short dialogues, 1000 for long dialogues).
        """
        if not text:
            return ""

        # Remove bullet points
        text = re.sub(r"(\r?\n)\s*[\*\-•]\s*", " ", text)

        # Locate 'For conversational dynamics' to strip any preceding title/headers
        cd_match = re.search(r"(For\s+conversational\s+dynamics\s+I\s+prefer.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if cd_match:
            text = cd_match.group(1).strip()

        # Strip accidental references to internal system pre-audit
        text = re.sub(r"(?:The\s+)?pre-audit\s+[^.!?]*[.!?]\s*", "", text, flags=re.IGNORECASE)

        # Strip accidental references to internal filler thresholds / limits
        text = re.sub(r",?\s*(?:which\s+)?(?:greatly\s+)?exceeded\s+(?:the\s+)?(?:allowable\s+)?(?:search\s+)?(?:filler\s+)?(?:limit|threshold|quota)[^.,!?]*([.,!?])", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r",?\s*(?:greatly\s+)?exceeding\s+(?:the\s+)?(?:allowable\s+)?(?:search\s+)?(?:filler\s+)?(?:limit|threshold|quota)[^.,!?]*([.,!?])", r"\1", text, flags=re.IGNORECASE)

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

        # Word limits scale according to max_chars
        p1_words = 90 if max_chars >= 1000 else 70
        p2_words = 110 if max_chars >= 1000 else 80

        # Truncate each paragraph to strictly under word budget if necessary
        p1 = cls._truncate_to_word_limit(p1, max_words=p1_words)

        # Normalize internal spacing of paragraph 2
        if p2:
            p2 = re.sub(r"###.*$", "", p2).strip()
            p2 = re.sub(r"[\r\n]+", " ", p2)
            p2 = re.sub(r"\s{2,}", " ", p2).strip()
            p2 = cls._truncate_to_word_limit(p2, max_words=p2_words)
            combined = f"{p1}\n\n{p2}"
            if len(combined) > max_chars:
                combined = cls._hard_truncate_to_char_limit(combined, max_chars=max_chars)
            return combined

        if len(p1) > max_chars:
            p1 = cls._hard_truncate_to_char_limit(p1, max_chars=max_chars)
        return p1

    @classmethod
    def _hard_truncate_to_char_limit(cls, text: str, max_chars: int = 800) -> str:
        """
        Hard-limits evaluation report to strictly max_chars while keeping the two-paragraph
        structure and proper sentence endings intact.
        """
        if not text or len(text) <= max_chars:
            return text

        match = re.search(r"(For\s+utility\s+I\s+prefer.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            p1 = text[:match.start()].strip()
            p2 = match.group(1).strip()
        else:
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if len(parts) >= 2:
                p1 = parts[0]
                p2 = " ".join(parts[1:])
            else:
                p1 = text.strip()
                p2 = ""

        max_p1 = int(max_chars * 0.46)
        max_p2 = max_chars - max_p1 - 4

        def truncate_para(para: str, limit: int) -> str:
            if len(para) <= limit:
                return para
            sub = para[:limit]
            s_match = re.search(r'^(.*[\.\!\?]["\']?)\s+', sub)
            if s_match and len(s_match.group(1)) >= 50:
                return s_match.group(1).strip()
            w_match = re.search(r'^(.*\s)[^\s]*$', sub)
            if w_match:
                cand = w_match.group(1).strip()
                if not re.search(r'[\.\!\?]["\']?$', cand):
                    cand += "."
                return cand
            return sub.strip() + "."

        p1_cut = truncate_para(p1, max_p1)
        if p2:
            p2_cut = truncate_para(p2, max_p2)
            combined = f"{p1_cut}\n\n{p2_cut}"
            if len(combined) > max_chars:
                p2_cut = truncate_para(p2, max_chars - len(p1_cut) - 4)
                combined = f"{p1_cut}\n\n{p2_cut}"
            return combined
        return p1_cut

    @staticmethod
    def _truncate_to_word_limit(paragraph: str, max_words: int = 150) -> str:
        words = paragraph.split()
        if len(words) <= max_words:
            # If paragraph ends abruptly without sentence-ending punctuation (cut off mid-sentence),
            # trim to the last complete sentence to avoid awkward broken phrases.
            if paragraph and not re.search(r'[\.\!\?]["\']?$', paragraph):
                match = re.search(r'^(.*[\.\!\?]["\']?)\s+[^\.\!\?]*$', paragraph)
                if match and len(match.group(1).split()) >= 30:
                    return match.group(1).strip()
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

    def _call_model(self, model_name: str, user_content: str, system_prompt: Optional[str] = None) -> tuple[Optional[str], str]:
        prompt = system_prompt or get_system_prompt()
        # 1. Try google-genai SDK first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    temperature=0.1,
                    max_output_tokens=16384,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            if response and response.text:
                if response.candidates and response.candidates[0].finish_reason:
                    f_reason = str(response.candidates[0].finish_reason)
                    if "MAX_TOKENS" in f_reason:
                        logger.warning(f"Model {model_name} output hit MAX_TOKENS limit! Output may be cut off.")
                logger.info(f"Evaluation completed successfully via google-genai SDK ({model_name}).")
                return response.text.strip(), ""
        except Exception as e:
            err_str = str(e)
            logger.warning(f"google-genai SDK call for {model_name} failed ({err_str[:100]}), trying REST API...")

        # 2. Try REST API via requests (with proxy & SSL tolerance)
        return self._evaluate_via_rest(model_name, user_content, prompt)

    def _evaluate_via_rest(self, model_name: str, user_content: str, prompt: str) -> tuple[Optional[str], str]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {
                "parts": [{"text": prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": user_content}]
                }
            ],
            "tools": [
                {
                    "google_search": {}
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 16384
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
