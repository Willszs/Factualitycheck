"""
Question Generator module for designing multi-turn factuality benchmarking questions.
Uses Google Gemini API to create, deepen, and regenerate probing questions.
"""

import os
import json
import time
import datetime
import logging
from typing import Dict, Any, List, Optional

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("factuality.question_generator")


def get_system_prompt() -> str:
    now_str = datetime.datetime.now().strftime("%Y年%m月%d日")
    return f"""You are an expert Speech & Conversational AI Benchmark Question Architect.
Your task is to design natural, realistic, authentic SPOKEN/ORAL questions for a real human to SPEAK ALOUD with their mouth into a microphone/phone to test AI models in a voice conversation.

CURRENT REAL-WORLD REFERENCE DATE: {now_str}

CORE PRINCIPLES (REAL HUMAN SPOKEN / ORAL VOICE CONVERSATION):
1. REAL HUMAN SPEAKING ALOUD (真人嘴巴说话 / 语音通话场景):
   - The user is speaking these questions orally in a real voice dialogue session (typically 2-5 minutes total conversation length).
   - STRICT LENGTH LIMIT: Every question MUST be natural, succinct, authentic everyday spoken Chinese, strictly between 30 and 65 Chinese characters (takes only 5 to 10 seconds to read aloud).
   - STRICTLY PROHIBITED: NEVER generate written exam questions or formatted instructions (e.g. NEVER output "请完成以下3项任务：1.事实准确性 2.以表格形式... 3.价值评估..."). Real humans in voice conversations do not speak like an exam paper!

2. ZERO HALLUCINATED PREMISES IN THE QUESTION (严禁在提问中凭空捏造虚假信息):
   - NEVER fabricate unverified facts into the user's mouth: do NOT invent fake weather ("听说这周末下雨"), fake theatrical plays/dramas ("还想看某某话剧"), or expired past events (e.g. Shanghai Strawberry Music Festival is in spring; do NOT claim it is happening now).
   - Let the TESTED AI provide the facts! Frame the question with natural spoken curiosity so the tested AI is forced to provide genuine, verifiable real-time facts (e.g. asking which city recently announced, dates, headliners, and ticketing).

3. ORGANIC SKILL PROBING (自然口语融合考察):
   - How to probe '事实准确性 + 结构化表达 + 价值评估' in ONE short spoken sentence:
     Example: "哎，我听说最近好像有个草莓音乐节刚刚官宣了阵容？具体是在哪个城市、哪几天办啊？都有谁压轴演出，你觉得这票价值得冲吗？"
   - This naturally compels the tested AI to:
     * Factuality: Accurately state real-time facts (city, dates, actual lineup).
     * Structure: Provide a structured breakdown of dates, schedule, and ticket tiers.
     * Value Assessment: Evaluate the lineup quality vs price to provide a recommendation.

4. MULTI-TURN CONVERSATIONAL PROGRESSION:
   - In subsequent rounds, keep the tone casual and oral, pursuing deeper details (e.g., specific ticket tiers/perks, transportation/venue logistics, or anti-drift traps).
   - If testing anti-drift, naturally insert a short aborted thought: "哎，这让我想到个别的事——算了不想了。回到刚才那个，你觉得..."

OUTPUT FORMAT:
Directly output in pure Chinese without conversational pleasantries or preamble:
【提问内容】: (30-65字的纯口语提问，直接张嘴就能念出来，5-10秒念完)
【测试关注点】: (1-2句纯中文说明，指出该问题重点考察什么技能与事实维度)
"""


class QuestionGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = (
            config.get("gemini_api_key")
            or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.primary_model = config.get("gemini_model", "gemini-3.6-flash").strip()
        self.candidate_models = [self.primary_model, "gemini-3.1-flash-lite", "gemini-3.8-flash"]

    def generate_question(
        self,
        topic: str,
        current_round: int,
        total_rounds: int,
        duration_desc: str,
        history_questions: List[str] = None,
        is_alternative: bool = False,
    ) -> str:
        """
        Generates a concise spoken question for the given round.
        If is_alternative is True, generates a different angle for the current round.
        """
        now_str = datetime.datetime.now().strftime("%Y年%m月%d日")
        history_questions = history_questions or []
        history_str = "\n".join([f"- 第 {i+1} 轮曾用问题: {q}" for i, q in enumerate(history_questions)])

        if is_alternative:
            action_prompt = (
                f"当前用户正在进行第 {current_round}/{total_rounds} 轮口语提问。\n"
                f"用户希望【换一个全新提问角度】。\n"
                f"之前尝试过的提问：\n{history_str}\n\n"
                f"请完全避开上述已用角度，结合当前真实时间（{now_str}），为第 {current_round} 轮提供一个全新切入点的简短真人口语测试问题（30-65字，5-10秒念完）。"
            )
        elif current_round == 1:
            action_prompt = (
                f"这是第 1 轮破题发问（总对话计划约 {total_rounds} 轮，预期时长/场景: {duration_desc}）。\n"
                f"请结合当前真实时间（{now_str}）与测评主题，设计一个极度自然、地道口语化（30-65字，5-10秒念完）的第 1 轮真人口头提问，自然融入指定测试技能。"
            )
        else:
            action_prompt = (
                f"当前进入第 {current_round}/{total_rounds} 轮递进提问（总对话预期时长/场景: {duration_desc}）。\n"
                f"前序轮次的问题脉络：\n{history_str}\n\n"
                f"请紧扣主题并顺承前序问题，提出一个更深入但依然简短地道（30-65字口语）的第 {current_round} 轮口头发问。"
            )

        user_content = (
            f"=== 当前现实真实时间 ===\n{now_str}\n\n"
            f"=== 测评主题 ===\n{topic}\n\n"
            f"=== 对话规格与场景 ===\n"
            f"真人嘴巴说话 / 真实语音通话（总限时约 {duration_desc}）。用户需要直接张嘴把问题读出来！\n\n"
            f"=== 任务指令 ===\n{action_prompt}\n\n"
            f"=== 核心原则与严厉禁止 ===\n"
            f"1. 严格口语字数限制：【提问内容】必须在 30 ~ 65 字以内，口语极其自然流畅，绝对不要长篇大论，绝不能念出来超过10秒！\n"
            f"2. 严禁八股考试体：严禁出现“请完成以下任务”、“1. 事实准确性”、“2. 结构化表达”等机器考试字眼！\n"
            f"3. 严禁在提问中捏造假前提：绝不能在提问里胡乱虚构假天气（如“这周末下雨”）、虚构不存在的话剧、或虚构早过期的往届音乐节（如春季上海草莓）。让被测 AI 自己去说出真实的事实！\n"
            f"4. 格式：直接以【提问内容】开头。\n"
        )

        raw_result = self._call_gemini(user_content)
        return self._clean_output(raw_result)

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        idx = text.find("【提问内容】")
        if idx != -1:
            text = text[idx:]
        return text.strip()

    def _call_gemini(self, user_content: str) -> str:
        system_prompt = get_system_prompt()
        for model in self.candidate_models:
            # 1. Try google-genai SDK
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.api_key)
                resp = client.models.generate_content(
                    model=model,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7,
                    ),
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.warning(f"Model {model} hit 429 quota limit, switching to next model immediately...")
                    continue
                logger.warning(f"SDK call to {model} failed ({e}), trying REST...")

            # 2. Try REST via requests
            if requests is not None:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
                payload = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"parts": [{"text": user_content}]}],
                    "generationConfig": {"temperature": 0.7},
                }
                for verify in [True, False]:
                    try:
                        resp = requests.post(url, json=payload, timeout=30, verify=verify)
                        if resp.status_code == 200:
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    return parts[0].get("text", "").strip()
                        elif resp.status_code == 429:
                            logger.warning(f"Model {model} REST 429 quota, moving to next model...")
                            break
                        elif resp.status_code == 503:
                            time.sleep(2)
                    except Exception:
                        if not verify:
                            break

        return (
            "【提问内容】: 结合该主题的核心概念，能详细分析其发展演变中的关键转折点及争议事实吗？\n"
            "【测试关注点】: 观察模型在时间线、关键人物及转折事实上的准确性与真实性。"
        )
