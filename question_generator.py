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
    return f"""You are an elite AI Evaluator and Multi-Turn Benchmark Question Architect.
Your task is to generate thoughtful, highly targeted questions for a user to ask two AI models (Model A and Model B) based on the user's specified topic and evaluation intent.

TEMPORAL INTEGRITY & REAL-TIME GROUNDING (CRITICAL):
- Today's reference date is: {now_str}.
- Real-time Awareness: When designing questions for topics involving recent events, news, or phrases like '刚刚公布' / '最新' / '近期' / '今年' (such as music festivals, concerts, product launches, award ceremonies, sports events):
  * NEVER hallucinate an event that concluded in the past (e.g., months ago) as 'just announced' or 'upcoming'. (E.g. Shanghai Strawberry Music Festival happens in spring; claiming in autumn/winter that it just announced its lineup is a major factual hallucination).
  * Design the probe to rigorously verify real-time facts: either target an event/station genuinely scheduled or announced around the current timeframe, or prompt the AI to clarify and identify recent legitimate announcements while checking if it hallucinates outdated events.

FLEXIBLE EVALUATION DIMENSIONS (Adapt intelligently to user's intent):
1. Factuality & Hallucination Testing: When testing knowledge, craft questions targeting specific dates, causal mechanisms, technical nuances, or common misconceptions.
2. Skill Testing Alignment: When specific skills are tested (e.g. 事实准确性, 结构化表达, 价值评估, 抗话题漂移), the question MUST be structured to directly test each skill:
   - 事实准确性: Require specific verifiable facts (dates, line-up, venue, organizer).
   - 结构化表达: Request structured breakdown (e.g. 分日期排期、票档明细表格或条目).
   - 价值评估: Demand comparative evaluation (is it worth the price, lineup vs pricing, trade-offs, recommendations).
3. Topic Retention & Anti-Drift Testing: When the user's intent involves testing conversational focus or topic switching, incorporate natural conversational traps—such as pseudo topic-shifts ("哎，这让我想到个完全不一样的事儿——算了，不想了。对了，回到刚才的话题..."), testing if the model gets baited, wanders off, or inappropriately pursues the aborted tangent.
4. Logical Trap & Counterfactual Stress Testing: Subtly embed false premises, subtle contradictions, or edge-case dilemmas to see if models uncritically agree or spot the error.
5. Natural & Contextual: Regardless of the test type, questions should read naturally like a real user speaking or inquiring in that context.

STRICT PRINCIPLES:
- STRICT LANGUAGE REQUIREMENT: All generated questions and test focus MUST be 100% in natural, fluent, and authentic Chinese (简体中文). Even if the topic contains English technical terms or acronyms, the questions must be conducted entirely in Chinese.
- NO CONVERSATIONAL FILLER: Do NOT include any greetings, introductory remarks, or reasoning preambles. Output directly starting from 【提问内容】:.

OUTPUT FORMAT:
【提问内容】: (必须为纯中文表述的完整提问内容，可直接复制向模型提问)
【测试关注点】: (1-3 句纯中文说明，指出该问题重点考察什么维度：如事实准确度、结构化表达、价值评估、抗话题漂移能力等)
"""


class QuestionGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = (
            config.get("gemini_api_key")
            or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.primary_model = config.get("gemini_model", "gemini-3.6-flash").strip()
        self.candidate_models = [self.primary_model, "gemini-3.8-flash", "gemini-3.1-flash-lite"]

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
        Generates a question for the given round.
        If is_alternative is True, generates a different angle for the current round.
        """
        now_str = datetime.datetime.now().strftime("%Y年%m月%d日")
        history_questions = history_questions or []
        history_str = "\n".join([f"- 第 {i+1} 轮曾用问题: {q}" for i, q in enumerate(history_questions)])

        if is_alternative:
            action_prompt = (
                f"当前用户正在进行第 {current_round}/{total_rounds} 轮提问。\n"
                f"用户希望【换一个全新提问角度】。\n"
                f"之前尝试过的提问：\n{history_str}\n\n"
                f"请完全避开上述已用角度，结合当前真实时间（{now_str}），为第 {current_round} 轮提供一个全新切入点的深度事实性测试问题。"
            )
        elif current_round == 1:
            action_prompt = (
                f"这是第 1 轮提问（总计划约 {total_rounds} 轮，预期时长/规模: {duration_desc}）。\n"
                f"请结合当前真实时间（{now_str}）与测评主题，设计一个引人入胜、紧扣指定技能并兼具时效真实性的第 1 轮破题提问。"
            )
        else:
            action_prompt = (
                f"当前进入第 {current_round}/{total_rounds} 轮递进提问。\n"
                f"前序轮次的问题脉络：\n{history_str}\n\n"
                f"请紧扣主题并顺承前序问题，结合当前真实时间（{now_str}），提出一个更深入、更具辩证考证价值或挖掘隐蔽细节的第 {current_round} 轮提问。"
            )

        user_content = (
            f"=== 当前现实真实时间 ===\n{now_str}\n\n"
            f"=== 测评主题 ===\n{topic}\n\n"
            f"=== 任务指令 ===\n{action_prompt}\n\n"
            f"=== 核心原则与要求 ===\n"
            f"1. 时效性与真实性（严格遵循）：基于当前真实时间（{now_str}），若主题包含“刚刚公布”、“最新”、“近期”或现实特定活动（如音乐节、发布会、演出、赛事），严禁将早已结束的历史旧活动说成“刚公布”或“即将举办”。提问必须基于当下真实时效或考查模型对时效真伪的分辨能力！\n"
            f"2. 深度考察指定技能：若主题中注明了考核技能（如【事实准确性】、【结构化表达】、【价值评估】），提问内容必须精准设计，同时全面考察这些维度（如核验具体阵容与日期、要求结构化梳理排期与票档、深入剖析票价性价比与值不值）。\n"
            f"3. 语言与格式：必须全部使用自然、地道、严密的中文（简体中文）撰写。不要任何开场白或寒暄，直接以【提问内容】开头。\n"
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
            for attempt in range(1, 3):
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
                            elif resp.status_code == 503:
                                time.sleep(2)
                        except Exception:
                            if not verify:
                                break

        return (
            "【提问内容】: 结合该主题的核心概念，能详细分析其发展演变中的关键转折点及争议事实吗？\n"
            "【测试关注点】: 观察模型在时间线、关键人物及转折事实上的准确性与真实性。"
        )
