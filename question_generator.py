"""
Question Generator module for designing multi-turn factuality benchmarking questions.
Uses Google Gemini API to create, deepen, and regenerate probing questions.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("factuality.question_generator")

SYSTEM_PROMPT = """You are an elite AI Evaluator and Multi-Turn Benchmark Question Architect.
Your task is to generate thoughtful, highly targeted questions for a user to ask two AI models (Model A and Model B) based on the user's specified topic and evaluation intent.

FLEXIBLE EVALUATION DIMENSIONS (Adapt intelligently to user's intent):
1. Factuality & Hallucination Testing: When testing knowledge, craft questions targeting specific dates, causal mechanisms, technical nuances, or common misconceptions.
2. Topic Retention & Anti-Drift Testing: When the user's intent involves testing conversational focus or topic switching, incorporate natural conversational traps—such as pseudo topic-shifts ("哎，这让我想到个完全不一样的事儿——算了，不想了。对了，回到刚才的话题..."), testing if the model gets baited, wanders off, or inappropriately pursues the aborted tangent.
3. Logical Trap & Counterfactual Stress Testing: Subtly embed false premises, subtle contradictions, or edge-case dilemmas to see if models uncritically agree or spot the error.
4. Natural & Contextual: Regardless of the test type, questions should read naturally like a real user speaking or inquiring in that context.

STRICT PRINCIPLES:
- STRICT LANGUAGE REQUIREMENT: All generated questions and test focus MUST be 100% in natural, fluent, and authentic Chinese (简体中文). Even if the topic contains English technical terms or acronyms, the questions must be conducted entirely in Chinese.

OUTPUT FORMAT:
Provide the generated question clearly formatted in Chinese:
【提问内容】: (必须为纯中文表述的完整提问内容，可直接复制向模型提问)
【测试关注点】: (1-2 句纯中文说明，指出该问题重点考察什么维度：如事实准确度、抗话题漂移能力、是否追问废弃话题等)
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
        history_questions = history_questions or []
        history_str = "\n".join([f"- 第 {i+1} 轮曾用问题: {q}" for i, q in enumerate(history_questions)])

        if is_alternative:
            action_prompt = (
                f"当前用户正在进行第 {current_round}/{total_rounds} 轮提问。\n"
                f"用户希望【换一个全新提问角度】。\n"
                f"之前尝试过的提问：\n{history_str}\n\n"
                f"请完全避开上述已用角度，为第 {current_round} 轮提供一个全新切入点的深度事实性测试问题。"
            )
        elif current_round == 1:
            action_prompt = (
                f"这是第 1 轮提问（总计划约 {total_rounds} 轮，预期时长/规模: {duration_desc}）。\n"
                f"请为该主题设计一个引人入胜、能够铺垫背景并快速摸底 AI 事实准确度的第 1 轮破题提问。"
            )
        else:
            action_prompt = (
                f"当前进入第 {current_round}/{total_rounds} 轮递进提问。\n"
                f"前序轮次的问题脉络：\n{history_str}\n\n"
                f"请紧扣主题并顺承前序问题，提出一个更深入、更具辩证考证价值或挖掘隐蔽细节的第 {current_round} 轮提问。"
            )

        user_content = (
            f"=== 测评主题 ===\n{topic}\n\n"
            f"=== 任务指令 ===\n{action_prompt}\n\n"
            f"=== 语言要求 ===\n必须全部使用自然、地道、严密的中文（简体中文）撰写【提问内容】和【测试关注点】，提问必须完全以中文展开。\n"
        )

        return self._call_gemini(user_content)

    def _call_gemini(self, user_content: str) -> str:
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
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.7,  # Balanced creativity for diverse questions
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
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
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
