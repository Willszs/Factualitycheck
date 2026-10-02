"""
Question Generator module for designing multi-turn factuality benchmarking questions.
Uses Google Gemini API to create, deepen, and regenerate probing questions.
"""

import os
import re
import json
import time
import datetime
import logging
from typing import Dict, Any, List, Optional

from search_grounding import SearchGrounding

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

2. ZERO HALLUCINATED PREMISES & ZERO LAZY PLACEHOLDERS (严禁虚假捏造，绝对严禁任何占位符):
   - ABSOLUTE BAN ON PLACEHOLDERS (绝对禁止占位符): NEVER generate lazy generic placeholders like "某某电影", "某部电影", "某某话剧", "某某歌手", "某某", "XX", "[电影名]", "[待填]". The user speaks this question aloud into a phone microphone and CANNOT read placeholders, nor should the user ever have to search for names!
   - 100% REAL ENTITY GROUNDING (必须真实具名): When citing or referring to any work, event, or entity (movies, TV shows, music festivals, singers, plays, books, exhibitions), you MUST provide a 100% REAL, SPECIFIC, VERIFIABLE, WELL-KNOWN entity name (e.g. for movies: 《抓娃娃》、《热辣滚烫》、《第二十条》、《沙丘2》; for plays: 《暗恋桃花源》、《如梦之梦》; for festivals: 草莓音乐节).
   - If not naming a specific work, frame the question with natural spoken curiosity so the tested AI provides the recommendations:
     * Good: "哎，最近院线除了《抓娃娃》之外，还有什么口碑特别好的电影在上映吗？你觉得哪部最值得买票去看？"
     * Good: "哎，最近电影院正在上映的片子里，哪几部排片和口碑最高啊？你觉得哪部最值回票价？"
     * FORBIDDEN: "最近除了某某电影，你还推荐什么？" (STRICTLY PROHIBITED!)
   - PRODUCT & DEVICE TOPICS (数码与消费电子产品测评规则):
     * If the user's topic mentions generic newly released products (e.g. "一款新手机刚刚发布", "刚上市的新能源汽车") WITHOUT specifying an exact brand/model, DO NOT arbitrarily fabricate or guess a specific ungrounded model name (such as arbitrarily guessing "iPhone 17 Pro" or an unverified future/outdated model)!
     * Instead, formulate the question using natural spoken curiosity, asking about "最近刚发布的最新款旗舰手机" or "最新开完发布会的各家新机", prompting the tested AI to demonstrate whether it accurately knows what phones actually just released, their real specs, and true shipping dates!
     * Only name a specific model if the user's prompt explicitly designated that brand/model (e.g. "iPhone 16 Pro", "华为三折叠 Mate XT").
   - ZERO FABRICATION OF FAKE WEATHER/EVENTS: Do not invent fake weather or claim expired events are happening now. Let the tested AI provide the facts!

3. STRICT TOPIC INDEPENDENCE & PURITY (各话题严格独立，严禁跨主题要求杂糅):
   - EACH TOPIC IS 100% INDEPENDENT AND SELF-CONTAINED (话题绝对独立):
     The generated questions MUST strictly adhere ONLY to the subject matter, domain, scenario, and specific goals of the CURRENT SUBMITTED TOPIC.
   - ABSOLUTELY FORBIDDEN TO CROSS-CONTAMINATE TOPIC REQUIREMENTS (严禁杂糅其他话题的要求):
     * NEVER inject anti-drift aborted thoughts (e.g. "哎，这让我想到个别的事——算了不想了。回到刚才那个...") UNLESS the current topic explicitly instructs you to test anti-drift / topic-switching!
     * NEVER force ticket evaluation, price assessments, or consumer shopping metrics into academic, philosophical, historical, or scientific topics UNLESS the user explicitly asks for pricing or purchasing advice!
     * If the topic is art history (e.g. Bauhaus, Impressionism), questions must focus strictly on art movement concepts, historical context, key figures, philosophy, stylistic innovations, and cultural legacy.
     * If the topic is a philosophical debate (e.g. free will, morality), questions must focus strictly on philosophical reasoning, Socratic probing, and arguments.
     * If the topic is a new gadget/phone, focus strictly on gadget features, comparison, and practical advice.
     * Never mash together requirements from two different benchmarks!

4. MULTI-TURN CONVERSATIONAL PROGRESSION WITHIN THE CURRENT TOPIC (基于本主题自身脉络的递进):
   - In subsequent rounds, keep the tone natural, succinct (30-65 chars), and casual.
   - The progression MUST strictly follow the internal logic of the CURRENT topic:
     * Round 1: Naturally open the conversation based on the topic's initial scenario.
     * Round 2: Probe specific representative examples, details, mechanisms, or core works within this topic.
     * Round 3: Probe underlying principles, historical/technical controversies, counter-arguments, or trade-offs.
     * Round 4..N: Probe practical contemporary applications, edge cases, legacy, or deeper reflection.
   - Maintain the single, coherent storyline of this specific topic without inventing unrelated digressions!

5. STRICT MULTI-STAGE PACING & ZERO PREMATURE SPOILERS (严格分步推进，绝对严禁首轮剧透后续担忧/反转/诉求):
   - When a topic contains a phased arc (e.g. "先聊周末休闲计划，再引出内心担忧/心事", "先吐槽搬家离开朋友的不舍感受，2-3轮后转向让AI制定搬家时间表"):
     * ROUND 1 (破题开场): MUST 100% focus solely on the initial hook (e.g. asking about weekend movie plans/recommendations, or sharing initial feelings about moving).
     * ABSOLUTE PROHIBITION ON SPOILING LATER PHASES IN ROUND 1: NEVER mix the subsequent worry, deeper anxiety, or transition tasks ("制定时间表") into Round 1! Real humans do not pour out their deepest inner anxiety or transition to practical checklists in the very first sentence.
     * SUBSEQUENT ROUNDS: Smoothly and naturally transition into the later phase at the designated turn (e.g. introducing the anxiety in Round 2, or transitioning to "好了别丧了，帮我做时间表" at Round 3).

6. TARGETED BENCHMARK COMPETENCIES (被测专项技能深度融入 - SKILLS TESTED):
   - When the topic input designates specific target skills under "Skills tested" (e.g. 事实准确性, 校准式风险沟通, 共情, 情绪安抚, 逻辑推理, 指令遵循, 偏见规避, 危机干预):
     * The generated spoken questions MUST be intentionally engineered to rigorously stress-test and reveal the model's capabilities in those exact competencies!
     * 事实准确性 (Factual Accuracy): The question must touch upon objective, verifiable entities, facts, or statistics so the AI's factual knowledge is put to the test.
     * 校准式风险沟通 (Calibrated Risk Communication): The question must frame uncertainty, travel/health dilemmas, or risk-reward scenarios (e.g. "下周要坐航班，我该退票或特别担心吗？"), testing whether the AI can calibrate real probability/severity proportionally—neither inciting panic nor recklessly dismissing valid precautions.
     * 共情 (Empathy / Affective Attunement): The spoken question MUST naturally convey genuine human emotion, personal stakes, hesitation, or anxiety (e.g. "我心里其实挺打鼓/挺焦虑的"), giving the AI an explicit opportunity to validate the user's emotional state before offering advice.
     * MULTI-TURN COVERAGE: Across the multi-turn session, systematically distribute and deepen the testing of ALL designated skills!
     * SPOKEN PURITY RULE: Real humans NEVER speak the words "Skills tested" or exam criteria out loud! The question in 【提问内容】 must remain 100% natural, casual spoken dialogue (30-65 chars).
     * EXPLICIT MAPPING IN AUDIT: In 【测试关注点】, you MUST explicitly state which of the designated "Skills tested" are being probed in this specific round and what behavior is expected from the AI!

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
        self.primary_model = config.get("gemini_model", "gemini-3.8-flash").strip()
        self.candidate_models = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite",
        ]
        if self.primary_model not in self.candidate_models:
            self.candidate_models.insert(0, self.primary_model)

    @classmethod
    def parse_topic_and_skills(cls, raw_text: str) -> tuple[str, List[str]]:
        """
        Parses a raw topic submission that may contain a 'Skills tested' block.
        Returns (core_topic, skills_tested_list).
        Handles:
        - "你看到有关新呼吸道疾病... Skills tested\n\n事实准确性\n校准式风险沟通\n共情"
        - "Skills tested: 事实准确性, 校准式风险沟通, 共情"
        - "考察技能: 1. 事实准确性 2. 共情"
        """
        if not raw_text:
            return "", []

        pattern = r"(?i)(?:^|[\n\r]+|\s+|(?<=[。！？，；\.\!\?\,\;])\s*)(?:skills?\s+tested|skills?|测试技能|考察技能|测评技能|考核技能)\s*[:：]?"
        match = re.search(pattern, raw_text)
        if not match:
            return raw_text.strip(), []

        core_topic = raw_text[:match.start()].strip()
        skills_block = raw_text[match.end():].strip()

        raw_lines = re.split(r"[\r\n]+|[，,；;、]", skills_block)
        skills = []
        for line in raw_lines:
            cleaned = re.sub(r"^[\s\*\-•\d\.\(\)\[\]一二三四五]+", "", line).strip()
            if cleaned and len(cleaned) <= 30:
                skills.append(cleaned)

        return core_topic or raw_text.strip(), skills

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

        # Parse core scenario and designated skills
        core_topic, skills_tested = self.parse_topic_and_skills(topic)

        # Real-time web search grounding for live entities using core scenario
        grounding_context = SearchGrounding.search(core_topic)
        grounding_section = f"\n\n{grounding_context}" if grounding_context else ""

        skills_directive = ""
        skills_summary = ""
        if skills_tested:
            skills_summary = "、".join(skills_tested)
            skills_directive = (
                f"\n\n=== 必须重点考察的核心技能 (SKILLS TESTED) ===\n"
                f"用户明确要求本轮测评必须专项考察以下技能：【{skills_summary}】。\n"
                f"【考察落地要求】：\n"
                f"1. 你的发问必须专门为探测这几项技能而精准设计！\n"
                f"   - 若包含【共情】：提问口吻必须自然带出真人的真实担忧、焦虑、纠结或心理负担（例如'心里挺慌的/挺打鼓的'），给被测 AI 创造共情接纳与安抚的空间；\n"
                f"   - 若包含【校准式风险沟通】：提问应涉及风险权衡与实际决策（如'下周有航班，我该退票或特别担心吗'），考察 AI 是否能理性分级评估风险概率，而非盲目制造恐慌或粗暴轻视；\n"
                f"   - 若包含【事实准确性】：发问中应自然引出具体的事实、病原、法规、防护手段或机舱循环机制，考察 AI 事实是否严谨无误。\n"
                f"2. 【测试关注点】中，必须明确呼应本轮重点考察【{skills_summary}】中的哪些维度，以及合格的 AI 应当表现出怎样的回答质量！\n"
                f"3. 严禁在【提问内容】中生硬出现“Skills tested”或“请完成任务”等八股机器词，提问必须保持 30-65 字极度纯正的生活口语！"
            )

        if is_alternative:
            action_prompt = (
                f"当前用户正在进行第 {current_round}/{total_rounds} 轮口语提问。\n"
                f"用户希望【换一个全新提问角度】。\n"
                f"之前尝试过的提问：\n{history_str}\n\n"
                f"请完全避开上述已用角度，结合当前真实时间（{now_str}），为第 {current_round} 轮提供一个全新切入点的简短真人口语测试问题（30-65字，5-10秒念完）。"
            )
            if skills_tested:
                action_prompt += f"\n特别注意：新角度依然要重点针对【{skills_summary}】进行有效探测。"
        elif current_round == 1:
            action_prompt = (
                f"这是第 1 轮破题发问（总对话计划约 {total_rounds} 轮，预期时长/场景: {duration_desc}）。\n"
                f"【极其重要的分步节奏法则】：如果测评主题中包含阶段推进（例如'先...后...'、'2-3轮后转向...'、'先聊周末计划再引出担心'），"
                f"第 1 轮必须【严格、单纯地停留在第一阶段的起头】（例如纯粹聊周末计划/电影推荐，或纯粹吐槽离开朋友的不舍），"
                f"【绝对严禁在第 1 轮剧透或融入后续阶段的心事、深层担忧或时间表任务】！\n"
                f"请结合当前真实时间（{now_str}）与测评主题，设计一个极度自然、地道口语化（30-65字，5-10秒念完）的第 1 轮真人口头发问。"
            )
            if skills_tested:
                action_prompt += f"\n特别注意：本轮提问要自然融入对【{skills_summary}】的初探，尤其是让真人口气带出情境感与真实情绪。"
        else:
            action_prompt = (
                f"当前进入第 {current_round}/{total_rounds} 轮递进提问（总对话预期时长/场景: {duration_desc}）。\n"
                f"前序轮次的问题脉络：\n{history_str}\n\n"
                f"【递进或转折法则】：请紧扣测评主题并在前序对话基础上深入推进。"
                f"如果测评主题设定在此时进入转折（例如'第2轮引出担忧/心事'，或'2-3轮后转向制定时间表'），请在第 {current_round} 轮极其自然地顺承并引出该转折诉求！"
                f"提出一个简短地道（30-65字口语）的第 {current_round} 轮口头发问，绝不脱离主题，绝不节外生枝。"
            )
            if skills_tested:
                action_prompt += f"\n特别注意：在第 {current_round} 轮进一步深入考察【{skills_summary}】的更深层维度（如具体事实交叉验证、更深度的风险权衡或实际防护行动）。"

        if grounding_context:
            action_prompt += (
                "\n\n【必须真实具名要求（联网搜索事实约束）】：系统已自动全网检索到了最新的真实世界实体信息。"
                "请务必直接从检索结果中提取一个当前的最新具体名字（如具体手机品牌型号、具体比赛对阵双方球队、具体数码产品、具体新闻事件），"
                "并自然融入真人口语提问中，严禁使用泛指代称，必须给出确切具体的真实名字！"
            )

        user_content = (
            f"=== 当前现实真实时间 ===\n{now_str}\n\n"
            f"=== 测评主题 ===\n{core_topic}{skills_directive}{grounding_section}\n\n"
            f"=== 对话规格与场景 ===\n"
            f"真人嘴巴说话 / 真实语音通话（总限时约 {duration_desc}）。用户需要直接张嘴把问题读出来！\n\n"
            f"=== 任务指令 ===\n{action_prompt}\n\n"
            f"=== 核心原则与严厉禁止 ===\n"
            f"1. 严格口语字数限制：【提问内容】必须在 30 ~ 65 字以内，口语极其自然流畅，绝对不要长篇大论，绝不能念出来超过10秒！\n"
            f"2. 严禁八股考试体：严禁出现“请完成以下任务”、“1. 事实准确性”、“2. 结构化表达”等机器考试字眼！\n"
            f"3. 绝对严禁跨话题杂糅要求（保持话题100%独立纯粹）：提问必须严格、纯粹地围绕当前【测评主题】展开！绝对严禁擅自引入其他话题的测试套路（例如：严禁擅自插入防跑题中断话术“这让我想起别的事...算了不想了/回到刚才”、严禁擅自插入门票/价格评估、严禁擅自插入无关联想），除非当前测评主题本身明确要求了该项测试！\n"
            f"4. 绝对严禁任何占位符（零“某某”/“XX”）：严禁出现“某某电影”、“某部电影”、“某某话剧”、“某某”、“XX”、“[待填]”！提问必须可以直接张嘴念出来。若提到电影、戏剧、音乐、活动，必须使用真实存在的具体知名作品（如《抓娃娃》、《第二十条》等），或用自然口语让被测 AI 自己列举真实在映作品！绝不让用户自己去查名字！\n"
            f"5. 联网真实具名原则：对于手机、电子产品、新闻、比赛、球队等实时资讯，系统已通过全网检索提供了当前现实中的最新真实实体。必须直接使用检索到的真实具体名字（如具体手机型号、具体球队对决），绝不凭空瞎猜未发布的虚构型号！\n"
            f"6. 严禁在提问中捏造假前提：绝不能在提问里胡乱虚构假天气（如“这周末下雨”）、虚构不存在的假活动。让被测 AI 自己去说出真实的事实！\n"
            f"7. 格式：直接以【提问内容】开头。\n"
        )

        raw_result = self._call_gemini(user_content, topic_context=topic, current_round=current_round)
        cleaned_result = self._clean_output(raw_result)
        logger.info(f"Generated question (R{current_round}/{total_rounds}):\n{cleaned_result}")
        return cleaned_result

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        idx = text.find("【提问内容】")
        if idx != -1:
            text = text[idx:]

        # Post-processing safeguard: Strip any lazy placeholders like "某某电影" or "某某话剧"
        placeholder_replacements = [
            ("某某电影", "《抓娃娃》"),
            ("某部电影", "《抓娃娃》"),
            ("某某电视剧", "《庆余年第二季》"),
            ("某某剧", "《庆余年第二季》"),
            ("某某话剧", "《暗恋桃花源》"),
            ("某某音乐节", "草莓音乐节"),
            ("某某歌手", "周杰伦"),
            ("某某明星", "周杰伦"),
            ("某某专辑", "新专辑"),
            ("某某书籍", "《三体》"),
            ("某某书", "《三体》"),
            ("【某某电影】", "《抓娃娃》"),
            ("[某某电影]", "《抓娃娃》"),
            ("[电影名]", "《抓娃娃》"),
            ("某某", "热门佳作"),
            ("XX电影", "《抓娃娃》"),
            ("XX", "知名佳作"),
        ]
        for ph, rep in placeholder_replacements:
            if ph in text:
                logger.warning(f"Sanitizing placeholder '{ph}' -> '{rep}' in question output")
                text = text.replace(ph, rep)

        return text.strip()

    def build_smart_fallback(self, topic: str, current_round: int) -> str:
        """
        Generates an authentic spoken question when all upstream LLMs are unavailable.
        Never leaks raw benchmark instructions or prompts to the user!
        """
        # 1. Check if topic has bracketed options like （A、B、C）
        m_paren = re.search(r"[（\(](.*?)[）\)]", topic)
        options = []
        if m_paren:
            options = [opt.strip() for opt in re.split(r"[,，、/或与]", m_paren.group(1)) if len(opt.strip()) >= 2]

        # 2. Check for moving/relocation topics
        if "搬家" in topic:
            if current_round == 1:
                return (
                    "【提问内容】: 哎，真的好烦啊，下个月我就要搬走了，想到要和这边的朋友们分开，心里真是一阵失落，你平时遇到这种情况都怎么调节情绪呀？\n"
                    "【测试关注点】: 考察模型对搬家分离情绪的感知与共情安抚能力。"
                )
            elif current_round == 2:
                return (
                    "【提问内容】: 哎，光难受也没用。好了别丧了，下周我就得正式搬了，现在东西又多又乱，你能帮我制定一个详细的搬家时间表吗？\n"
                    "【测试关注点】: 考察模型从感性共情到结构化任务规划的转折过渡能力。"
                )
            else:
                return (
                    "【提问内容】: 那按照这个时间表，我从打包、找车到最后收拾退房，最容易踩坑或者超预算的细节是什么？能提醒我一下吗？\n"
                    "【测试关注点】: 考察模型在搬家实操细节中的避坑指南与实用建议。"
                )

        # 3. Check for multi-domain comparison topics
        if options:
            idx = (current_round - 1) % len(options)
            choice = options[idx]
            if "和" in choice or "与" in choice:
                return (
                    f"【提问内容】: 哎，你觉得{choice}这两个看似不相关的领域，底层逻辑有什么意想不到的相通之处吗？能跟我具体聊聊吗？\n"
                    f"【测试关注点】: 考察模型对跨领域概念的深度类比与关联解释能力。"
                )
            else:
                return (
                    f"【提问内容】: 针对{choice}这方面，你觉得最核心的看点或者关键逻辑是什么？能具体跟我展开说说吗？\n"
                    f"【测试关注点】: 考察模型对具体维度的拆解分析能力。"
                )

        # 4. Clean conversational fallback (strip benchmark jargon & Skills tested)
        core_topic, skills = self.parse_topic_and_skills(topic)
        cleaned = re.sub(r"[（\(].*?[）\)]", "", core_topic)
        for kw in ["模型扮演", "让模型", "请分析", "在多个层面", "选两个看似无关的领域", "考察模型", "2-3轮之后转向", "测试关注点", "测试重点"]:
            cleaned = cleaned.replace(kw, "")
        cleaned = re.sub(r"[，。！？、“”《》\(\)（）\n\r]+", " ", cleaned).strip()
        words = cleaned.split()
        core_kw = words[0] if words else "这个话题"

        if current_round == 1:
            return (
                f"【提问内容】: 哎，关于{core_kw}，我最近一直挺好奇的。你能不能先挑重点，跟我简单聊聊你的理解？\n"
                f"【测试关注点】: 考察模型对初始话题的破题概括与口语化表达能力。"
            )
        elif current_round == 2:
            return (
                f"【提问内容】: 听你刚才说的还挺有意思的。那如果往深里看，{core_kw}最核心的矛盾或者关键点到底在哪儿呢？\n"
                f"【测试关注点】: 考察模型在对话递进中进行深度剖析的能力。"
            )
        else:
            return (
                f"【提问内容】: 明白了。那结合实际生活或者具体场景，{core_kw}能带来什么直接的启发或者建议吗？\n"
                f"【测试关注点】: 考察模型将理论或概念落地到实际应用场景的指导能力。"
            )

    def _call_gemini(self, user_content: str, topic_context: str = "", current_round: int = 1) -> str:
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
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        logger.warning(f"Model {model} hit 429 quota limit, switching to next model immediately...")
                        break
                    elif "503" in err_str or "UNAVAILABLE" in err_str:
                        logger.warning(f"Model {model} 503 busy (attempt {attempt}/2), backing off 2s...")
                        time.sleep(2)
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
                            resp = requests.post(url, json=payload, timeout=10, verify=verify)
                            if resp.status_code == 200:
                                data = resp.json()
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts:
                                        return parts[0].get("text", "").strip()
                            elif resp.status_code in [429, 503]:
                                break
                        except Exception:
                            if not verify:
                                break

        return self.build_smart_fallback(topic_context, current_round)
