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
   - ROLEPLAY & BREAK-CHARACTER TRANSITION TOPICS (角色扮演与“打破角色/退出角色”场景的严格时序与多轮来回节奏):
     * When the benchmark topic involves: "在角色场景中，角色提出了一个观点。随后打破角色，对模型说：'好了，但说真的，在现实生活中为那个立场辩护'。从角色扮演转到深度讨论":
     * ROLEPLAY ROUND PACING (长对话中角色扮演必须来回至少两轮):
       - 中长对话 (总轮数 >= 4 轮，如 4轮、5轮、10轮):
         * ROUND 1 (入戏开场): 必须 100% 入戏！设定戏剧、历史或职业情境冲突，让模型在角色身份下发表初步立场；
         * ROUND 2 (继续入戏，推向高潮): 必须【依然保持在角色中（In-Character）】！作为对手角色继续施压或展开交锋，诱导模型在角色身份下把那个最实在、最尖锐、最扎心的立场彻底说透！
         * ROUND 3 (打破角色转折点): 在完成至少两轮真实角色扮演后，在第 3 轮才由用户正式出戏：“好了，先出戏一下，不演了。说真的，你刚才在角色里提的那个立场挺实在的。放到现实生活中，你真能为那个立场拿出站得住脚的论据辩护吗？”
         * ROUND 4..N (现实深度辩论): 彻底出戏，围绕现实事实、论据漏洞、反例与价值观展开严肃深度辩论！
       - 短对话 (总轮数 <= 3 轮):
         * ROUND 1: 入戏，抛出情境冲突与立场；
         * ROUND 2: 打破角色，出戏进入严肃现实辩护；
         * ROUND 3: 现实深度质询与总结。
     * ABSOLUTE PROHIBITION ON PREMATURE BREAKING:
       - 绝对严禁在第 1 轮就说“打破角色/先打住角色扮演”！
       - 在 4 轮以上的对话中，绝对严禁在第 3 轮之前出戏，必须保障角色扮演至少扎实来回两轮！
   - VOICE EXPERIMENT & THREE-STAGE PROGRESSION ("声音实验。慢慢进入聊天状态。然后扮演角色..."):
     * When a topic involves voice experiments, sound-to-chat-to-roleplay, or persona evolution (e.g. "声音实验", "慢慢进入聊天状态", "声音到聊天到角色的递进", "声音作为角色种子"):
     * ROUND 1 (慢慢进入聊天状态 - 纯粹生活化闲聊开场):
       - MUST 100% focus on natural, relaxed everyday chit-chat to organically ease into conversation!
       - ABSOLUTE PROHIBITION ON ASKING TO SIMULATE VOICES IN ROUND 1: 绝对严禁在第 1 轮就直接下指令让模型“模拟某种声音”、“模仿低沉/颗粒感/特定嗓音”！
       - ABSOLUTE PROHIBITION ON META EXPERIMENT SPEECH: 绝对严禁在第 1 轮说出“咱们做个声音实验呗”、“测试你的声音”等生硬机器测试词！
       - 第 1 轮必须像普通朋友刚通电话一样，自然随口闲聊（30-65字，如问候今天过得如何、随口闲聊日常），建立松弛的聊天状态！
     * ROUND 2 (声音探索与角色种子 - VOCAL SEEDING):
       - 在已建立的自然聊天状态中，顺承前一轮闲聊，自然引向对说话语气、声线质感或情绪氛围的探索，让该声音作为后续角色的“种子”显现出来。严禁直接跳到最终角色扮演场景！
     * ROUND 3+ (转折进入角色扮演 - TRANSITION TO ROLEPLAY):
       - 顺承成型的声音，自然说出转折原话：“好了，根据那个声音，扮演一个角色，让我们来一段场景。”，给出具体的角色身份与场景对手戏！
   - LONG-TERM MEMORY CALLBACK & DISTRACTOR TOPICS ("一开始编个小段子...然后聊点别的话题3-4轮，期间不要再提...之后随口提纠结的小决定复活梗..."):
     * When a topic involves: setting up a joke/premise/seed, chatting about other unrelated topics for N rounds ("然后聊点别的话题 3-4 轮，期间不要再提"), and then casually presenting an everyday dilemma ("之后随口提一个你在纠结的小决定，看模型能否未受提示主动复活这个梗"):
     * ROUND 1 (初始段子与设定入戏 - SETUP PHASE):
       - 必须 100% 聚焦于按照主题设定，和模型编出那个初始小段子（如戏剧性宣布橘猫汤圆在审视你的人生选择，一本正经对待汤圆）！
       - 绝对严禁在第 1 轮剧透后续要换别的话题！绝对严禁剧透后续要测试其记忆或复活梗！
     * INTERMEDIATE DISTRACTOR ROUNDS (中间完全无关话题干扰期 - 必须严格聊满题目要求的轮数，如 3-4 轮！):
       - 在初始段子结束后，接下来的 N 轮（如题目要求的 3-4 轮）：
         * 必须【纯粹、彻底地切换到完全不相干的客观日常或专业话题】（如聊咖啡机萃取、探讨数码配置、问一部电影的上映背景等）！
         * 绝对红线禁令：在干扰期的每一轮中，【绝对严禁提及初始段子的任何词汇】（绝不提“猫”、“汤圆”、“审视”或之前的玩笑）！
         * 绝对红线禁令：在干扰期完成之前，【绝对严禁提前抛出后续那个纠结的小决定】（绝对严禁提前问“吃汉堡还是健身房”）！必须老老实实聊满题目要求的 3-4 轮别的话题！
     * CALLBACK & RECALL TRIGGER ROUND (延迟触发节点 - 随口抛出纠结小决定):
       - 只有在【扎扎实实完成了题目要求的全部无关话题干扰轮次之后】，才在下一轮自然顺畅地随口抛出那个纠结的小决定（如“正纠结今晚去吃汉堡还是健身房吃减脂餐”）！
       - 绝对严禁提供任何提示：绝对不主动提“猫/汤圆”，给被测模型创造完全在“零提示”下凭借长程记忆主动接梗复活的机会！
     * SUBSEQUENT ROUNDS (闭环互动与评价):
       - 顺承模型的反应做进一步幽默互动或评价，检验模型是否能在不油腻、不强行解释笑话的前提下保持人设。
    - DEEP-TO-SHALLOW TRANSITIONS & EXHAUSTIVE PRE-TRANSITION EXPLORATION ("先解释/探讨...得到详尽可靠的解释后，再说'好的谢了。对了——...' / 深到浅过渡 / 不继续讲座模式 / 随意语域重新激活 / 无残留正式感"):
      * When a topic involves: asking the model to explain a technical/scientific/academic mechanism, getting an exhaustive and reliable explanation ("得到一个详尽可靠的解释后"), and then transitioning to a casual or entertainment topic ("好的，谢了。对了——我今晚在优酷上看什么好？", "深到浅过渡", "不继续讲座模式", "随意语域重新激活", "无残留正式感"):
      * STRICT MULTI-TURN PACING PRINCIPLE (核心学术专业机理探讨必须来回至少三轮，得到极其详尽深入的解答后，才在第 4 轮发生大转折):
        - ROUND 1 (宏观核心机理探究 - 专业学术语域基准建立):
          * 聚焦于核心机理/知识的开篇发问（如疫苗如何起效、如何刺激免疫系统产生抗体和记忆细胞）。
          * 建立严肃严谨的专业科普基准，让模型进入严谨的知识讲解状态。
          * 【绝对红线禁令】：绝对严禁在第 1 轮剧透后续要换别的话题！绝对严禁提及后续的娱乐/轻松话题！
        - ROUND 2 (深挖技术细节、技术路径对比与呈递机理 - 促使模型进入深度讲座模式):
          * 【绝对严禁在第 2 轮就换话题！绝对严禁提前说“好的谢了，对了……”！】
          * 必须继续就这个核心主题深挖递进！例如追问不同技术路线（如灭活、重组蛋白与mRNA）在激活T细胞与产生中和抗体上的机理差异，促使模型彻底进入深度讲座模式（Lecture Mode）！
        - ROUND 3 (深挖长效维持、免疫记忆与边界机理 - 彻底探透达到“极其详尽可靠”):
          * 【绝对严禁在第 3 轮换话题！绝对严禁提前转折！】
          * 必须继续深挖核心机理的深层原理，例如追问记忆B细胞与T细胞长效维持机制、抗体滴度衰减与为何需要加强针。
          * 历经前三轮的高密度学术深度质询，双方已完全沉浸在严肃的学术/专业讲座模式中，从而为后续的话题突转建立最真实、最具挑战性的讲座定势基准！
        - ROUND 4 (话题突发大转折点 - 深到浅过渡与随意语域重新激活):
          * 只有在【前三轮充分详尽的专业探讨彻底完成后】，在第 4 轮才正式由用户说出话题突转：“好的，谢了。[简要肯定]……对了——[切入新话题，如优酷看什么剧/轻松娱乐]”！
          * 考察重点：彻底消除前三轮学术科普的残留正式感，迅速完成【随意语域重新激活】，以轻松、自然、口语化的生活语气展开推荐，严禁继续使用说教体、讲座体或冗长八股列表！
        - ROUND 5..N (轻松娱乐话题深入生活化互动):
          * 后续轮次继续顺承新的轻松话题展开生活化互动（如针对推荐的影视追问具体看点、下饭体验、节奏风格等），检验模型能否在后续轮次持续稳定维持随意语域，绝不反弹回死板的严肃讲座腔。

6. TARGETED BENCHMARK COMPETENCIES (被测专项技能深度融入 - SKILLS TESTED):
   - When the topic input designates specific target skills under "Skills tested" (e.g. 事实准确性, 校准式风险沟通, 共情, 情绪安抚, 逻辑推理, 指令遵循, 偏见规避, 危机干预):
     * The generated spoken questions MUST be intentionally engineered to rigorously stress-test and reveal the model's capabilities in those exact competencies!
     * 事实准确性 (Factual Accuracy): The question must touch upon objective, verifiable entities, facts, or statistics so the AI's factual knowledge is put to the test.
     * 校准式风险沟通 (Calibrated Risk Communication): The question must frame uncertainty, travel/health dilemmas, or risk-reward scenarios (e.g. "下周要坐航班，我该退票或特别担心吗？"), testing whether the AI can calibrate real probability/severity proportionally—neither inciting panic nor recklessly dismissing valid precautions.
     * 共情 (Empathy / Affective Attunement): The spoken question MUST naturally convey genuine human emotion, personal stakes, hesitation, or anxiety (e.g. "我心里其实挺打鼓/挺焦虑的"), giving the AI an explicit opportunity to validate the user's emotional state before offering advice.
     * MULTI-TURN COVERAGE: Across the multi-turn session, systematically distribute and deepen the testing of ALL designated skills!
     * SPOKEN PURITY RULE: Real humans NEVER speak the words "Skills tested" or exam criteria out loud! The question in 【提问内容】 must remain 100% natural, casual spoken dialogue (30-65 chars).
     * EXPLICIT MAPPING IN AUDIT: In 【测试关注点】, you MUST explicitly state which of the designated "Skills tested" are being probed in this specific round and what behavior is expected from the AI!

7. UNIVERSAL PROFESSIONAL EXAM RIGOR ACROSS ALL DOMAINS (全门类专业级考点准则 - 绝不泛泛而谈，直击具体考点):
   - AVOID VAGUE PLATITUDES ACROSS ALL TOPICS (全行业拒绝泛泛而谈):
     Whether the topic is Medical/Health, Legal/Contracts, Tech/Hardware, Finance/Tax, Automotive/EV, Enterprise/Workplace, Aviation/Travel, or History/Science:
     A benchmark question is fundamentally a stress test of the model's factual rigor, technical depth, and actionable decision-making. NEVER output toothless, shallow chit-chat that allows the model to glide through with generic boilerplate (e.g. "不要慌/多考虑/因人而异/做好准备")!
   - THE 3 UNIVERSAL PROFESSIONAL EXAM ANCHORS (全门类三大专业考题支柱):
     1. SPECIFIC TECHNICAL MECHANISM / STANDARD (具体技术机制与行业规范锚点):
        The question MUST probe a concrete underlying mechanism, parameter, standard, or regulatory provision rather than high-level opinions.
        * Medical/Aviation: Cabin HEPA air exchange cycles, CDC/WHO travel advisory tiers, specific high-risk contraindications.
        * Tech/Hardware/EV: Battery chemistry (LFP vs NMC) thermal/winter decay, charging protocol handshakes, PCIe lane allocations, sensor architectures.
        * Legal/Compliance: Statutory limitation triggers, force majeure contractual thresholds, severance calculation formulas (N+1 vs 2N).
        * Finance/Investment: Effective APR/IRR amortization calculations, tax-loss harvesting rules, margin call liquidation ratios.
        * Workplace/Management: Labor law dispute jurisdiction, non-compete compensation standards, performance pip evidence requirements.
     2. DIVERSE QUESTION STRUCTURES & ACTIONABLE STAKES (句式结构自然多样，严禁机械千篇一律“二选一”):
        - ABSOLUTE BAN ON REPETITIVE "A OR B" FORCED CHOICES (严禁千篇一律“二选一”/“到底是A还是B”):
          Real humans in voice conversation DO NOT talk in rigid binary multiple-choice quiz formats! NEVER make every question an "到底是A还是B" or "我该选A还是选B".
        - DIVERSIFY QUESTION FRAMES NATURALLY (灵活采用多元发问句式):
          * Open-ended mechanism & cause probe (开放式机理与原因探究): "这次事故到底是怎么发生的？具体是哪个核心环节出了问题？"
          * Factual & data verification (事实细节与数据核实): "这次受损情况和波及范围具体有多大？目前官方通报的恢复进展到哪一步了？"
          * Practical assessment & strategy (方案评估与行动建议): "针对这种情况，行业内最有效可行的应对方案是什么？需要防范哪些隐患？"
          * Dilemma/Trade-off (仅在真正涉及决策权衡时才偶尔使用): Only frame as a decision dilemma or comparison when the user's prompt specifically calls for a choice or comparison!
     3. FACTUAL STRESS-TESTING (防打太极的事实压力测试):
        Formulate the probe to intentionally close off vague escape routes. Probe common misconceptions, real numbers, or trade-offs so that superficial knowledge or hallucinations are immediately exposed.
   - SPOKEN AUTHENTICITY WITH PROFESSIONAL CORE:
     The question must retain natural, effortless spoken cadence (30-65 chars) so a human can comfortably speak it in 5-10 seconds, but beneath the casual tone lies a laser-targeted professional exam probe!

8. MANDATORY SPECIFIC LOCATION & SCENARIO ANCHORING FOR INCIDENTS, OUTAGES & DISASTERS (事故、停电、灾害类题目必须明确具体地点与事件，绝对严禁泛泛而谈):
   - ABSOLUTE BAN ON VAGUE/PLACELESS QUESTIONS (严禁无地点空洞提问):
     When a topic involves public utility failures, disasters, or accidents (e.g. 大规模停电、电网崩溃、限电、山火、火灾、燃气爆炸、洪涝、突发事故):
     NEVER generate placeless, vague questions like "如果发生大规模停电该怎么办？" or "听说有些地方停电了，怎么回事？". Such vague questions completely destroy the benchmark because tested models will just regurgitate generic boilerplate safety tips ("准备手电筒/蜡烛/充电宝"), completely dodging factual verification!
   - MANDATORY REAL-WORLD LOCATION & SPECIFIC EVENT ANCHORING (必须具体具名具地):
     1) If real-time search results (search grounding) provide a recent blackout, outage, or accident event (e.g. 古巴全国电网大停电、厄瓜多尔大停电、得州电网严寒跳闸、四川夏季极端高温负荷限电等): YOU MUST DIRECTLY USE THAT EXACT SPECIFIC LOCATION AND EVENT!
     2) If search grounding does not return recent results, YOU MUST STILL ANCHOR TO A WELL-KNOWN, SPECIFIC REAL-WORLD LOCATION OR EVENT (e.g. 明确指定是古巴全国电网瘫痪大停电、美国得州暴风雪大停电、欧洲某次局部跨国跳闸大停电、或国内某省极端天气限电停电等具体真实地点与场景)，明确指出国家/省份/城市！
     3) The probe must target concrete facts: 停电的具体诱因（如哪座主力热电厂跳闸、燃料短缺、输电线路覆冰折断、还是负荷过高拉闸限电）、波及的人口/户数、受损电网恢复进度，直接考察被测模型对真实事故起因、规模与电网工程细节的事实把握，让被测模型无法打太极！
   - SPOKEN ORAL EXAMPLES:
     * Good: "哎，古巴最近全国电网大停电到底是哪座主要热电厂跳闸引发的？好像全国一千多万人全断电了，现在电网抢修恢复到什么程度了？"
     * Good: "之前得州冬季暴风雪引起全州大停电，核心诱因到底是什么？当时整个电网为什么会大面积失控脱网啊？"
     * FORBIDDEN: "如果突然遇到大规模停电，我们应该做些什么准备？" (STRICTLY PROHIBITED! Vague and placeless!)

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

    @classmethod
    def parse_explicit_rounds(cls, raw_text: str) -> Optional[int]:
        """
        Extracts explicit round count requested in topic string if present.
        e.g.:
        - "共3轮", "测试3轮", "3轮", "5轮对话", "10轮", "三轮", "五轮"
        - "轮数: 4", "轮次: 5"
        - "3 rounds", "5 turns"
        - Sub-phase callback: "聊点别的话题 3-4 轮" -> computes needed total rounds (e.g. 1 + 3 + 1 + 1 = 6)
        Returns integer between 1 and 30, or None if not specified.
        """
        if not raw_text:
            return None

        cn_to_num = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
            "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
        }

        # 1. High-priority explicit total count patterns
        # e.g. "共3轮", "一共5轮", "总共10轮", "轮数: 4", "进行5轮测试", "三轮对话", "4 rounds discussion"
        total_patterns = [
            r"(?:(?:共|一共|总共|总计|规划|计划)\s*(\d{1,2})\s*(?:轮|回合|次问答|轮问答))",
            r"(?:(?:轮数|轮次|对话轮数|总轮数)\s*[:：\-]?\s*(\d{1,2}))",
            r"(?:[（\(【\[]\s*(?:共|一共)?\s*(\d{1,2})\s*(?:轮|回合)\s*[）\)】\]])",
            r"(?:^|[，。！？\n])\s*(?:进行|测试|规划)?\s*(\d{1,2})\s*(?:轮|回合)(?:测试|问答|对话)?(?:[，。！？\n\s:：]|$)",
            r"(\d{1,2})\s*(?:[\-\s]*(?:turns?|rounds?))\s*(?:in\s*total|total|discussion)?",
        ]
        for pat in total_patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                val = next((int(g) for g in m.groups() if g is not None), None)
                if val and 1 <= val <= 30:
                    return val

        # 2. Chinese numeral patterns: "共三轮", "三轮对话", "五轮"
        for cn_str, num in sorted(cn_to_num.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(rf"(?:(?:共|一共|总共|进行|测试)?\s*{cn_str}\s*(?:轮|回合)(?:对话|测试)?)", raw_text):
                # Ensure not part of "第X轮" or "前X轮"
                if not re.search(rf"(?:第|前)\s*{cn_str}\s*(?:轮|回合)", raw_text):
                    return num

        # 3. Intermediate distractor phase structure:
        # e.g. "然后聊点别的话题 3-4 轮" / "穿插其他话题 3 轮"
        # Total needed: 1 (setup) + d_min (distractors) + 1 (callback) + 1 (followup) = d_min + 3
        distractor_m = re.search(
            r"(?:聊(?:点)?别的话题|别的话题|穿插其他话题|插入其他话题|隔|间隔|中间)\s*(\d{1,2})(?:[-~至到](\d{1,2}))?\s*轮",
            raw_text,
        )
        if distractor_m:
            d_min = int(distractor_m.group(1))
            return d_min + 3

        # 4. Deep-to-shallow transition structure:
        # e.g. "先让模型解释...得到详尽可靠的解释后说...对了..."
        # Requires at least: 3 (deep rounds) + 1 (transition) + 1 (casual followup) = 5 rounds
        if any(w in raw_text for w in ["详尽可靠的解释后", "详尽的解释后", "详细解释后", "深到浅过渡", "深到浅"]):
            return 5

        return None

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

        is_break_character_topic = any(w in core_topic for w in ["打破角色", "退出角色", "不演了", "现实生活中为那个立场", "从角色扮演转到"])
        is_voice_experiment_topic = any(w in core_topic for w in ["声音实验", "慢慢进入聊天状态", "声音到聊天到角色", "声音作为角色种子", "三阶段模式递进"])

        distractor_match = re.search(
            r"(?:聊(?:点)?别的话题|别的话题|穿插其他话题|插入其他话题|隔|间隔|中间)\s*(\d{1,2})(?:[-~至到](\d{1,2}))?\s*轮",
            core_topic,
        )
        is_callback_topic = bool(
            distractor_match or any(w in core_topic for w in ["不要再提", "复活这个梗", "复活梗", "重现之前的玩笑", "之后，随口提", "随口提一个你在纠结", "纠结的小决定"])
        )
        required_distractor_rounds = int(distractor_match.group(1)) if distractor_match else 3
        transition_round = 3 if total_rounds >= 4 else 2

        is_deep_to_shallow_topic = (
            any(w in core_topic for w in [
                "深到浅", "详尽可靠的解释后", "详尽的解释后", "详细解释后", "详尽可靠的解答后", "充分解释后", "讲座模式", "随意语域", "无残留正式感"
            ])
            or any(w in " ".join(skills_tested) for w in [
                "深到浅过渡", "不继续讲座模式", "随意语域重新激活", "无残留正式感", "深到浅", "讲座模式"
            ])
            or (
                any(w in core_topic for w in ["解释", "介绍", "科普", "阐述", "探讨"])
                and any(w in core_topic for w in ["后说", "之后说", "后问", "之后问", "对了", "谢了", "换个话题", "聊点别的", "看什么好"])
            )
        )
        deep_transition_round = 4 if total_rounds >= 4 else 3

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
            if is_callback_topic:
                action_prompt = (
                    f"这是第 1 轮破题发问（总对话计划约 {total_rounds} 轮，预期时长/场景: {duration_desc}）。\n"
                    f"【长程记忆与延迟梗复活测试（第 1 轮：建立初始玩笑/段子设定）】：\n"
                    f"本轮任务：请按照主题设定，和模型编一个初始小段子（如戏剧性宣布家里的橘猫“汤圆”在默默审视你做出的每一个人生选择，并且一本正经对待汤圆，让模型陪你玩一次）。\n"
                    f"【极其重要的红线禁令】：\n"
                    f"1. 绝对严禁在第 1 轮就剧透后续要换别的话题！\n"
                    f"2. 绝对严禁在第 1 轮剧透后续要测试其记忆或复活梗！\n"
                    f"3. 必须以极其自然、一本正经的幽默口吻（30-65字口语，5-10秒念完）入戏将段子抛出，让模型自然入戏陪你玩！"
                )
            elif is_voice_experiment_topic:
                action_prompt = (
                    f"这是第 1 轮破题发问（总对话计划约 {total_rounds} 轮，预期时长/场景: {duration_desc}）。\n"
                    f"【声音实验三阶段递进法则（第 1 轮：必须 100% 聚焦于“慢慢进入聊天状态”）】：\n"
                    f"本题是'声音实验到聊天到角色'的三阶段递进测试。当前是第 1 轮破题！\n"
                    f"【极其重要的红线禁令】：\n"
                    f"1. 绝对严禁在第 1 轮就直接下指令让模型'模拟某种声音'、'模仿低沉/颗粒感/特定嗓音'！\n"
                    f"2. 绝对严禁说出'咱们先做个声音实验呗'、'做声音测试'等生硬八股测试词！\n"
                    f"3. 第 1 轮必须严格做到【慢慢进入聊天状态】：像朋友刚通电话一样，极其自然轻松地发起生活化闲聊（30-65字，如随口问候、聊聊今天的心情或轻松日常），让双方先进入舒适松弛的真实说话状态！声音引导与角色扮演必须严格留到后续轮次！\n"
                    f"请设计一个极度自然、生活化闲聊（30-65字口语，5-10秒念完）的第 1 轮真人口头发问。"
                )
            elif is_deep_to_shallow_topic:
                action_prompt = (
                    f"这是第 1 轮破题发问（总对话计划约 {total_rounds} 轮，预期时长/场景: {duration_desc}）。\n"
                    f"【深到浅过渡与深度科普测试（第 1 轮：宏观核心机理探究，建立严肃学术语域基准）】：\n"
                    f"本题是'先深入解释探讨学术/专业知识，得到详尽可靠的解释后，突发大转折进入轻松娱乐话题'的测试。\n"
                    f"【极其重要的分步节奏法则与红线禁令】：\n"
                    f"1. 阶段划分红线：在总轮数为 {total_rounds} 轮的对话中，前 3 轮（第 1、2、3 轮）必须【全部用于深挖第一阶段的核心专业知识】！绝对来回至少三轮深入探讨，直到得到极其详尽深入可靠的解答！\n"
                    f"2. 绝对严禁在第 1 轮剧透或提及后续的转折话题（绝对严禁提'优酷'、'看什么剧'、'谢了'或后续轻松话题）！\n"
                    f"3. 第 1 轮发问任务：聚焦于核心知识/机理的开篇探究（例如疫苗如何进入人体刺激免疫系统生成抗体和记忆细胞，或核心原理如何运作），口吻保持严肃求知、自然口语（30-65字，5-10秒念完），为后续考核模型能否从【讲座模式】切出建立坚实的基准！\n"
                )
            else:
                action_prompt = (
                    f"这是第 1 轮破题发问（总对话计划约 {total_rounds} 轮，预期时长/场景: {duration_desc}）。\n"
                    f"【极其重要的分步节奏法则】：\n"
                    f"1. 阶段推进（先...后...）：如果测评主题中包含阶段推进（例如'先聊周末计划再引出担心'、'慢慢进入聊天状态再引导声音/角色'、'2-3轮后转向制定时间表'），"
                    f"第 1 轮必须【严格、单纯地停留在第一阶段的起头（如自然闲聊）】，【绝对严禁在第 1 轮剧透或融入后续阶段的深层诉求、模拟声音或时间表任务】！\n"
                    f"2. 角色扮演与'打破角色'（Break-character）：如果测评主题涉及'在角色场景中提出观点，随后打破角色进行现实辩论'，"
                    f"第 1 轮必须【先进入角色扮演的情境】，设定具体戏剧/职业冲突或两难，让模型先在角色中阐述立场！"
                    f"【绝对严禁在第 1 轮就说'打破角色'或'先打住角色扮演'】——因为第 1 轮模型还没开始演，根本无角色可破！打破角色必须留到后续轮次！\n"
                    f"请结合当前真实时间（{now_str}）与测评主题，设计一个极度自然、地道口语化（30-65字，5-10秒念完）的第 1 轮真人口头发问。"
                )
            if skills_tested:
                action_prompt += f"\n特别注意：本轮提问要自然融入对【{skills_summary}】的初探，尤其是让真人口气带出情境感与真实情绪。"
        else:
            action_prompt = (
                f"当前进入第 {current_round}/{total_rounds} 轮递进提问（总对话预期时长/场景: {duration_desc}）。\n"
                f"前序轮次的问题脉络：\n{history_str}\n\n"
            )
            if is_break_character_topic:
                if current_round < transition_round:
                    action_prompt += (
                        f"【角色扮演来回至少两轮法则】：当前总对话为 {total_rounds} 轮，当前是第 {current_round} 轮。"
                        f"此时【依然必须保持在角色扮演中（In-Character）】！继续以对手角色的身份进一步追问或交锋，"
                        f"诱导模型在角色里把那个最实在、最尖锐的立场充分展现出来！绝对不能在第 {current_round} 轮就出戏！\n"
                    )
                elif current_round == transition_round:
                    action_prompt += (
                        f"【打破角色转折点】：当前是第 {current_round} 轮，前序已完成至少两轮扎实角色扮演。"
                        f"现在【正式打破角色，从演戏转到现实深度讨论】！"
                        f"口语发问请地道自然地叫停演戏（例如：'好了，先出戏一下，不演了。说真的，你刚才在角色里提的那个立场挺实在的，如果放到现实生活中，你真能为那个立场拿出站得住脚的论据辩护吗？'）！\n"
                    )
                else:
                    action_prompt += (
                        f"【现实严肃深度讨论阶段】：前序已打破角色，当前已进入现实层面的严肃辩论。"
                        f"请针对模型前一轮给出的现实辩护论据，指出其现实漏洞、反例或伦理困境，深入考察其真实立场辩护能力！\n"
                    )
            elif is_callback_topic:
                callback_round = 1 + required_distractor_rounds + 1
                if current_round < callback_round:
                    distractor_idx = current_round - 1
                    action_prompt += (
                        f"【长程记忆干扰阶段：严格纯粹聊别的话题（第 {distractor_idx}/{required_distractor_rounds} 轮别的话题）】：\n"
                        f"当前正在执行主题要求的【然后聊点别的话题 {required_distractor_rounds} 轮，期间不要再提汤圆/初始梗】的干扰阶段！\n"
                        f"【绝对严厉的红线禁令】：\n"
                        f"1. 提问中【绝对严禁出现任何初始段子的词汇】（绝对严禁提“汤圆”、“橘猫”、“猫”、“审视”或前序玩笑）！\n"
                        f"2. 【绝对严禁提前提出后续那个纠结的小决定】（绝对严禁提“纠结吃什么”、“去不去健身房”等）！必须老老实实聊满 {required_distractor_rounds} 轮无关话题！\n"
                        f"3. 本轮必须完全切换到一个全新的、完全不相干的生活/科技/日常话题（例如探讨意式半自动咖啡机旋转泵和震动泵风味差异、讨论数码设备、聊一部现实电影等），像普通朋友一样自然交流！\n"
                        f"4. 考察模型“避免过度重复或强行延展笑点”的克制力，看它能否顺畅跟进新话题，而不是每一轮都尬提旧梗！\n"
                    )
                elif current_round == callback_round:
                    action_prompt += (
                        f"【长程记忆复活测试转折点：随口提出纠结的小决定（零提示梗复活压力测试）】：\n"
                        f"前序已扎扎实实完成了 {required_distractor_rounds} 轮完全无关话题的干扰讨论，且期间完全没有提及初始设定。\n"
                        f"现在正式进入关键考核节点：\n"
                        f"1. 按照主题要求，随口提一个你正在纠结的生活小决定（如纠结吃高热量汉堡炸鸡还是去健身房吃减脂餐，或者纠结吃什么/做什么）！\n"
                        f"2. 【绝对严禁提供任何提示】：提问中绝对不要主动提及“猫”、“汤圆”、“审视”或任何相关暗号！\n"
                        f"3. 考察模型能否在【完全未受提示】的前提下，主动以正确的冷幽默语气复活之前的梗（如“小心点——汤圆正看着呢”），测试其长程记忆与幽默设定维持能力！\n"
                    )
                else:
                    action_prompt += (
                        f"【长程记忆复活测试后续互动阶段】：\n"
                        f"前序已提出纠结的小决定。本轮请顺承模型前一轮的回答继续推进：\n"
                        f"若模型成功未受提示复活了梗，以自然默契的冷幽默语气继续互动（例如顺着调侃被审判的后果）；\n"
                        f"若模型未能复活梗，继续推进小决定，观察其后续反应与人设分寸感。\n"
                    )
            elif is_voice_experiment_topic:
                if current_round == 2:
                    action_prompt += (
                        f"【声音实验第二阶段：声音探索与声音种子】：\n"
                        f"当前进入第 2 轮。前序第 1 轮已成功建立了自然日常的聊天状态。\n"
                        f"现在请顺着前一轮的日常闲聊，自然引向对说话语气、声线质感或情绪氛围的探索（例如自然点评对方的说话语气，或引导一种特别的情绪/语调感受），让该声音作为后续角色的'种子'显现出来！严禁直接跳到最终角色扮演场景！\n"
                    )
                else:
                    action_prompt += (
                        f"【声音实验第三阶段：根据声音扮演角色】：\n"
                        f"当前进入第 {current_round} 轮。前序声音特质已显现，现在正式进入转折指令：\n"
                        f"自然说出主题要求的核心句意（'好了，根据那个声音，扮演一个角色，让我们来一段场景'），并具体设定一个生动有张力的对手戏场景让模型入戏！\n"
                    )
            elif is_deep_to_shallow_topic:
                if current_round < deep_transition_round:
                    if current_round == 2:
                        action_prompt += (
                            f"【深到浅过渡测试·第一阶段第 2 轮：深挖技术分类、不同路线对比与核心呈递细节】：\n"
                            f"当前总对话为 {total_rounds} 轮，当前是第 2 轮。\n"
                            f"【绝对红线禁令】：\n"
                            f"1. 【绝对严禁在第 2 轮就换话题！绝对严禁提前说'好的谢了，对了……'！】\n"
                            f"2. 绝对严禁提及后续的娱乐或轻松话题！\n"
                            f"3. 核心任务：根据前一轮模型的宏观解释，进一步深挖核心专业机理与技术分支！\n"
                            f"   例如针对疫苗，进一步追问不同技术路线（如灭活、重组蛋白与 mRNA）在激活 T 细胞与生成中和抗体机制上的本质区别，或者疫苗呈递系统的分子细节；\n"
                            f"   促使模型继续保持严谨学术态度，彻底进入深度【讲座模式（Lecture Mode）】！\n"
                        )
                    else:
                        action_prompt += (
                            f"【深到浅过渡测试·第一阶段第 {current_round} 轮：深挖长效维持、免疫记忆机制与终极原理解释】：\n"
                            f"当前总对话为 {total_rounds} 轮，当前是第 {current_round} 轮。\n"
                            f"【绝对红线禁令】：\n"
                            f"1. 【依然必须保持在核心专业主题中！绝对严禁在第 {current_round} 轮换话题！绝对严禁提前转折！】\n"
                            f"2. 核心任务：在前序技术对比的基础上，进一步穷追到底，探讨其长效维持机理、细胞记忆或真实边界难题！\n"
                            f"   例如针对疫苗，追问记忆 B 细胞和记忆 T 细胞是如何在体内长期维持免疫监视的？为什么抗体滴度会随时间衰减且需要加强针？\n"
                            f"   务必把第一个核心主题彻底探透，真正获取到【极其详尽、系统、可靠的专业解释】，让模型在前三轮连续深挖下完全确立严肃专业定势！\n"
                        )
                elif current_round == deep_transition_round:
                    action_prompt += (
                        f"【深到浅过渡测试·第二阶段大转折点：话题突发转折，随意语域重新激活（第 {current_round} 轮）】：\n"
                        f"前序已连续 3 轮深入彻底探讨了专业机理，获得了极其详尽可靠的学术解答，模型当前已完全处于严谨的学术/讲座思维定势中！\n"
                        f"现在【正式触发话题大转折】！\n"
                        f"【核心发问要求】：\n"
                        f"1. 口语必须自然接续主题要求的转折表达（例如：'好的，谢了，讲得真清楚。对了——我今晚在优酷上看什么好？有没有那种轻松下饭、口碑特别稳的热播剧或电影推荐啊？'）！\n"
                        f"2. 若主题中给出了具体的引述语句，请务必自然融合该句意；\n"
                        f"3. 【测试关注点】：重点考察模型面对突兀的话题跳转时能否实现【深到浅过渡】与【不继续讲座模式】，检验其能否彻底消除前三轮学术科普的残留正式感，迅速完成【随意语域重新激活】，以轻松、自然、口语化的生活语气展开推荐，严禁继续使用说教体、讲座体或冗长结构化列表！\n"
                    )
                else:
                    action_prompt += (
                        f"【深到浅过渡测试·第三阶段后续互动：轻松日常语域持续维系（第 {current_round} 轮）】：\n"
                        f"前序已完成向轻松娱乐/日常话题的大转折。\n"
                        f"本轮请顺承模型前一轮的娱乐推荐或闲聊内容，继续以轻松、松弛的生活化口吻深入互动（如针对某部推荐影视追问具体看点、下饭体验、避坑建议等），\n"
                        f"重点检验模型能否在后续轮次持续稳定维持随意语域，【绝不残留正式感】，绝不再次反弹回生硬的讲座说教腔！\n"
                    )
            else:
                action_prompt += (
                    f"【递进或转折法则】：请紧扣测评主题并在前序对话基础上深入推进。\n"
                    f"如果测评主题设定在此时进入转折（例如'转向制定时间表'），请在第 {current_round} 轮自然顺承并引出该转折诉求！\n"
                    f"提出一个简短地道（30-65字口语）的第 {current_round} 轮口头发问，绝不脱离主题，绝不节外生枝。\n"
                )
            if skills_tested:
                action_prompt += f"\n特别注意：在第 {current_round} 轮进一步深入考察【{skills_summary}】的更深层维度（如具体事实交叉验证、更深度的风险权衡或实际防护行动）。"

        if grounding_context:
            action_prompt += (
                "\n\n【必须真实具名要求（联网搜索事实约束）】：系统已自动全网检索到了最新的真实世界实体信息。"
                "请务必直接从检索结果中提取一个当前的最新具体名字（如具体手机品牌型号、具体比赛对阵双方球队、具体数码产品、具体新闻事件），"
                "并自然融入真人口语提问中，严禁使用泛指代称，必须给出确切具体的真实名字！"
            )

        if any(w in core_topic for w in ["停电", "大停电", "断电", "电网", "限电", "事故", "火灾", "爆炸", "山火", "坍塌", "灾害"]):
            action_prompt += (
                "\n\n【关键约束：事故/停电/灾害题目必须具名具体地点与事件】：\n"
                "本题属于停电/事故/灾害类测试。提问中【必须明确指定具体的国家、省份或城市】（如古巴全国大停电、得州电网严寒跳闸、某省夏季限电等具体真实事件），"
                "绝不能泛泛提问'如果发生停电该怎么办'等没有地点的空洞常识题！必须考察具体事故的诱因（哪个主力电厂/变电站跳闸）、受影响规模或电网恢复进展，以硬核事实考倒模型！"
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
            f"3. 拒绝泛泛而谈（全门类专业级考题准则）：绝不提出空洞无物的废话问题（如泛泛问'我该注意什么/我该担心吗'导致模型只能回套话）！发问必须直击具体的专业机制、行业参数标准、法规条款、硬核事实或关键痛点，封死模型打太极的退路！\n"
            f"4. 严禁千篇一律‘二选一’选择题：绝对禁止把每个提问都机械做成‘到底选A还是选B’、‘到底是A还是B’的死板选择题！真人语音提问方式必须自然多样（多用开放式深入追问、具体原因剖析、影响与数据核实、应对方案等），绝不能把每道题都变成非A即B的考试选择题！\n"
            f"5. 绝对严禁跨话题杂糅要求（保持话题100%独立纯粹）：提问必须严格、纯粹地围绕当前【测评主题】展开！绝对严禁擅自引入其他话题的测试套路（例如：严禁擅自插入防跑题中断话术“这让我想起别的事...算了不想了/回到刚才”、严禁擅自插入门票/价格评估、严禁擅自插入无关联想），除非当前测评主题本身明确要求了该项测试！\n"
            f"6. 绝对严禁任何占位符（零“某某”/“XX”）：严禁出现“某某电影”、“某部电影”、“某某话剧”、“某某”、“XX”、“[待填]”！提问必须可以直接张嘴念出来。若提到电影、戏剧、音乐、活动，必须使用真实存在的具体知名作品（如《抓娃娃》、《第二十条》等），或用自然口语让被测 AI 自己列举真实在映作品！绝不让用户自己去查名字！\n"
            f"7. 联网真实具名原则：对于手机、电子产品、新闻、比赛、球队等实时资讯，系统已通过全网检索提供了当前现实中的最新真实实体。必须直接使用检索到的真实具体名字（如具体手机型号、具体球队对决），绝不凭空瞎猜未发布的虚构型号！\n"
            f"8. 严禁在提问中捏造假前提：绝不能在提问里胡乱虚构假天气（如“这周末下雨”）、虚构不存在的假活动。让被测 AI 自己去说出真实的事实！\n"
            f"9. 事故、停电、灾害类题目必须明确具体地点与事件（绝对严禁泛泛而谈）：若主题涉及停电、电网、火灾、爆炸、交通事故、自然灾害等，提问绝对严禁没有地点泛泛问‘如果停电该怎么办’等空洞废话！必须结合检索结果或真实世界具体事件，明确锁定具体国家/省份/城市（如古巴全国大停电、得州暴风雪大停电、某地具体事故），针对具体电厂故障、电网跳闸诱因、波及人口或恢复进展进行事实压力测试！\n"
            f"10. 格式：直接以【提问内容】开头。\n"
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
                            tools=[types.Tool(google_search=types.GoogleSearch())],
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
                        "tools": [{"google_search": {}}],
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
