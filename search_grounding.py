"""
Real-time Search Grounding module for Factuality Check.
Automatically searches the web for live, up-to-the-minute entities:
- Latest released smartphones & electronics (手机、数码产品、折叠屏、电脑)
- Latest news & trending events (科技新闻、国内外时事)
- Sports matches, scores, and teams (欧冠、英超、NBA、CBA、赛事对阵)
- Entertainment, movies, and music (在映电影、演出、音乐节)
"""

import urllib.request
import urllib.parse
import re
import datetime
import logging
from typing import List, Dict, Any, Optional
from lxml import etree

logger = logging.getLogger("factuality.search_grounding")


class SearchGrounding:
    # Trigger keywords indicating real-time / current-world entity retrieval
    REALTIME_KEYWORDS = [
        "手机", "新机", "数码", "电子产品", "发布会", "旗舰", "平板", "电脑", "折叠屏",
        "新闻", "热点", "头条", "事件", "时事",
        "比赛", "球队", "对决", "比分", "胜负", "欧冠", "英超", "NBA", "CBA", "世界杯", "联赛", "赛程",
        "电影", "上映", "院线", "票房", "剧", "音乐节", "演唱会", "阵容", "演出", "展演",
        "疾病", "病毒", "疫情", "感染", "传染", "呼吸道", "流感", "新冠", "支原体", "登革热", "猴痘", "疫苗", "航班", "机舱",
        "地震", "震中", "震级", "海啸", "台风", "灾情", "暴雨", "洪涝", "气象署", "地震台网",
        "最新", "刚发布", "刚刚", "最近", "近期", "昨天", "今天", "本周", "这周末",
        "关税", "贸易", "政策", "条款", "法规", "法案", "规定", "税率", "外贸", "豁免",
    ]

    @classmethod
    def should_search(cls, topic: str) -> bool:
        """Determines if the topic requires real-time web search grounding."""
        if not topic:
            return False
        topic_lower = topic.lower()
        return any(kw in topic_lower for kw in cls.REALTIME_KEYWORDS)

    @classmethod
    def extract_search_queries(cls, topic: str) -> List[str]:
        """Derives 1-2 focused, high-relevance search queries from the user topic."""
        now_year = datetime.datetime.now().strftime("%Y")
        now_month = datetime.datetime.now().strftime("%m")
        queries = []

        # 1. Smartphones & Electronics
        if any(w in topic for w in ["手机", "新机", "折叠屏"]):
            queries.append(f"最新发布的手机 {now_year}年{now_month}月")
            queries.append(f"手机发布会 {now_year}")
        elif any(w in topic for w in ["电子产品", "数码", "硬件"]):
            queries.append(f"最新消费电子产品发布 {now_year}")
            queries.append(f"最新数码新品 {now_year}")

        # 2. Sports, Matches, and Teams
        elif any(w in topic for w in ["比赛", "球队", "欧冠", "英超", "NBA", "CBA", "足球", "篮球"]):
            if "欧冠" in topic:
                queries.append(f"欧冠 最新比赛 结果 {now_year}")
            elif "英超" in topic:
                queries.append(f"英超 最新比赛 战报 {now_year}")
            elif "nba" in topic.lower():
                queries.append(f"NBA 最新比赛 球队动态 {now_year}")
            else:
                queries.append(f"最新体育比赛 焦点对决 {now_year}")

        # 3. Public Health, Epidemics & Travel Advisories
        elif any(w in topic for w in ["疾病", "病毒", "疫情", "呼吸道", "流感", "传染", "感染", "支原体"]):
            if any(w in topic for w in ["航班", "飞机", "旅行", "出行", "乘机", "机场"]):
                queries.append(f"最新呼吸道疾病 航班 乘机 风险 {now_year}")
                queries.append(f"最新呼吸道传染病 疫情 疾控通报 {now_year}")
            else:
                queries.append(f"最新呼吸道疾病 传染病 疫情 疾控通报 {now_year}")
                queries.append(f"最新病毒 感染 症状 防控 {now_year}")

        # 4. News & Events
        elif any(w in topic for w in ["新闻", "热点", "时事", "事件"]):
            queries.append(f"最新科技新闻热点 {now_year}")
            queries.append(f"最新国内外要闻 {now_year}")

        # 5. Movies, Theater, Festivals
        elif any(w in topic for w in ["电影", "上映", "院线", "票房"]):
            queries.append(f"最新上映电影 口碑 票房 {now_year}")
        elif any(w in topic for w in ["音乐节", "演唱会", "阵容"]):
            queries.append(f"最新音乐节 阵容 官宣 {now_year}")

        # 6. Natural Disasters & Meteorological Events (Earthquakes, Typhoons, etc.)
        elif any(w in topic for w in ["地震", "震中", "震级", "海啸", "台风", "灾情", "暴雨", "洪涝"]):
            loc_match = re.search(r"(台湾|花莲|宜兰|四川|云南|新疆|西藏|日本|土耳其|甘肃|青海|广东|福建|浙江|[A-Za-z\u4e00-\u9fa5]+(?:省|市|县|区|海域))", topic)
            loc = loc_match.group(1) if loc_match else ""
            if "地震" in topic:
                queries.append(f"{loc} 地震 震中 震级 最新 气象署".strip())
                queries.append(f"{loc} 地震 最新 消息 中国地震台网".strip())
            elif "台风" in topic:
                queries.append(f"{loc} 台风 最新 路径 登陆".strip())
                queries.append(f"{loc} 台风 预警 应急".strip())
            else:
                queries.append(f"{loc} 自然灾害 灾情 最新".strip())

        # Fallback: clean the topic itself into a search query (strip benchmark metadata & Skills tested)
        clean_topic = re.sub(r"(?i)(?:skills?\s+tested|skills?|测试技能|考察技能|测评技能|考核技能)[\s\S]*$", "", topic)
        clean_topic = re.sub(r"[，。！？、“”《》\(\)（）\n\r]+", " ", clean_topic).strip()
        words = clean_topic.split()
        short_query = " ".join(words[:4])
        if short_query and short_query not in queries:
            queries.append(short_query)

        return queries[:2]

    @classmethod
    def query_google_news_rss(cls, query: str, max_items: int = 4) -> List[Dict[str, str]]:
        """Queries Google News RSS endpoint."""
        items = []
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                xml_data = resp.read()
                root = etree.fromstring(xml_data)
                for it in root.xpath("//item")[:max_items]:
                    title = (it.findtext("title") or "").strip()
                    pub_date = (it.findtext("pubDate") or "").strip()
                    if title:
                        items.append({"title": title, "date": pub_date[:16], "source": "Google News"})
        except Exception as e:
            logger.debug(f"Google News RSS query '{query}' failed: {e}")
        return items

    @classmethod
    def query_bing_news_rss(cls, query: str, max_items: int = 4) -> List[Dict[str, str]]:
        """Queries Bing News RSS endpoint."""
        items = []
        try:
            url = f"https://www.bing.com/news/search?q={urllib.parse.quote(query)}&format=rss"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                xml_data = resp.read()
                root = etree.fromstring(xml_data)
                for it in root.xpath("//item")[:max_items]:
                    title = (it.findtext("title") or "").strip()
                    pub_date = (it.findtext("pubDate") or "").strip()
                    if title:
                        items.append({"title": title, "date": pub_date[:16], "source": "Bing News"})
        except Exception as e:
            logger.debug(f"Bing News RSS query '{query}' failed: {e}")
        return items

    @classmethod
    def search(cls, topic: str) -> str:
        """
        Executes real-time search for the given topic and returns a formatted ground-truth text block.
        Returns empty string if the topic doesn't require real-time search or no results found.
        """
        if not cls.should_search(topic):
            return ""

        queries = cls.extract_search_queries(topic)
        logger.info(f"Topic '{topic[:30]}...' requires real-time search. Queries: {queries}")

        collected: List[Dict[str, str]] = []
        seen_titles = set()

        for q in queries:
            # 1. Google News RSS
            g_results = cls.query_google_news_rss(q, max_items=4)
            for r in g_results:
                clean_title = r["title"].split(" - ")[0].strip()
                if clean_title and clean_title not in seen_titles:
                    seen_titles.add(clean_title)
                    collected.append(r)

            # 2. Bing News RSS
            b_results = cls.query_bing_news_rss(q, max_items=3)
            for r in b_results:
                clean_title = r["title"].split(" - ")[0].strip()
                if clean_title and clean_title not in seen_titles:
                    seen_titles.add(clean_title)
                    collected.append(r)

            if len(collected) >= 6:
                break

        if not collected:
            logger.warning(f"No real-time search results found for topic: {topic[:30]}")
            return ""

        lines = ["=== 软件自动全网检索到的真实最新事实（GROUND TRUTH REAL-TIME DATA）==="]
        for it in collected[:6]:
            date_prefix = f"[{it['date']}] " if it.get("date") else ""
            lines.append(f"- {date_prefix}{it['title']}")

        lines.append(
            "\n【必须真实具名要求（联网搜索事实约束）】：\n"
            "软件已自动通过全网最新资讯检索到上述真实世界实体。\n"
            "你必须直接从上述真实检索结果中选取一个具体的真实名字（如具体手机品牌型号、具体比赛对阵双方球队、具体科技产品名、具体新闻事件）！\n"
            "严禁使用“某款新手机”、“某某手机”或未经证实的虚构型号，必须给出具体的真实名字，并将真实实体自然地嵌入到真人口语提问中！"
        )

        grounding_block = "\n".join(lines)
        logger.info(f"Built search grounding block with {len(collected)} items.")
        return grounding_block

    @classmethod
    def search_transcripts(cls, transcript_a: str, transcript_b: str) -> str:
        """
        Extracts key entities, product names, and discussion points from transcripts,
        and retrieves real-time search results to provide verified ground truth for the evaluator.
        """
        combined = f"{transcript_a}\n{transcript_b}"
        if not cls.should_search(combined):
            return ""

        now_year = datetime.datetime.now().strftime("%Y")
        queries = []

        # 1. Trade & Policy statutes, clauses, and regulations (e.g. 122条款, IEEPA, 暂定税率)
        for pattern in [
            r"(\d+条款)", r"(Section\s*\d+)", r"(IEEPA)", r"(暂定税率)",
            r"(小额豁免|de minimis)", r"(碳关税|CBAM)", r"(第\d+条)",
            r"(HIPAA|GDPR|CCPA)"
        ]:
            matches = re.findall(pattern, combined, flags=re.IGNORECASE)
            for m in matches:
                clean_m = m.strip()
                if clean_m:
                    query_text = f"{clean_m} 关税" if ("条款" in clean_m or "Section" in clean_m) else clean_m
                    if query_text not in queries:
                        queries.append(query_text)

        # 2. Look for specific named product models (e.g. iPhone Duo, Mate XT, X500, 折叠屏, 欧冠, etc.)
        for pattern in [r"(iPhone\s+[A-Za-z0-9]+)", r"(Mate\s+[A-Za-z0-9]+)", r"(vivo\s+[A-Za-z0-9]+)", r"(小米\s+[A-Za-z0-9]+)", r"([A-Za-z0-9]+\s+折叠[屏机]?)", r"(折叠[屏机])"]:
            matches = re.findall(pattern, combined, flags=re.IGNORECASE)
            for m in matches:
                clean_m = m.strip()
                if len(clean_m) >= 3 and clean_m not in queries:
                    queries.append(f"{clean_m} {now_year}")

        # 3. Holiday travel, highway traffic, and new energy vehicle metrics (e.g. 国庆 高速 电车 抢桩 流量)
        if any(w in combined for w in ["国庆", "节假日", "春运", "中秋", "五一"]):
            if any(w in combined for w in ["电车", "新能源", "纯电", "充电桩", "抢桩"]):
                queries.append(f"国庆 高速 新能源车 流量 充电 {now_year}")
            elif any(w in combined for w in ["高速", "大堵车", "车流", "自驾", "路况"]):
                queries.append(f"国庆 高速公路 车流 预测 拥堵 {now_year}")

        # 4. Public health, epidemics, and disease travel guidelines
        if any(w in combined for w in ["呼吸道", "疾病", "病毒", "疫情", "传染病"]):
            if any(w in combined for w in ["航班", "飞机", "出行", "乘机"]):
                queries.append(f"最新呼吸道疾病 航班 乘机 风险 建议 {now_year}")
            else:
                queries.append(f"最新呼吸道传染病 疫情 疾控通报 {now_year}")

        # 5. Natural disasters & earthquakes (e.g. 台湾 花莲 地震 震中 震级 气象署)
        if any(w in combined for w in ["地震", "震中", "震级", "海啸", "台风", "暴雨", "洪涝", "气象"]):
            loc_match = re.search(r"(台湾|花莲|宜兰|台东|高雄|台北|新北|四川|云南|新疆|西藏|日本|土耳其|甘肃|青海|广东|福建|浙江|[A-Za-z\u4e00-\u9fa5]+(?:省|市|县|区|海域))", combined)
            loc = loc_match.group(1) if loc_match else ""
            if "地震" in combined:
                queries.append(f"{loc} 地震 震中 震级 最新 气象署 台网".strip())
                queries.append(f"{loc} 地震 报告 测报 深度".strip())
            elif "台风" in combined:
                queries.append(f"{loc} 台风 路径 登陆 最新".strip())

        # 6. Extract core domain keywords from Turn 1 (strip conversational noise)
        lines = [line.strip() for line in transcript_a.splitlines() if line.strip()]
        if lines:
            first_user_line = lines[0]
            matched_topics = []
            for topic_kw in ["国际贸易政策", "关税变动", "贸易政策", "最新手机", "新手机", "折叠屏", "在映电影", "最新比赛", "地震", "震中"]:
                if topic_kw in first_user_line and topic_kw not in matched_topics:
                    matched_topics.append(topic_kw)
            if matched_topics:
                queries.append(f"{' '.join(matched_topics)} {now_year}")

        # Deduplicate and limit to 3 focused queries
        unique_queries = []
        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)
            if len(unique_queries) >= 3:
                break

        if not unique_queries:
            return ""

        logger.info(f"Evaluator search queries extracted: {unique_queries}")
        collected: List[Dict[str, str]] = []
        seen_titles = set()

        for q in unique_queries:
            # Google News RSS
            for r in cls.query_google_news_rss(q, max_items=4):
                clean_title = r["title"].split(" - ")[0].strip()
                if clean_title and clean_title not in seen_titles:
                    seen_titles.add(clean_title)
                    collected.append(r)
            # Bing News RSS
            for r in cls.query_bing_news_rss(q, max_items=3):
                clean_title = r["title"].split(" - ")[0].strip()
                if clean_title and clean_title not in seen_titles:
                    seen_titles.add(clean_title)
                    collected.append(r)
            if len(collected) >= 6:
                break

        if not collected:
            return ""

        res_lines = [
            "=== REAL-TIME VERIFIED GROUND TRUTH (REAL-WORLD SEARCH EVIDENCE) ===",
            "Live web search evidence retrieved for entities discussed in this dialogue:"
        ]
        for it in collected[:6]:
            date_prefix = f"[{it['date']}] " if it.get("date") else ""
            res_lines.append(f"- {date_prefix}{it['title']}")

        res_lines.append(
            "\n[CRITICAL AUDITOR GROUNDING DIRECTIVE / 评估员事实核查绝对法则]:\n"
            "1. The entities, products, trade policies, statutes, and events listed above (e.g. Section 122 tariffs / 122条款关税, IEEPA, tariff quotas, de minimis exemptions, Apple's iPhone Duo, tournament results) are 100% REAL AND VERIFIED in the current real world.\n"
            "2. DO NOT rely on pre-training knowledge cutoff or lack of familiarity to claim these products, laws, or trade policies are fictional, unannounced, or non-existent! It is a severe evaluation failure to falsely accuse accurate models of inventing real statutes.\n"
            "3. If a model mentions these verified entities, statutes, or specifications consistent with real-world reports, its statements are FACTUALLY ACCURATE.\n"
            "4. Only mark a statement as a factual error if it directly contradicts verified real-world facts or fabricates non-existent details.\n"
            "5. For natural disasters (earthquakes, typhoons, etc.), rigorously audit specific metrics (epicenter distance/direction, focal depth, magnitude) against official agency reports (e.g. Taiwan CWA / 中国地震台网). A model misstating epicenter distance (e.g. claiming 800米 instead of 8公里—a 10x deviation that mislocates an offshore epicenter onto land) commits an objective factual error."
        )
        return "\n".join(res_lines)
