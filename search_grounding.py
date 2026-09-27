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
        "最新", "刚发布", "刚刚", "最近", "近期", "昨天", "今天", "本周", "这周末",
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

        # 3. News & Events
        elif any(w in topic for w in ["新闻", "热点", "时事", "事件"]):
            queries.append(f"最新科技新闻热点 {now_year}")
            queries.append(f"最新国内外要闻 {now_year}")

        # 4. Movies, Theater, Festivals
        elif any(w in topic for w in ["电影", "上映", "院线", "票房"]):
            queries.append(f"最新上映电影 口碑 票房 {now_year}")
        elif any(w in topic for w in ["音乐节", "演唱会", "阵容"]):
            queries.append(f"最新音乐节 阵容 官宣 {now_year}")

        # Fallback: clean the topic itself into a search query
        clean_topic = re.sub(r"[，。！？、“”《》\(\)（）\n\r]+", " ", topic).strip()
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
