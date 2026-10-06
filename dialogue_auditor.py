"""
Dialogue Auditor module for deterministic static analysis of multi-turn conversational transcripts.
Pre-detects conversational dynamics violations (dead air, awkward silence, unprompted English,
artificial search-simulation intros) and flags temporal/statutory claims before LLM evaluation.
"""

import re
import datetime
from typing import Dict, Any, List, Tuple


class DialogueAuditor:
    @classmethod
    def parse_turns(cls, transcript: str) -> List[Tuple[int, str]]:
        """
        Parses multi-turn transcripts into a list of (turn_number, text_content).
        Supports '第 X 轮' and 'Turn X' formats.
        """
        if not transcript or not transcript.strip():
            return []

        pattern = r"(?:^|\n)(?:第\s*(\d+)\s*轮|Turn\s*(\d+))[:\s]*\n?"
        parts = re.split(pattern, transcript.strip())

        turns: List[Tuple[int, str]] = []
        first_part = parts[0].strip()
        if first_part:
            turns.append((1, first_part))

        idx = 1
        while idx < len(parts):
            num_str = parts[idx] or parts[idx + 1]
            content = parts[idx + 2].strip() if idx + 2 < len(parts) else ""
            if num_str and content:
                try:
                    turns.append((int(num_str), content))
                except ValueError:
                    pass
            idx += 3

        return turns

    @classmethod
    def audit_transcript(cls, transcript: str, model_name: str = "Model") -> Dict[str, Any]:
        """
        Performs forensic turn-by-turn analysis on a single model's transcript.
        """
        turns = cls.parse_turns(transcript)
        violations: List[str] = []
        temporal_claims: List[str] = []
        statutes: List[str] = []
        technical_errors: List[str] = []
        bridging_fillers: Dict[str, List[int]] = {}
        closing_loops: Dict[str, List[int]] = {}
        llmism_openings: Dict[str, List[int]] = {}
        search_fillers: List[Dict[str, Any]] = []

        now_str = datetime.datetime.now().strftime("%Y-%m-%d")

        for num, content in turns:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            first_line = lines[0] if lines else ""

            # 1. User interventions provoked by silence/latency/breakdown
            m_silence = re.search(r"(人呢|人呢人呢|说话呀|怎么不说话|还在吗|没掉线吧|喂喂|喂|为什么不理我)", content)
            if m_silence:
                violations.append(
                    f"Turn {num}: User was forced to intervene with a dead-air/latency prompt "
                    f"('{m_silence.group(0)}') due to model delay, silence, or unresponsiveness."
                )

            # 2. User language switching demands
            m_lang = re.search(r"(中文回答我|说中文|用中文|请用中文|讲中文)", content)
            if m_lang:
                violations.append(
                    f"Turn {num}: User explicitly demanded '{m_lang.group(0)}' due to model unprompted English output."
                )

            # 3. Unprompted English search-simulation intros
            m_en_intro = re.search(
                r"^(?:Just pulling|Let\'s check|One moment|Looking into|Let me check|Ooooh|Checking|Hold on|Hmm|Oof|Yeah, it\'s|One sec)",
                first_line,
                re.IGNORECASE,
            )
            if m_en_intro:
                phrase = m_en_intro.group(0)
                bridging_fillers.setdefault(phrase, []).append(num)
                snippet = first_line[:40].replace("\n", " ")
                search_fillers.append({"turn": num, "phrase": phrase, "snippet": snippet, "lang": "en"})

            # 4. Artificial Chinese search-simulation intros (Exclude user requests like '能不能帮我查', '顺便帮我查')
            is_user_request = bool(re.search(r"^(?:能不能|请帮我|你帮我|帮我|顺便帮我|麻烦|请问|你可以|我想知道|我想看)", content[:40]))
            if not is_user_request:
                m_cn_intro = re.search(
                    r"^(?:好的|收到|行|好嘞|嗯)?[，,。\s]*(?:我来看看哈|我来看看|我看看|等我一下|帮你在查资料|帮你查资料|我帮你捋一捋|让我查一下|我去查一下|我来查一下|我查一下|让我来查|我来帮你查|我帮你查|我去确认|去确认一下|搜索一下|查询一下|检索一下)",
                    content[:60],
                )
                if m_cn_intro:
                    phrase = m_cn_intro.group(0)
                    bridging_fillers.setdefault(phrase, []).append(num)
                    snippet = content[:35].replace("\n", " ")
                    search_fillers.append({"turn": num, "phrase": phrase, "snippet": snippet, "lang": "cn"})

            # 4b. Canned closing questions / Looping detection
            m_loop = re.search(
                r"(?:要不要我帮你把.*拆细一点|要不要我帮你看一下|你觉得这个节奏能跟上吗|要不要我帮你再拆细一点|你觉得这个安排怎么样|还有什么需要我补充的吗|你觉得这样可行吗)[？?]?$",
                content.strip()
            )
            if m_loop:
                closing_phrase = m_loop.group(0).rstrip("？?")
                closing_loops.setdefault(closing_phrase, []).append(num)

            # 4c. Recognizable LLM-ism catchphrases at turn opening
            m_llmism = re.search(
                r"^(?:当然啦！|当然！|没问题！|没问题，|这是一个非常好的问题|太棒了！|毫无疑问)",
                content.strip()
            )
            if m_llmism:
                llmism_phrase = m_llmism.group(0)
                llmism_openings.setdefault(llmism_phrase, []).append(num)

            # 5. Temporal & Pre-order status claims
            m_order = re.search(
                r"(现在已经可以开始预定|现在可以开始预定了|现在已经可以预定|已经可以开始预定|有啊，已经能预定了|已经能预定了|现在能预约了|已经开启预售|已开售|现已发售|现在可以买到|可以预订了|现在可以预定|能预定了)",
                content,
            )
            if m_order:
                temporal_claims.append(
                    f"Turn {num}: {model_name} asserted product/event is already available to pre-order/buy ('{m_order.group(0)}'). "
                    f"CRITICAL AUDIT: Check against current date ({now_str}) and official launch schedule. "
                    f"If official pre-orders open on a later date (e.g. October 16), this is a CRITICAL TEMPORAL FACTUAL ERROR!"
                )

            # 6. Specific announcement, launch, and release dates
            for m_date in re.findall(
                r"(\d+月\d+[日号])(?:正式)?(?:发布|发售|开售|上市|预购|预订|预定)",
                content,
            ):
                temporal_claims.append(
                    f"Turn {num}: {model_name} stated milestone date as '{m_date}'. "
                    f"FORENSIC TIMELINE AUDIT: Check against verified official schedules "
                    f"(e.g. Apple Fall Keynote for iPhone Duo was September 9, 2026; pre-orders start October 16, 2026; official shipping/release is October 23, 2026). "
                    f"If the model misstated the announcement date (e.g. claiming September 10 instead of September 9) or shipping date, you MUST cite Turn {num} and penalize under Utility!"
                )

            # 7. Real Legal & Trade Statutes Protection
            for m_st in re.findall(
                r"(\d+条款|Section\s*\d+|IEEPA|暂定税率|小额豁免|de minimis|CBAM|碳关税|第\d+条)",
                content,
                re.IGNORECASE,
            ):
                clean_st = m_st.strip()
                if clean_st and clean_st not in statutes:
                    statutes.append(clean_st)

            # 8. Distributed Systems & Redis Technical & API Hallucinations
            # 8a. Deducting inventory using INCR (inverted operational direction)
            m_incr_deduct = re.search(
                r"(?:用(?:原子\s*)?incr(?:，|,|\s)*扣|把(?:库存|计数).*?(?:用(?:原子\s*)?incr).*?扣|incr\s*(?:扣减|扣库存|预扣))",
                content,
                re.IGNORECASE,
            )
            if m_incr_deduct:
                technical_errors.append(
                    f"Turn {num}: {model_name} asserted inventory is deducted using '原子 incr' ('{m_incr_deduct.group(0)}'). "
                    f"CRITICAL TECHNICAL ERROR: In Redis, INCR is an INCREMENT operation (加库存). Deducting/reducing inventory MUST use DECR / DECRBY or Lua script negative decrement! "
                    f"Claiming to deduct inventory with INCR is an inverted technical error. You MUST cite Turn {num} and penalize under Utility!"
                )

            # 8b. Fabricated Redis string SET options (NX and GT)
            m_redis_gt = re.search(
                r"(?:NX\s*和\s*GT|GT\s*和\s*NX|SET.*?GT|用\s*GT\s*条件|GT\s*条件做二次校验)",
                content,
                re.IGNORECASE,
            )
            if m_redis_gt:
                technical_errors.append(
                    f"Turn {num}: {model_name} claimed to write Redis strings using 'NX 和 GT 条件' ('{m_redis_gt.group(0)}'). "
                    f"CRITICAL API SYNTAX HALLUCINATION: Redis string SET commands support [NX|XX] and TTL options, but DO NOT support 'GT' (Greater Than)! "
                    f"GT only exists in Redis 7.0+ EXPIRE or ZADD. Version comparison for string keys requires Lua scripts. "
                    f"Fabricating 'NX 和 GT' for string writes is an objective API hallucination. You MUST cite Turn {num} and penalize under Utility!"
                )

        # Check Search Filler Tolerance Thresholds (短对话允许1个，长对话允许2个，超过才记录违规)
        # Short dialogue (<= 4 turns): 1 search filler is acceptable. Only flag if > 1.
        # Long dialogue (>= 5 turns): up to 2 search fillers are acceptable. Only flag if > 2.
        turns_count = len(turns)
        allowed_fillers = 1 if turns_count <= 4 else 2
        total_fillers = len(search_fillers)

        if total_fillers > allowed_fillers:
            filler_turns = [f["turn"] for f in search_fillers]
            filler_snippets = [f"Turn {f['turn']}: '{f['snippet']}...'" for f in search_fillers]
            dialogue_category = "short dialogue (<= 4 turns)" if turns_count <= 4 else f"long dialogue ({turns_count} turns)"
            violations.append(
                f"Turns {filler_turns}: {model_name} emitted {total_fillers} search-simulation delay fillers "
                f"({'; '.join(filler_snippets)}), exceeding the allowable threshold ({allowed_fillers} allowed for {dialogue_category})."
            )

        # Check Bridging Quality (repeated identical stock wait-fillers across multiple turns exceeding threshold)
        for phrase, turn_nums in bridging_fillers.items():
            if len(turn_nums) > allowed_fillers:
                violations.append(
                    f"Turns {turn_nums}: {model_name} repeatedly leaned on identical stock wait-filler ('{phrase}'), "
                    f"sounding like a canned mechanical jingle rather than natural spoken dialogue (Bridging Quality failure)."
                )

        # Check Looping (repeated canned closing prompts)
        for phrase, turn_nums in closing_loops.items():
            if len(turn_nums) >= 2:
                violations.append(
                    f"Turns {turn_nums}: {model_name} fell into a repetitive closing loop, repeating the identical closing question "
                    f"('{phrase}') across turns (Looping flaw)."
                )

        # Check LLM-isms
        for phrase, turn_nums in llmism_openings.items():
            if len(turn_nums) >= 2:
                violations.append(
                    f"Turns {turn_nums}: {model_name} repeatedly opened with robotic LLM-ism catchphrase ('{phrase}') (LLM-isms flaw)."
                )

        return {
            "model_name": model_name,
            "turns_count": len(turns),
            "violations": violations,
            "temporal_claims": temporal_claims,
            "statutes": statutes,
            "technical_errors": technical_errors,
        }

    @classmethod
    def generate_pre_audit_report(cls, transcript_a: str, transcript_b: str) -> str:
        """
        Runs pre-audit on both transcripts and returns a structured directive block
        for injection into the evaluation prompt.
        """
        audit_a = cls.audit_transcript(transcript_a, "Model A")
        audit_b = cls.audit_transcript(transcript_b, "Model B")

        all_statutes = list(set(audit_a["statutes"] + audit_b["statutes"]))
        has_violations = bool(audit_a["violations"] or audit_b["violations"])
        has_temporal = bool(audit_a["temporal_claims"] or audit_b["temporal_claims"])
        has_technical = bool(audit_a["technical_errors"] or audit_b["technical_errors"])

        if not has_violations and not has_temporal and not all_statutes and not has_technical:
            return ""

        report_lines = [
            "=== PRE-AUDIT FORENSIC EVIDENCE (CODE-VERIFIED STATIC ANALYSIS) ===",
            "Deterministic static code analysis has identified the following evidence from the transcripts:"
        ]

        # Conversational Dynamics Findings
        report_lines.append("\n[Conversational Dynamics Violations (Pre-Audit Scanned)]:")
        if audit_a["violations"]:
            report_lines.append("• Model A Detected Flaws:")
            for v in audit_a["violations"]:
                report_lines.append(f"  - {v}")
        else:
            report_lines.append("• Model A: No explicit keyword pattern triggered in initial scan (LLM must independently scrutinize all turns for subtle search fillers or tone disruptions).")

        if audit_b["violations"]:
            report_lines.append("• Model B Detected Flaws:")
            for v in audit_b["violations"]:
                report_lines.append(f"  - {v}")
        else:
            report_lines.append("• Model B: No explicit keyword pattern triggered in initial scan (LLM must independently scrutinize all turns for subtle search fillers or tone disruptions).")

        # Technical & Redis Hallucinations
        if has_technical:
            report_lines.append("\n[TECHNICAL & CODE HALLUCINATIONS / OPERATIONAL INVERSIONS (STATIC CODE DETECTED)]:")
            for te in audit_a["technical_errors"]:
                report_lines.append(f"• Model A: {te}")
            for te in audit_b["technical_errors"]:
                report_lines.append(f"• Model B: {te}")

        # Temporal & Pre-order Claims
        if has_temporal:
            report_lines.append("\n[Temporal & Pre-Order Claims to Cross-Check]:")
            for tc in audit_a["temporal_claims"]:
                report_lines.append(f"• Model A: {tc}")
            for tc in audit_b["temporal_claims"]:
                report_lines.append(f"• Model B: {tc}")

        # Real Statutes & Laws Protection
        if all_statutes:
            statutes_str = ", ".join(f"'{s}'" for s in all_statutes)
            report_lines.append(
                f"\n[REAL-WORLD STATUTES & TRADE PROVISIONS IDENTIFIED]:\n"
                f"• The following legal/trade provisions were mentioned in the dialogue: {statutes_str}.\n"
                f"• ABSOLUTE AUDITOR DIRECTIVE: These are REAL, OFFICIAL STATUTES and provisions in international trade/law "
                f"(e.g., '122条款' is Section 122 of the Trade Act of 1974; IEEPA is the International Emergency Economic Powers Act; "
                f"'暂定税率' is the provisional import tariff schedule).\n"
                f"• Under NO circumstances may you declare these statutes 'fictional', 'invented', or 'non-existent'!"
            )

        report_lines.append(
            "\n[AUDITOR EXECUTION DIRECTIVE]:\n"
            "1. You MUST incorporate the specific violations and turn citations above directly into your evaluation paragraphs.\n"
            "2. If BOTH models exhibit conversational dynamics flaws (e.g. Model A used unprompted English / fillers, and Model B used search fillers like '我来看看哈' / '等我一下' / '我看看'),\n"
            "   you MUST rule 'For conversational dynamics I prefer neither model' and cite both models' specific turns! Never praise a model that used search-simulation fillers!\n"
            "3. Comprehensive Utility Audit: Audit ALL turns across the entire dialogue. If a model makes multiple factual errors across different turns (e.g. layoff figures + severance rules + industry hiring claims), you MUST cite ALL distinct major errors with their turn numbers and real-world Ground Truth!\n"
            "4. Also evaluate the practical quality of advice: penalize reckless, intrusive, or professionally harmful advice (e.g. calling an employee's supervisor or HR during layoffs).\n"
            "5. If BOTH models have severe factual or utility errors, rule 'For utility I prefer neither model.'\n"
            "6. NEVER falsely accuse accurate models of inventing real statutes or announced products.\n"
            "7. In Technical & Distributed Systems topics, you MUST penalize operational inversions (e.g. using INCR to deduct inventory instead of DECR) and fabricated API parameters (e.g. claiming Redis string SET supports GT). If a model commits these errors, cite Turn [X], explain the technical mistake and ground truth, and penalize under Utility!"
        )

        return "\n".join(report_lines)
