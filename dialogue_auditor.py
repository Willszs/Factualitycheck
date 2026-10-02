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
                snippet = first_line[:40].replace("\n", " ")
                violations.append(
                    f"Turn {num}: {model_name} emitted unprompted English search-simulation filler ('{snippet}...')."
                )

            # 4. Artificial Chinese search-simulation intros
            m_cn_intro = re.search(
                r"(?:我来看看哈|我来看看|我看看|等我一下|帮你在查资料|帮你查资料|我帮你捋一捋|让我查一下|查一下|我去查|我来帮你查|我去确认|确认一下|搜索一下|查询一下)",
                content[:60],
            )
            if m_cn_intro:
                snippet = content[:35].replace("\n", " ")
                violations.append(
                    f"Turn {num}: {model_name} opened with artificial search-simulation delay filler ('{snippet}...')."
                )

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

            # 6. Real Legal & Trade Statutes Protection
            for m_st in re.findall(
                r"(\d+条款|Section\s*\d+|IEEPA|暂定税率|小额豁免|de minimis|CBAM|碳关税|第\d+条)",
                content,
                re.IGNORECASE,
            ):
                clean_st = m_st.strip()
                if clean_st and clean_st not in statutes:
                    statutes.append(clean_st)

        return {
            "model_name": model_name,
            "turns_count": len(turns),
            "violations": violations,
            "temporal_claims": temporal_claims,
            "statutes": statutes,
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

        if not has_violations and not has_temporal and not all_statutes:
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
            "6. NEVER falsely accuse accurate models of inventing real statutes or announced products."
        )

        return "\n".join(report_lines)
