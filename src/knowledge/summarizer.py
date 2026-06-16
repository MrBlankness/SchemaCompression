import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.core.interfaces import ILLMClient

logger = logging.getLogger(__name__)


class ExternalKnowledgeSummarizer:
    def __init__(self, llm_client: Optional[ILLMClient], config: Any):
        self.llm = llm_client
        self.config = config

    def summarize_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, int]:
        summary_root = str(getattr(self.config, "external_knowledge_summary_root", "")).strip()
        if not summary_root:
            logger.info(
                "External knowledge summarization skipped: external_knowledge_summary_root is empty."
            )
            return {"total": len(dataset), "with_doc": 0, "generated": 0, "reused": 0, "fallback_raw": 0}

        os.makedirs(summary_root, exist_ok=True)
        generated = 0
        reused = 0
        fallback_raw = 0
        with_doc = 0

        force = bool(getattr(self.config, "force_regenerate_external_knowledge_summary", False))
        use_summary = bool(getattr(self.config, "use_external_knowledge_summary", False))
        allow_generate = bool(getattr(self.config, "generate_external_knowledge_summary", False))

        logger.info(
            "Start external knowledge summarization: total=%d, root=%s, allow_generate=%s, use_summary=%s, force=%s",
            len(dataset),
            summary_root,
            allow_generate,
            use_summary,
            force,
        )

        for index, item in enumerate(dataset, start=1):
            item_tag = self._item_tag(item)
            doc_ref = str(item.get("external_knowledge") or "").strip()
            if not doc_ref:
                logger.debug("[%s] skip: no external_knowledge reference", item_tag)
                continue

            with_doc += 1
            source_path = self._resolve_doc_path(doc_ref)
            if not source_path:
                logger.warning("[%s] skip: unresolved external_knowledge path for ref=%s", item_tag, doc_ref)
                continue

            summary_path = self._summary_path_for_item(summary_root, item, source_path)
            logger.debug(
                "[%s] (%d/%d) source=%s summary=%s",
                item_tag,
                index,
                len(dataset),
                source_path,
                summary_path,
            )

            if force or not os.path.exists(summary_path):
                if allow_generate:
                    logger.info("[%s] generating summary (force=%s)", item_tag, force)
                    evidence = self._generate_evidence(item=item, source_path=source_path)
                    if evidence is not None:
                        self._write_summary(summary_path, source_path, item, evidence)
                        generated += 1
                        logger.info("[%s] summary generated: %s", item_tag, summary_path)
                    else:
                        logger.warning("[%s] summary generation returned empty evidence", item_tag)
                else:
                    fallback_raw += 1
                    logger.info(
                        "[%s] generation disabled; fallback to raw doc reference", item_tag
                    )
            else:
                reused += 1
                logger.debug("[%s] summary reused: %s", item_tag, summary_path)

            if use_summary:
                if os.path.exists(summary_path):
                    item["external_knowledge"] = summary_path
                    logger.debug("[%s] external_knowledge switched to summary file", item_tag)
                else:
                    # If summary requested but missing, keep raw reference as fallback.
                    item["external_knowledge"] = source_path
                    logger.debug("[%s] summary missing; keep raw source path", item_tag)

        logger.info(
            "External knowledge summarization completed: total=%d, with_doc=%d, generated=%d, reused=%d, fallback_raw=%d",
            len(dataset),
            with_doc,
            generated,
            reused,
            fallback_raw,
        )

        return {
            "total": len(dataset),
            "with_doc": with_doc,
            "generated": generated,
            "reused": reused,
            "fallback_raw": fallback_raw,
        }

    def _resolve_doc_path(self, doc_ref: str) -> str:
        candidates: List[str] = []
        if os.path.isabs(doc_ref):
            candidates.append(doc_ref)
        else:
            root = str(getattr(self.config, "external_knowledge_root", "") or "").strip()
            if root:
                candidates.append(os.path.join(root, doc_ref))
                if not doc_ref.lower().endswith(".md"):
                    candidates.append(os.path.join(root, f"{doc_ref}.md"))
            candidates.append(doc_ref)

        existing = next((path for path in candidates if os.path.exists(path)), "")
        if not existing:
            logger.warning(f"External knowledge file not found for summarization: {doc_ref}")
        return existing

    def _summary_path_for_item(self, summary_root: str, item: Dict[str, Any], source_path: str) -> str:
        question_id = str(item.get("question_id") or "").strip()
        if question_id:
            base_name = question_id
        else:
            fallback = f"{item.get('db_id', '')}::{item.get('question', '')}::{source_path}"
            base_name = hashlib.md5(fallback.encode("utf-8")).hexdigest()[:16]

        return os.path.join(summary_root, f"{base_name}.json")

    @staticmethod
    def _item_tag(item: Dict[str, Any]) -> str:
        question_id = str(item.get("question_id") or "").strip()
        db_id = str(item.get("db_id") or "").strip()
        if question_id and db_id:
            return f"{question_id}@{db_id}"
        if question_id:
            return question_id
        if db_id:
            return db_id
        return "unknown_item"

    def _generate_evidence(self, item: Dict[str, Any], source_path: str) -> Optional[str]:
        item_tag = self._item_tag(item)
        if self.llm is None:
            logger.warning("[%s] LLM client is unavailable; cannot generate summary", item_tag)
            return None

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                knowledge = f.read()
        except Exception as exc:
            logger.warning("[%s] failed to read source file '%s': %s", item_tag, source_path, exc)
            return None

        question = str(item.get("question") or "").strip()
        if not question:
            logger.warning("[%s] skip generation: empty question", item_tag)
            return None

        logger.debug("[%s] generating evidence from source length=%d", item_tag, len(knowledge))
        prompt = self._build_summary_prompt(knowledge=knowledge, question=question)
        response_text, _ = self.llm.generate(prompt, temperature=0.0)
        evidence = self._extract_evidence(response_text)
        if evidence is None:
            logger.warning("[%s] failed to extract evidence from LLM response", item_tag)
            return None

        max_chars = int(getattr(self.config, "external_knowledge_summary_max_chars", 2000) or 0)
        if max_chars > 0 and len(evidence) > max_chars:
            logger.debug(
                "[%s] evidence truncated from %d to %d chars",
                item_tag,
                len(evidence),
                max_chars,
            )
            evidence = evidence[:max_chars].rstrip() + "\n...[TRUNCATED]"

        logger.debug("[%s] evidence generated, length=%d", item_tag, len(evidence))

        return evidence

    @staticmethod
    def _build_summary_prompt(knowledge: str, question: str) -> List[Dict[str, str]]:
        user = f"""---Knowledge Base---
{knowledge}
---Knowledge Base---
## Requirements
You are now a database documentation analysis expert, assisting with the Text-to-SQL query task.

You will receive a user's natural language question (User Question) and a pre-screened document fragment (Knowledge). **The content of this document is all prepared for this question**, but it still contains some redundant or irrelevant information.
Your tasks are:
- Extract from the document the key information most helpful for generating SQL queries, such as key confusing concepts, formulas, unit, and usage methods of unfamiliar formulas (concepts).
- Analyze and extract ambiguous points in the user's question, then retrieve content from the knowledge base that can resolve these ambiguities.
- Assume that there are some function examples in the document that are helpful for solving the problem; it is sufficient to extract them verbatim.
- If the document contains explicit indications of the tables or columns that need to be used, extract them directly, as this will assist with schema linking.
- Analyze and extract other knowledge required to solve the user's question, such as domain-specific names, numbers, etc., which cannot be inferred by the LLM.
- No need to generate SQL; only extract evidence fragments supporting SQL construction.
- No inference, association, or fabrication allowed; extraction must be strictly based on the original text.
- Answers should be concise, precise, and focused, avoiding whole paragraph duplication.

## User Question
{question}
## Output Format:
```json
{{
  "evidence": "Concise and condensed document content that can directly help construct the SQL statement"
}}
```
"""
        return [
            {"role": "system", "content": "You are a careful extraction assistant."},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _extract_evidence(text: str) -> Optional[str]:
        if not text:
            return None

        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        parsed = None

        try:
            parsed = json.loads(cleaned)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            evidence = parsed.get("evidence")
            if isinstance(evidence, str) and evidence.strip():
                return evidence.strip()

        # fallback: treat entire text as evidence when JSON parse fails
        return cleaned if cleaned else None

    @staticmethod
    def _write_summary(summary_path: str, source_path: str, item: Dict[str, Any], evidence: str) -> None:
        payload = {
            "question_id": item.get("question_id"),
            "question": item.get("question"),
            "db_id": item.get("db_id"),
            "source_document": os.path.basename(source_path),
            "evidence": evidence,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.debug(
            "[%s] summary file written: %s (evidence_len=%d)",
            ExternalKnowledgeSummarizer._item_tag(item),
            summary_path,
            len(evidence),
        )
