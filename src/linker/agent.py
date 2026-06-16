import os
import json
import re
import logging
import ast
from typing import List, Dict, Any, Optional
from src.core.interfaces import ISchemaLinker, ILLMClient

logger = logging.getLogger(__name__)

class SIPSchemaLinker(ISchemaLinker):
    """
    Schema Linking Agent that uses SIP (Schema Inheritance Protocol) compressed prompts.
    """
    def __init__(self, llm_client: ILLMClient, config: Any):
        self.llm = llm_client
        self.config = config
        
        # Resolve template directory based on strategy
        strategy = getattr(config, "strategy", "inheritance")
        version = getattr(config, "version", "v0")
        enable_thinking = bool(getattr(config, "enable_thinking", False))
        templates_root = os.path.join(os.path.dirname(__file__), "templates")

        versioned_dir = os.path.join(templates_root, version, strategy)
        versioned_think_dir = os.path.join(templates_root, version, "think", strategy)
        legacy_dir = os.path.join(templates_root, strategy)
        legacy_think_dir = os.path.join(templates_root, "think", strategy)

        if enable_thinking and os.path.exists(versioned_think_dir):
            self.template_dir = versioned_think_dir
        elif enable_thinking and os.path.exists(legacy_think_dir):
            logger.warning(
                f"Thinking template directory not found at {versioned_think_dir}. "
                f"Falling back to legacy thinking path: {legacy_think_dir}"
            )
            self.template_dir = legacy_think_dir
        elif os.path.exists(versioned_dir):
            if enable_thinking:
                logger.warning(
                    f"Thinking templates not found for version='{version}', strategy='{strategy}'. "
                    f"Falling back to standard templates: {versioned_dir}"
                )
            self.template_dir = versioned_dir
        elif os.path.exists(legacy_dir):
            logger.warning(
                f"Versioned template directory not found at {versioned_dir}. "
                f"Falling back to legacy path: {legacy_dir}"
            )
            self.template_dir = legacy_dir
        else:
            logger.warning(
                f"No template directory found for version='{version}', strategy='{strategy}'. "
                f"Tried: {versioned_dir}, {legacy_dir}. Falling back to templates root: {templates_root}"
            )
            self.template_dir = templates_root

    def _load_prompt_context(self, db_id: str) -> str:
        """Loads the pre-generated SIP text file for the specific DB."""
        # Use config to resolve path
        path = self.config.get_prompt_path(db_id)
        if not os.path.exists(path):
            logger.warning(f"No preprocessed context found for {db_id} at {path}")
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_template(self, template_name: str) -> str:
        path = os.path.join(self.template_dir, template_name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Template not found: {path}")
            return ""

    def _load_external_knowledge(self, external_knowledge: Optional[str]) -> str:
        if not bool(getattr(self.config, "enable_external_knowledge", False)):
            return ""

        if not external_knowledge:
            return ""

        ref = str(external_knowledge).strip()
        if not ref:
            return ""

        candidates: List[str] = []
        if os.path.isabs(ref):
            candidates.append(ref)
        else:
            root = str(getattr(self.config, "external_knowledge_root", "") or "").strip()
            if root:
                candidates.append(os.path.join(root, ref))
                if not ref.lower().endswith(".md"):
                    candidates.append(os.path.join(root, f"{ref}.md"))
            candidates.append(ref)

        existing_path = next((path for path in candidates if os.path.exists(path)), None)
        if not existing_path:
            logger.warning(f"External knowledge file not found: {ref}")
            return ""

        try:
            with open(existing_path, "r", encoding="utf-8") as f:
                raw_content = f.read().strip()

            content = raw_content
            if existing_path.lower().endswith(".json"):
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, dict) and isinstance(parsed.get("evidence"), str):
                        content = parsed.get("evidence", "").strip()
                except Exception:
                    content = raw_content

            max_chars = int(getattr(self.config, "external_knowledge_max_chars", 4000) or 0)
            if max_chars > 0 and len(content) > max_chars:
                content = content[:max_chars].rstrip() + "\n...[TRUNCATED]"
            return content
        except Exception as exc:
            logger.warning(f"Failed to load external knowledge '{existing_path}': {exc}")
            return ""

    @staticmethod
    def _normalize_external_knowledge_text(knowledge_text: str) -> str:
        normalized = (knowledge_text or "").strip()
        if normalized:
            return normalized
        return "NONE (No external knowledge provided for this question.)"

    @staticmethod
    def _build_external_knowledge_block(knowledge_text: str) -> str:
        return (
            "[EXTERNAL_KNOWLEDGE]\n"
            "-----------------------\n"
            f"{knowledge_text}\n"
            "-----------------------\n"
        )

    def link_schema_details(
        self,
        db_id: str,
        question: str,
        strategy: str = "inheritance",
        external_knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Strategy override is deprecated in favor of config, but keeping arg for interface compat
        save_thought = bool(getattr(self.config, "save_thought", False))
        call_seed = getattr(self.config, "seed", None)
        context = self._load_prompt_context(db_id)
        if not context:
            logger.warning(f"Skipping linking for {db_id} due to missing context.")
            return {"columns": [], "token_cost": 0, "thought": "", "seed": call_seed}
            
        system_msg_tmpl = self._load_template("system_prompt.txt")
        user_msg_tmpl = self._load_template("user_prompt.txt")
        
        if not system_msg_tmpl or not user_msg_tmpl:
               return {"columns": [], "token_cost": 0, "thought": "", "seed": call_seed}

        external_knowledge_text = self._load_external_knowledge(external_knowledge)
        external_knowledge_text = self._normalize_external_knowledge_text(external_knowledge_text)
        external_knowledge_block = self._build_external_knowledge_block(external_knowledge_text)

        user_msg = user_msg_tmpl.format(
            context=context,
            question=question,
            external_knowledge=external_knowledge_text,
            external_knowledge_block=external_knowledge_block,
        )

        messages = [
            {"role": "system", "content": system_msg_tmpl},
            {"role": "user", "content": user_msg}
        ]
        
        # Retry Loop
        max_retries = getattr(self.config, "max_retries", 3)
        effective_attempts = max(1, max_retries + 1)
        last_token_cost = 0
        
        for attempt in range(1, effective_attempts + 1):
            # Invoke LLM
            # Unpack response and token count
            response, token_count = self.llm.generate(messages, temperature=0.0, seed=call_seed)
            last_token_cost = token_count if token_count > 0 else last_token_cost
            thought = self._extract_thought_block(response) if save_thought else ""
            
            if not response:
                logger.warning(f"Attempt {attempt}/{effective_attempts} failed: Empty response from LLM.")
                continue
            
            # Parse Result
            columns = self._parse_json_list(response)
            
            if columns is not None:
                return {
                    "columns": columns,
                    "token_cost": token_count,  # Return the token cost of the successful generation
                    "thought": thought,
                    "seed": call_seed,
                }
                
            logger.warning(f"Attempt {attempt}/{effective_attempts} failed: Could not parse JSON list from response.")
        
        logger.error(f"All {effective_attempts} attempts failed for {db_id}.")
        return {"columns": [], "token_cost": last_token_cost, "thought": "", "seed": call_seed}
        
    def _parse_json_list(self, text: str) -> Optional[List[str]]:
        """Robust JSON parser for lists. Returns None if parsing fails."""
        if not text: 
            return None

        text = self._strip_think_content(text)
        
        # 0) Prefer explicit FINAL_JSON block when model outputs CoT + final answer.
        final_json = self._extract_final_json_block(text)
        if final_json is not None:
            return final_json

        # Clean Markdown code blocks
        clean_text = re.sub(r'```(?:json)?', '', text).strip()
        
        # Attempt 1: Greedy match for JSON list
        try:
            result = json.loads(clean_text)
            if isinstance(result, list):
                return [str(r) for r in result]
        except json.JSONDecodeError:
            pass
            
        # Extract potential list substring
        match = re.search(r'\[.*\]', clean_text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return [str(r) for r in result]
            except json.JSONDecodeError:
                # Attempt 2: Trailing comma fix
                try:
                    fixed_text = re.sub(r',\s*\]', ']', candidate)
                    result = json.loads(fixed_text)
                    if isinstance(result, list):
                        return [str(r) for r in result]
                except json.JSONDecodeError:
                    pass
                
                # Attempt 3: AST literal eval (Python list style)
                try:
                    result = ast.literal_eval(candidate)
                    if isinstance(result, list):
                        return [str(r) for r in result]
                except (ValueError, SyntaxError):
                    pass

        # Regex Fallback
        logger.warning(f"Strict JSON parsing failed. Attempting regex fallback on: {text}...")
        
        potential_items = re.findall(r'["\']([^"\']+)["\']', clean_text)
        if potential_items:
             filtered = [item for item in potential_items if '.' in item and ' ' not in item]
             if filtered:
                  logger.info(f"Regex fallback recovered {len(filtered)} items.")
                  return filtered

        return None

    def _strip_think_content(self, text: str) -> str:
        """Remove model thinking blocks like <think>...</think> before parsing."""
        stripped = re.sub(r'<think\b[^>]*>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
        stripped = re.sub(r'<think\b[^>]*>[\s\S]*$', '', stripped, flags=re.IGNORECASE)
        return stripped.strip()

    def _extract_final_json_block(self, text: str) -> Optional[List[str]]:
        """Extract JSON list from an explicit FINAL_JSON block."""
        # Accept patterns like:
        # FINAL_JSON:\n[...]
        # FINAL_JSON: [...]
        patterns = [
            r'FINAL_JSON\s*:\s*(\[[\s\S]*?\])(?:\s*$|\n[A-Z_]+\s*:)',
            r'FINAL_JSON\s*:\s*(\[[\s\S]*\])\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            candidate = match.group(1).strip()
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return [str(item) for item in result]
            except json.JSONDecodeError:
                try:
                    fixed = re.sub(r',\s*\]', ']', candidate)
                    result = json.loads(fixed)
                    if isinstance(result, list):
                        return [str(item) for item in result]
                except json.JSONDecodeError:
                    try:
                        result = ast.literal_eval(candidate)
                        if isinstance(result, list):
                            return [str(item) for item in result]
                    except (ValueError, SyntaxError):
                        pass

        return None

    def _extract_thought_block(self, text: str) -> str:
        """Extract THOUGHT content from model response. Return empty string if absent."""
        if not text:
            return ""

        # 1) XML-like wrapper support: <THOUGHT>...</THOUGHT>
        xml_match = re.search(r'<\s*THOUGHT\s*>([\s\S]*?)<\s*/\s*THOUGHT\s*>', text, re.IGNORECASE)
        if xml_match:
            return xml_match.group(1).strip()

        # 2) Block style support:
        # THOUGHT:
        # ...
        # FINAL_JSON:
        block_match = re.search(r'THOUGHT\s*:\s*([\s\S]*?)(?:\n\s*FINAL_JSON\s*:|$)', text, re.IGNORECASE)
        if block_match:
            return block_match.group(1).strip()

        return ""