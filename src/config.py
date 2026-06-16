import os
import yaml
import argparse
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

@dataclass
class AppConfig:
    mode: str
    input_file: str
    db_root_path: str # Path to raw spider2 databases
    preprocess_root: str
    output_dir: str
    run_id: Optional[str]
    resume: bool
    limit: int
    model_name: str
    api_base: Optional[str]
    api_key: Optional[str]
    log_level: str
    version: str
    save_thought: bool
    strategy: str
    force_preprocess: bool
    preprocess_only: bool
    local_model_path: Optional[str]
    device: Optional[str]
    max_retries: int
    seed: Optional[int]
    enable_thinking: Optional[bool]
    max_new_tokens: int
    enable_column_description: bool
    enable_column_type: bool
    enable_sample_values: bool
    sample_values_max_items: int
    sample_value_max_chars: int
    external_knowledge_root: str
    enable_external_knowledge: bool
    external_knowledge_max_chars: int
    external_knowledge_summary_mode: str
    generate_external_knowledge_summary: bool
    use_external_knowledge_summary: bool
    external_knowledge_summary_only: bool
    force_regenerate_external_knowledge_summary: bool
    external_knowledge_summary_root: str
    external_knowledge_summary_max_chars: int

    @staticmethod
    def derive_external_knowledge_summary_mode(
        generate: bool,
        use: bool,
        force: bool,
    ) -> str:
        if force:
            return "refresh"
        if generate and use:
            return "use"
        if generate and not use:
            return "prepare"
        if use and not generate:
            return "use_existing"
        return "off"

    @staticmethod
    def resolve_external_knowledge_summary_flags(mode: str) -> tuple[bool, bool, bool]:
        normalized = (mode or "off").strip().lower()
        if normalized == "refresh":
            return True, True, True
        if normalized == "use":
            return True, True, False
        if normalized == "prepare":
            return True, False, False
        if normalized == "use_existing":
            return False, True, False
        return False, False, False
    
    def get_prompt_path(self, db_id: str) -> str:
        """
        Resolves the path to the pre-generated prompt file for a given DB and strategy.
        """
        # Centralized path logic helps separate Linker from Preprocessor internals
        return os.path.join(self.preprocess_root, self.strategy, "prompts", f"{db_id}.txt")
    
    @classmethod
    def load_yaml(cls, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def from_args(cls):
        load_dotenv()
        
        # 1. Parse CLI to get potential config file path first
        conf_parser = argparse.ArgumentParser(add_help=False)
        conf_parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to YAML config file")
        args, remaining_argv = conf_parser.parse_known_args()
        
        # 2. Load YAML Defaults
        defaults = cls.load_yaml(args.config)
        
        # 3. Define Main Parser
        parser = argparse.ArgumentParser(description="Schema Linking Agent Main Entry", parents=[conf_parser])
        
        # Helper to get default from YAML or Code or Env
        def get_default(key, fallback):
            return defaults.get(key, fallback)

        parser.add_argument("--mode", type=str, choices=["eval", "inference"], default=get_default("mode", "eval"), help="Mode: detailed evaluation with golden records vs pure inference")

        parser.add_argument("--input_file", type=str, default=get_default("input_file", None), required=not get_default("input_file", None), help="Path to input JSON/JSONL file (e.g., dev.json)")
        parser.add_argument("--db_root_path", type=str, default=get_default("db_root_path", None), required=not get_default("db_root_path", None), help="Root path of raw Spider2 databases")
        parser.add_argument("--preprocess_root", type=str, default=get_default("preprocess_root", "data/preprocess"), help="Root directory for saving/loading preprocessed SIP prompts")
        parser.add_argument("--output_dir", type=str, default=get_default("output_dir", "output"), help="Directory to save results")
        parser.add_argument("--run_id", type=str, default=get_default("run_id", None), help="Reuse a specific timestamp output directory under output_dir/<strategy>/<run_id>")
        parser.add_argument("--resume", action="store_true", default=get_default("resume", False), help="Resume from existing eval_details.jsonl in selected run directory")
        parser.add_argument("--limit", type=int, default=get_default("limit", 0), help="Limit number of items to process for testing")

        parser.add_argument("--model_name", type=str, default=get_default("model_name", os.getenv("LLM_MODEL_NAME", "gpt-4o")), help="LLM Model name or 'local'")
        parser.add_argument("--local_model_path", type=str, default=get_default("local_model_path", None), help="Path to local HF model if model='local'")
        parser.add_argument("--max_retries", type=int, default=get_default("max_retries", 3), help="Maximum number of retries for LLM calls")
        parser.add_argument("--seed", type=int, default=get_default("seed", None), help="Global random seed for each LLM evaluation call")
        parser.add_argument("--enable_thinking", action="store_true", default=get_default("enable_thinking", None), help="Enable reasoning thought output (for supported models)")
        parser.add_argument("--max_new_tokens", type=int, default=get_default("max_new_tokens", 1024), help="Maximum new tokens for generation")
        parser.add_argument("--device", type=str, default=get_default("device", "cuda"), help="Device map for local model (e.g., 'cuda', 'cuda:0', 'auto')")
        parser.add_argument("--api_base", type=str, default=get_default("api_base", os.getenv("OPENAI_BASE_URL")), help="DeepSeek/OpenAI API Base URL")
        parser.add_argument("--api_key", type=str, default=get_default("api_key", os.getenv("OPENAI_API_KEY")), help="API Key for OpenAI-compatible services")

        parser.add_argument("--log_level", type=str, default=get_default("log_level", "INFO"), help="Logging level")
        parser.add_argument("--version", type=str, default=get_default("version", "v0"), help="Prompt template version, used under src/linker/templates/<version>")
        parser.add_argument("--save_thought", action="store_true", default=get_default("save_thought", False), help="Save THOUGHT/CoT content to evaluation outputs")

        parser.add_argument("--strategy", type=str, default=get_default("strategy", "inheritance"), choices=["inheritance", "factorization", "raw"], help="Compression strategy")
        parser.add_argument("--force_preprocess", action="store_true", default=get_default("force_preprocess", False), help="Force regeneration of SIP prompts")
        parser.add_argument("--preprocess_only", action="store_true", default=get_default("preprocess_only", False), help="Only run preprocess stage and exit without schema linking/evaluation")

        parser.add_argument("--enable_column_description", action="store_true", default=get_default("enable_column_description", False), help="Include column descriptions in compressed prompt context")
        parser.add_argument("--enable_column_type", action="store_true", default=get_default("enable_column_type", False), help="Include column types in compressed prompt context")
        parser.add_argument("--enable_sample_values", action="store_true", default=get_default("enable_sample_values", False), help="Include sampled column values in compressed prompt context")
        parser.add_argument("--sample_values_max_items", type=int, default=get_default("sample_values_max_items", 2), help="Maximum number of sampled values per column")
        parser.add_argument("--sample_value_max_chars", type=int, default=get_default("sample_value_max_chars", 120), help="Maximum serialized characters for each sampled value")
        parser.add_argument("--enable_external_knowledge", action="store_true", default=get_default("enable_external_knowledge", False), help="Inject per-question external knowledge document into user prompt")
        parser.add_argument("--external_knowledge_root", type=str, default=get_default("external_knowledge_root", "data/raw/resource/documents"), help="Root directory of external knowledge markdown documents")
        parser.add_argument("--external_knowledge_max_chars", type=int, default=get_default("external_knowledge_max_chars", 4000), help="Maximum characters to keep from external knowledge document")
        parser.add_argument(
            "--external_knowledge_summary_mode",
            type=str,
            choices=["off", "use_existing", "prepare", "use", "refresh"],
            default=get_default("external_knowledge_summary_mode", None),
            help=(
                "Unified summary mode: off=no summary; use_existing=use only existing summaries; "
                "prepare=generate missing summaries but keep using raw docs; "
                "use=generate missing summaries and use summaries in linking; "
                "refresh=regenerate all summaries and use summaries in linking"
            ),
        )
        parser.add_argument("--generate_external_knowledge_summary", action="store_true", default=get_default("generate_external_knowledge_summary", False), help="[Deprecated] Generate per-question summarized external knowledge using LLM")
        parser.add_argument("--use_external_knowledge_summary", action="store_true", default=get_default("use_external_knowledge_summary", False), help="[Deprecated] Use generated summary instead of raw external knowledge document")
        parser.add_argument("--external_knowledge_summary_only", action="store_true", default=get_default("external_knowledge_summary_only", False), help="Only run summary stage and exit")
        parser.add_argument("--force_regenerate_external_knowledge_summary", action="store_true", default=get_default("force_regenerate_external_knowledge_summary", False), help="[Deprecated] Force regenerate summary files even when they already exist")
        parser.add_argument("--external_knowledge_summary_root", type=str, default=get_default("external_knowledge_summary_root", "data/preprocess_2/external_knowledge_summary"), help="Directory to save generated external knowledge summaries")
        parser.add_argument("--external_knowledge_summary_max_chars", type=int, default=get_default("external_knowledge_summary_max_chars", 2000), help="Maximum characters kept for generated evidence summaries")
        
        args = parser.parse_args()

        mode = args.external_knowledge_summary_mode
        if mode is None:
            mode = cls.derive_external_knowledge_summary_mode(
                generate=bool(args.generate_external_knowledge_summary),
                use=bool(args.use_external_knowledge_summary),
                force=bool(args.force_regenerate_external_knowledge_summary),
            )

        normalized_generate, normalized_use, normalized_force = cls.resolve_external_knowledge_summary_flags(mode)
        
        return cls(
            mode=args.mode,
            input_file=args.input_file,
            db_root_path=args.db_root_path,
            preprocess_root=args.preprocess_root,
            output_dir=args.output_dir,
            run_id=args.run_id,
            resume=args.resume,
            limit=args.limit,
            model_name=args.model_name,
            local_model_path=args.local_model_path,
            device=args.device,
            api_base=args.api_base,
            api_key=args.api_key,
            log_level=args.log_level,
            version=args.version,
            save_thought=args.save_thought,
            strategy=args.strategy,
            force_preprocess=args.force_preprocess,
            preprocess_only=args.preprocess_only,
            max_retries=args.max_retries,
            seed=args.seed,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            enable_column_description=args.enable_column_description,
            enable_column_type=args.enable_column_type,
            enable_sample_values=args.enable_sample_values,
            sample_values_max_items=args.sample_values_max_items,
            sample_value_max_chars=args.sample_value_max_chars,
            enable_external_knowledge=args.enable_external_knowledge,
            external_knowledge_root=args.external_knowledge_root,
            external_knowledge_max_chars=args.external_knowledge_max_chars,
            external_knowledge_summary_mode=mode,
            generate_external_knowledge_summary=normalized_generate,
            use_external_knowledge_summary=normalized_use,
            external_knowledge_summary_only=args.external_knowledge_summary_only,
            force_regenerate_external_knowledge_summary=normalized_force,
            external_knowledge_summary_root=args.external_knowledge_summary_root,
            external_knowledge_summary_max_chars=args.external_knowledge_summary_max_chars,
        )
