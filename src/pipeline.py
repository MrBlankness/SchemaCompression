import os
import sys
import json
import logging
import random
from datetime import datetime
from typing import Set, List, Dict, Any
from tqdm import tqdm

from src.config import AppConfig
from src.data_loader import Spider2Loader
from src.linker import SIPSchemaLinker
from src.llm_service import OpenAIClient, LocalLLMClient
from src.evaluation import EvaluationRunner
from src.preprocessor import PreprocessorFactory
from src.knowledge import ExternalKnowledgeSummarizer
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Orchestrates the entire Schema Linking pipeline:
    Setup -> Data Loading -> Preprocessing Check -> Execution -> Reporting
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm_client = None
        self.linker = None

    def _setup_llm(self):
        logger.info(f"Initializing LLM Client | Model: {self.config.model_name}")
        if self.config.model_name.lower() == "local":
            if not self.config.local_model_path:
                msg = "Model set to 'local' but --local_model_path not provided."
                logger.critical(msg)
                raise ValueError(msg)
                
            return LocalLLMClient(
                model_path=self.config.local_model_path,
                device=self.config.device,
                max_retries=self.config.max_retries,
                seed=self.config.seed,
                enable_thinking=self.config.enable_thinking,
                max_new_tokens=self.config.max_new_tokens
            )
        else:
            return OpenAIClient(
                model_name=self.config.model_name,
                base_url=self.config.api_base,
                api_key=self.config.api_key,
                max_retries=self.config.max_retries,
                seed=self.config.seed,
                max_new_tokens=self.config.max_new_tokens,
                enable_thinking=self.config.enable_thinking
            )

    def load_data(self) -> List[Dict[str, Any]]:
        path = self.config.input_file
        logger.info(f"Loading data from {path}...")
        
        if not path or not os.path.exists(path):
             raise FileNotFoundError(f"Input file not found at: {path}")

        if path.endswith('.jsonl'):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def preprocess(self, db_ids: Set[str]):
        """
        Ensures that SIP prompts exist for all required databases.
        """
        output_base = os.path.join(self.config.preprocess_root, self.config.strategy, "prompts")
        os.makedirs(output_base, exist_ok=True)
        
        loader = Spider2Loader(self.config.db_root_path)
        
        to_process = []
        for db_id in db_ids:
            target_file = os.path.join(output_base, f"{db_id}.txt")
            if self.config.force_preprocess or not os.path.exists(target_file):
                to_process.append(db_id)
                
        if not to_process:
            logger.info(f"All {len(db_ids)} databases are already preprocessed.")
            return

        logger.info(f"Preprocessing {len(to_process)} databases using strategy '{self.config.strategy}'...")
        
        # Use Factory to get the strategy instance
        preprocessor = PreprocessorFactory.create(
            self.config.strategy,
            enable_column_description=self.config.enable_column_description,
            enable_column_type=self.config.enable_column_type,
            enable_sample_values=self.config.enable_sample_values,
            sample_values_max_items=self.config.sample_values_max_items,
            sample_value_max_chars=self.config.sample_value_max_chars,
        )
        
        for db_id in tqdm(to_process, desc="Preprocessing"):
            try:
                db = loader.load_database(db_id)
                compressed_context = preprocessor.compress(db)
                
                target_file = os.path.join(output_base, f"{db_id}.txt")
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(compressed_context)
                    
            except Exception as e:
                logger.error(f"Failed to preprocess database {db_id}: {e}")

    def run(self):
        # 1. Resolve run directory and configure file logging (no console logging)
        output_dir = self._resolve_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        setup_logger(level=self.config.log_level, log_file=os.path.join(output_dir, "app.log"))

        # 1.1 Resolve global seed: use configured seed; else reuse from resumed run;
        # otherwise generate a random seed for this run.
        self.config.seed = self._resolve_global_seed(output_dir)
        logger.info(f"Using global seed: {self.config.seed}")

        # 2. Load Data
        try:
            dataset = self.load_data()
        except Exception as e:
            logger.critical(f"Failed to load data: {e}", exc_info=True)
            sys.exit(1)

        # 3. Optional external knowledge summarization stage
        summary_mode = str(getattr(self.config, "external_knowledge_summary_mode", "off") or "off")
        logger.info(f"External knowledge summary mode: {summary_mode}")
        needs_summary_stage = bool(self.config.generate_external_knowledge_summary or self.config.use_external_knowledge_summary)
        if needs_summary_stage:
            if self.llm_client is None and self.config.generate_external_knowledge_summary:
                self.llm_client = self._setup_llm()

            summarizer = ExternalKnowledgeSummarizer(self.llm_client, self.config)
            summary_stats = summarizer.summarize_dataset(dataset)
            logger.info(f"External knowledge summary stage finished: {summary_stats}")

            if self.config.external_knowledge_summary_only:
                logger.info("external_knowledge_summary_only=true, exiting after summary stage.")
                print("\nExternal knowledge summary generation completed (--external_knowledge_summary_only).\n")
                return
        
        # 4. Preprocessing Check
        required_dbs = set(item['db_id'] for item in dataset)
        self.preprocess(required_dbs)

        if self.config.preprocess_only:
            logger.info("preprocess_only=true, skipping schema linking and evaluation stages.")
            print("\nPreprocess completed. Skipped schema linking/evaluation (--preprocess_only).\n")
            return

        # 5. Setup LLM and linker after preprocess stage
        if self.llm_client is None:
            self.llm_client = self._setup_llm()
        self.linker = SIPSchemaLinker(self.llm_client, self.config)
        
        # 5. Execution
        
        evaluator = EvaluationRunner(self.linker, output_dir)
        summary = evaluator.run(
            dataset,
            limit=self.config.limit,
            strategy=self.config.strategy,
            resume=self.config.resume,
        )
        
        # 6. Report
        self._print_summary(summary)

    def _resolve_global_seed(self, output_dir: str) -> int:
        # If user explicitly configured seed, respect it.
        if self.config.seed is not None:
            return int(self.config.seed)

        # For resume mode, try to recover seed from existing summary for continuity.
        if self.config.resume:
            summary_path = os.path.join(output_dir, "eval_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    existing_seed = summary.get("seed")
                    if existing_seed is not None:
                        return int(existing_seed)
                except Exception as e:
                    logger.warning(f"Failed to load seed from existing summary: {e}")

        # Otherwise generate a random global seed for this run.
        return random.SystemRandom().randint(0, 2**31 - 1)

    def _print_summary(self, summary: Dict[str, Any]):
        print("\n" + "="*40)
        print("EXECUTION SUMMARY")
        print("="*40)
        print(f"Mode:          {self.config.mode}")
        print(f"Items:         {summary['count']}")
        if summary['count'] > 0:
            print(f"Recall (Case): {summary['recall_case']:.4f}")
            print(f"Avg Recall:    {summary['avg_recall_column']:.4f}") 
            print(f"Avg Precision: {summary['avg_precision']:.4f}")
        print("="*40 + "\n")

    def _resolve_output_dir(self) -> str:
        strategy_root = os.path.join(self.config.output_dir, self.config.strategy)
        os.makedirs(strategy_root, exist_ok=True)

        if self.config.run_id:
            timestamp = self.config.run_id
            return os.path.join(strategy_root, timestamp)

        if self.config.resume:
            existing_runs = [
                d for d in os.listdir(strategy_root)
                if os.path.isdir(os.path.join(strategy_root, d))
            ]
            if existing_runs:
                latest = sorted(existing_runs)[-1]
                return os.path.join(strategy_root, latest)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(strategy_root, timestamp)
