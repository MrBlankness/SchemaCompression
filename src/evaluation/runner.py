import json
import os
import logging
import numpy as np
from typing import List, Dict, Any, Set
from tqdm import tqdm
from src.core.interfaces import ISchemaLinker
from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class EvaluationRunner:
    def __init__(self, linker: ISchemaLinker, output_dir: str):
        self.linker = linker
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run(self, dataset: List[Dict[str, Any]], limit: int = 0, strategy: str = "inheritance", resume: bool = False) -> Dict[str, Any]:
        """
        Run evaluation on a dataset.
        """
        if limit > 0:
            dataset = dataset[:limit]
            
        results = []
        global_seed = getattr(getattr(self, "linker", None), "config", None)
        global_seed = getattr(global_seed, "seed", None)
        
        # Accumulators for aggregating metrics
        agg_precision = []
        agg_recall_column = [] 
        agg_token_cost = [] # Added accumulator for tokens
        
        # Count of cases where ALL relevant columns were recalled (Recall = 1.0)
        perfect_recall_cases = 0
        processed_case_ids: Set[str] = set()
        
        logger.info(f"Starting evaluation on {len(dataset)} items with strategy='{strategy}'...")
        
        details_file = os.path.join(self.output_dir, "eval_details.jsonl")
        details_mode = 'w'

        if resume and os.path.exists(details_file):
            logger.info(f"Resume enabled. Loading existing checkpoint: {details_file}")
            kept_checkpoint_rows: List[Dict[str, Any]] = []
            retry_case_ids: Set[str] = set()

            with open(details_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_item = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSONL checkpoint line.")
                        continue

                    case_id = self._get_case_id(existing_item)
                    pred = existing_item.get("pred", [])
                    if isinstance(pred, list) and len(pred) == 0:
                        retry_case_ids.add(case_id)
                        continue

                    kept_checkpoint_rows.append(existing_item)
                    results.append(existing_item)
                    processed_case_ids.add(case_id)

                    metric_res = existing_item.get("metrics")
                    if metric_res:
                        p = float(metric_res.get("precision", 0.0))
                        r = float(metric_res.get("recall_column", 0.0))
                        agg_precision.append(p)
                        agg_recall_column.append(r)
                        if r >= 0.9999:
                            perfect_recall_cases += 1

                    agg_token_cost.append(float(existing_item.get("token_cost", 0) or 0))

            if retry_case_ids:
                logger.info(
                    f"Found {len(retry_case_ids)} failed checkpoint cases (pred=[]). "
                    "They will be retried in this resume run."
                )

            # Rewrite checkpoint file with only kept successful/non-empty predictions,
            # then append newly processed rows during this run.
            with open(details_file, 'w', encoding='utf-8') as f:
                for row in kept_checkpoint_rows:
                    f.write(json.dumps(row) + "\n")

            details_mode = 'a'
            logger.info(f"Checkpoint loaded: {len(processed_case_ids)} cases kept as processed.")

        if details_mode == 'w':
            # Clear existing details file if we are starting a new run
            open(details_file, 'w').close()
        
        for item in tqdm(dataset, desc="Eval"):
            current_case_id = self._get_case_id(item)
            if current_case_id in processed_case_ids:
                continue

            q_id = item.get('question_id', 'unknown')
            db_id = item['db_id']
            db_cols_num = item.get("db_cols_num", "N/A") # Optional: number of columns in the DB, if available
            question = item['question']
            external_knowledge = item.get('external_knowledge')
            gold = item.get('gold_schema', [])
            
            token_cost = 0
            thought = ""
            case_seed = global_seed
            try:
                # Link with details
                link_res = self.linker.link_schema_details(
                    db_id,
                    question,
                    strategy,
                    external_knowledge=external_knowledge,
                )
                pred = link_res.get("columns", [])
                token_cost = link_res.get("token_cost", 0)
                thought = link_res.get("thought", "") or ""
                case_seed = link_res.get("seed", global_seed)
                
            except Exception as e:
                logger.error(f"Error processing {q_id}: {e}", exc_info=True)
                pred = []
                thought = ""
                case_seed = global_seed
                
            # Metrics Calculation
            if gold:
                p, r, f1 = calculate_metrics(pred, gold)
                
                if r >= 0.9999: 
                    perfect_recall_cases += 1
                    
                metric_res = {
                    "precision": p,
                    "recall_column": r,
                    "f1": f1
                }
                
                # Add to aggregators
                agg_precision.append(p)
                agg_recall_column.append(r)
                
            else:
                metric_res = None
            
            # Aggregate token cost for all cases (even if gold is missing, cost is real)
            agg_token_cost.append(token_cost)
                
            result_item = {
                "question_id": q_id,
                "db_id": db_id,
                "db_cols_num": db_cols_num,
                "question": question,
                "seed": case_seed,
                "thought": thought,
                "pred": pred,
                "gold": gold,
                "metrics": metric_res,
                "token_cost": token_cost # Added to per-item result
            }
            results.append(result_item)
            processed_case_ids.add(current_case_id)
            
            # Streaming save to JSONL
            with open(details_file, details_mode, encoding='utf-8') as f:
                f.write(json.dumps(result_item) + "\n")
            details_mode = 'a'
            
        # Overall Summary
        total_cases = len(results)
        linker_config = getattr(self.linker, "config", None)
        summary = {
            "count": total_cases,
            "recall_case": perfect_recall_cases / total_cases if total_cases > 0 else 0.0,
            "avg_recall_column": float(np.mean(agg_recall_column)) if agg_recall_column else 0.0,
            "avg_precision": float(np.mean(agg_precision)) if agg_precision else 0.0,
            "avg_token_cost": float(np.mean(agg_token_cost)) if agg_token_cost else 0.0, # Added average token cost
            "seed": global_seed,
            "enable_column_type": bool(getattr(linker_config, "enable_column_type", False)),
            "enable_column_description": bool(getattr(linker_config, "enable_column_description", False)),
            "enable_sample_values": bool(getattr(linker_config, "enable_sample_values", False)),
            "enable_external_knowledge": bool(getattr(linker_config, "enable_external_knowledge", False)),
            "external_knowledge_summary_mode": str(getattr(linker_config, "external_knowledge_summary_mode", "off") or "off"),
            "external_knowledge_summary_root": str(getattr(linker_config, "external_knowledge_summary_root", "") or ""),
            "sample_values_max_items": int(getattr(linker_config, "sample_values_max_items", 0) or 0),
            "sample_value_max_chars": int(getattr(linker_config, "sample_value_max_chars", 0) or 0),
            "external_knowledge_max_chars": int(getattr(linker_config, "external_knowledge_max_chars", 0) or 0),
            "external_knowledge_summary_max_chars": int(getattr(linker_config, "external_knowledge_summary_max_chars", 0) or 0),
        }
        
        with open(os.path.join(self.output_dir, f"eval_details.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        with open(os.path.join(self.output_dir, f"eval_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        return summary

    @staticmethod
    def _get_case_id(item: Dict[str, Any]) -> str:
        question_id = item.get("question_id")
        if question_id:
            return f"qid::{question_id}"
        return f"fallback::{item.get('db_id', '')}::{item.get('question', '')}"