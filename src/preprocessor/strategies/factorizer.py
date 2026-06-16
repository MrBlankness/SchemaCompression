from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from src.core.domain import Database
from .base import ICompressionStrategy

class FactorizationStrategy(ICompressionStrategy):
    """
    Compression Strategy 1: Column Factorization.
    Mines frequent column sets (Components) and refactors tables to reference these components.
    """
    
    def __init__(
        self,
        min_support: int = 2,
        min_cols: int = 2,
        min_gain: int = 1,
        max_components_per_table: int = 10**9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_support = min_support
        self.min_cols = min_cols
        self.min_gain = min_gain
        self.max_components_per_table = max_components_per_table

    def analyze(self, database: Database) -> Dict[str, Any]:
        """
        Implementation of the greedy frequent itemset mining adapted for Domain objects.
        """
        col_to_tables = defaultdict(set)
        
        # 1. Build Index: Column Signature -> Set of Table Indices
        for idx, table in enumerate(database.tables):
            for col in table.columns:
                col_to_tables[col.signature].add(idx)
                
        # 2. Group Columns by Table Sets
        # Key: frozenset(table_indices), Value: List[Column Signatures]
        table_set_to_cols = defaultdict(list)
        
        # We also need to keep track of the original column details (type, desc) 
        # to define the component later. We'll pick the 'most common' or first definition.
        col_definitions = {} 
        
        for idx, table in enumerate(database.tables):
            for col in table.columns:
                if col.signature not in col_definitions:
                    col_definitions[col.signature] = col # Cache first occurrence
        
        for col_sig, table_indices in col_to_tables.items():
            if len(table_indices) < self.min_support:
                continue
            key = frozenset(table_indices)
            table_set_to_cols[key].append(col_sig)
            
        # 3. Build candidate components and greedily keep only high-gain ones.
        candidates = []
        for table_indices, col_sigs in table_set_to_cols.items():
            if len(col_sigs) < self.min_cols:
                continue

            support = len(table_indices)
            width = len(col_sigs)
            # Approximate token-saving score:
            # old ~= support * width
            # new ~= width (component def) + support (table refs)
            gain = (support * width) - (width + support)
            if gain < self.min_gain:
                continue

            candidates.append({
                "table_indices": set(table_indices),
                "col_sigs": sorted(col_sigs),
                "support_count": support,
                "gain": gain,
            })

        candidates.sort(
            key=lambda x: (x["gain"], x["support_count"], len(x["col_sigs"])),
            reverse=True,
        )

        selected_components: List[Dict[str, Any]] = []
        table_component_counts = defaultdict(int)

        for candidate in candidates:
            if any(
                table_component_counts[t_idx] >= self.max_components_per_table
                for t_idx in candidate["table_indices"]
            ):
                continue
            selected_components.append(candidate)
            for t_idx in candidate["table_indices"]:
                table_component_counts[t_idx] += 1

        components: Dict[str, Dict[str, Any]] = {}
        col_to_comp_map: Dict[Tuple[str, str], str] = {}

        for comp_idx, candidate in enumerate(selected_components, start=1):
            comp_name = f"@C{comp_idx}"
            comp_cols = []
            for sig in candidate["col_sigs"]:
                c_def = col_definitions[sig]
                comp_cols.append({
                    "name": c_def.name,
                    "type": c_def.original_type,
                    "desc": c_def.description,
                    "sample_values": (c_def.metadata or {}).get("sample_values", []),
                })
                col_to_comp_map[sig] = comp_name

            components[comp_name] = {
                "columns": comp_cols,
                "support_count": candidate["support_count"],
                "gain": candidate["gain"],
            }
            
        # 4. Rewrite Tables
        processed_tables = []
        for table in database.tables:
            kept_cols = []
            used_comps = set()
            
            for col in table.columns:
                if col.signature in col_to_comp_map:
                    used_comps.add(col_to_comp_map[col.signature])
                else:
                    kept_cols.append({
                        "name": col.name,
                        "type": col.original_type,
                        "desc": col.description,
                        "sample_values": (col.metadata or {}).get("sample_values", []),
                    })
            
            processed_tables.append({
                "name": table.name,
                "full_name": table.full_name,
                "kept_columns": kept_cols,
                "components": sorted(list(used_comps))
            })
            
        return {
            "strategy": "factorization",
            "components": components,
            "tables": processed_tables
        }

    def generate_prompt_context(self, artifacts: Dict[str, Any]) -> str:
        lines: List[str] = []

        lines.append("[COMPONENTS]")
        comps = artifacts.get("components", {})
        for c_name, c_data in comps.items():
            lines.append(c_name)
            for col in c_data["columns"]:
                lines.append(self.render_column_name_with_metadata(col))
            lines.append("")

        lines.append("[TABLES]")
        for t in artifacts.get("tables", []):
            lines.append(f"## {t['full_name']}")
            if t["components"]:
                refs = [comp[1:] if comp.startswith("@") else comp for comp in t["components"]]
                lines.append(f"@use: [{', '.join(refs)}]")
            for col in t["kept_columns"]:
                lines.append(self.render_column_name_with_metadata(col))
            lines.append("")

        return "\n".join(lines)
