from typing import Dict, Any, List, Set
from collections import defaultdict

from src.core.domain import Database
from .base import ICompressionStrategy


class InheritanceStrategy(ICompressionStrategy):
    """
    Compression Strategy 2: Template Inheritance.
    - Group identical schemas into shared templates.
    - Allow larger templates to extend compact parent templates when beneficial.
    """

    def __init__(self, min_parent_cols: int = 3, min_gain: int = 2, max_delta_ratio: float = 0.45, **kwargs):
        super().__init__(**kwargs)
        self.min_parent_cols = min_parent_cols
        self.min_gain = min_gain
        self.max_delta_ratio = max_delta_ratio

    def analyze(self, database: Database) -> Dict[str, Any]:
        table_by_name = {t.full_name: t for t in database.tables}

        # 1) Exact-group templates by column signature.
        sig_to_tables: Dict[tuple, List[str]] = defaultdict(list)
        for table in database.tables:
            signature = tuple(sorted(c.signature for c in table.columns))
            sig_to_tables[signature].append(table.full_name)

        templates: List[Dict[str, Any]] = []
        for idx, (signature, table_names) in enumerate(sig_to_tables.items(), start=1):
            first_table = table_by_name[table_names[0]]
            columns = [
                {
                    "name": c.name,
                    "type": c.original_type,
                    "desc": c.description,
                    "sample_values": (c.metadata or {}).get("sample_values", []),
                }
                for c in first_table.columns
            ]
            templates.append(
                {
                    "id": idx,
                    "name": f"T{idx}",
                    "signature": set(signature),
                    "columns": columns,
                    "instances": sorted(table_names),
                    "is_derived": False,
                    "parent": None,
                    "delta_columns": [],
                }
            )

        # 2) Try inheritance among templates, from smaller -> larger.
        templates.sort(key=lambda t: len(t["signature"]))
        for i in range(len(templates)):
            current = templates[i]
            current_sig: Set[tuple] = current["signature"]
            current_col_count = len(current_sig)
            if current_col_count == 0:
                continue

            best_parent_idx = -1
            best_overlap = -1

            for j in range(i):
                parent = templates[j]
                parent_sig = parent["signature"]
                if len(parent_sig) < self.min_parent_cols:
                    continue
                if not parent_sig.issubset(current_sig):
                    continue
                overlap = len(parent_sig)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent_idx = j

            if best_parent_idx < 0:
                continue

            parent = templates[best_parent_idx]
            parent_sig = parent["signature"]
            delta_sigs = current_sig - parent_sig
            delta_count = len(delta_sigs)

            # Heuristic: derive only when compact enough and truly saves representation cost.
            # Full representation cost ~= current_col_count
            # Derived representation cost ~= 1(parent ref) + delta_count
            gain = current_col_count - (1 + delta_count)
            if gain < self.min_gain:
                continue
            if delta_count / max(1, current_col_count) > self.max_delta_ratio:
                continue

            current["is_derived"] = True
            current["parent"] = parent["name"]
            current["delta_columns"] = [
                c
                for c in current["columns"]
                if (c["name"].lower(), str(c["type"]).lower() if c["type"] else "unknown")
                in delta_sigs
            ]
            current["gain"] = gain
            del current["columns"]

        by_name = {t["name"]: t for t in templates}
        return {
            "strategy": "inheritance",
            "templates": by_name,
        }

    def generate_prompt_context(self, artifacts: Dict[str, Any]) -> str:
        templates: Dict[str, Dict[str, Any]] = artifacts.get("templates", {})
        all_parents = {
            t["parent"] for t in templates.values() if t.get("is_derived") and t.get("parent")
        }

        standalone_tables: List[tuple] = []
        base_templates: List[tuple] = []
        derived_templates: List[tuple] = []

        for t_name, t_data in templates.items():
            if t_data.get("is_derived"):
                derived_templates.append((t_name, t_data))
                continue

            # Single-table, non-parent template => flatten as standalone for readability.
            if len(t_data["instances"]) == 1 and t_name not in all_parents:
                only_table = t_data["instances"][0]
                cols = [self.render_column_name_with_metadata(c) for c in t_data["columns"]]
                standalone_tables.append((only_table, cols))
            else:
                base_templates.append((t_name, t_data))

        # Sort for stable output and easier diffing.
        base_templates.sort(key=lambda x: (-len(x[1]["instances"]), x[0]))
        derived_templates.sort(key=lambda x: (-len(x[1]["instances"]), x[0]))
        standalone_tables.sort()

        lines: List[str] = []

        if base_templates:
            lines.append("[TEMPLATES]")
            for t_name, t_data in base_templates:
                lines.append(f"<{t_name}>")
                for col in t_data["columns"]:
                    lines.append(self.render_column_name_with_metadata(col))
                covered = ",".join(t_data["instances"])
                lines.append(f"@Tables: [{covered}]")
                lines.append("")

        if derived_templates:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[DERIVED]")
            for t_name, t_data in derived_templates:
                parent = t_data["parent"]
                lines.append(f"<{t_name}> extends <{parent}>")
                for col in t_data.get("delta_columns", []):
                    lines.append(self.render_column_name_with_metadata(col))
                covered = ",".join(t_data["instances"])
                lines.append(f"@Tables: [{covered}]")
                lines.append("")

        if standalone_tables:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[STANDALONE]")
            for table_name, cols in standalone_tables:
                lines.append(f"## {table_name}")
                for col in cols:
                    lines.append(col)
                lines.append("")

        return "\n".join(lines)
