import os
import glob
import json
from collections import defaultdict
from typing import List, Optional
from src.core.interfaces import IDatabaseLoader
from src.core.domain import Database, Table, Column

class Spider2Loader(IDatabaseLoader):
    """
    Adapter for loading Spider 2.0 format database schemas.
    Spider 2 format consists of nested directories with individual JSON files per table.
    """
    
    def __init__(self, root_path: str):
        """
        Args:
            root_path: Absolute path to the directory containing database folders.
                       e.g. /data/spider2/resource/databases
        """
        self.root_path = root_path
        
    def list_available_databases(self) -> List[str]:
        if not os.path.exists(self.root_path):
            return []
        # Return only directories
        return [
            d for d in os.listdir(self.root_path) 
            if os.path.isdir(os.path.join(self.root_path, d))
        ]
        
    def load_database(self, db_id: str) -> Database:
        db_path = os.path.join(self.root_path, db_id)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database {db_id} not found at {db_path}")
            
        tables = []
        # Spider 2 structure: db_id/schema_folder/*.json
        # Sometimes structure is flat, sometimes nested. Recursive search is safest.
        # We look for *.json files, but filter out potential non-table JSONs if needed.
        json_files = glob.glob(os.path.join(db_path, "**", "*.json"), recursive=True)
        
        for jf in json_files:
            try:
                table = self._parse_single_json(jf)
                if table:
                    tables.append(table)
            except Exception as e:
                # Log warning but continue loading other tables
                # print(f"[Spider2Loader] Warning: Failed to parse {jf}: {e}")
                pass
                
        # Optional: Load DB level description if available (e.g. from README or description.json)
        # For now, we leave it None or implement specific logic later.
        
        return Database(id=db_id, tables=tables, source_path=db_path)
        
    def _parse_single_json(self, file_path: str) -> Optional[Table]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Basic validation: It must look like a table definition
        if 'table_name' not in data:
            return None
            
        # Extract columns
        cols = []
        c_names = data.get('column_names', [])
        c_types = data.get('column_types', [])
        # 'description' field in Spider 2 JSON is often a list of descriptions corresponding to columns
        # or sometimes null.
        c_descs = data.get('description', []) 
        
        sample_rows = data.get('sample_rows', [])
        sample_values_map = self._collect_sample_values(c_names, sample_rows)

        # Normalize lengths for zip
        length = len(c_names) if c_names else 0
        if length == 0:
            # Empty table?
            pass
            
        # Ensure types and descs list match length
        safe_types = c_types if (c_types and len(c_types) == length) else ["TEXT"] * length
        safe_descs = c_descs if (c_descs and len(c_descs) == length) else [None] * length
        
        for n, t, d in zip(c_names, safe_types, safe_descs):
            # Spider 2 column types might be None or weird strings
            dtype = str(t) if t is not None else "TEXT"
            desc = str(d) if d else None
            
            if not n: continue # Skip empty column names
            
            cols.append(Column(
                name=str(n),
                original_type=dtype,
                description=desc,
                metadata={
                    "sample_values": sample_values_map.get(str(n), [])
                }
            ))
            
        # Extract Schema Namespace
        # Usually from 'table_fullname': 'DB.SCHEMA.TABLE'
        # Or from file path structure
        schema_ns = None
        full_name = data.get('table_fullname', '')
        if full_name:
            parts = full_name.split('.')
            if len(parts) > 1:
                # "DB.SCHEMA.TABLE" -> namespace is parts[-2]
                # "SCHEMA.TABLE" -> namespace is parts[0]
                # Heuristic: If 3 parts, take middle. If 2 parts, take first.
                if len(parts) >= 3:
                     schema_ns = parts[-2]
                elif len(parts) == 2:
                     schema_ns = parts[0]
        
        raw_name = data.get('table_name', 'unknown')
        if raw_name:
            parts = raw_name.split('.')
            raw_name = parts[-1]  # Just the table name without schema

        return Table(
            name=raw_name,
            columns=cols,
            schema_namespace=schema_ns,
            description=None, # Table desc not standard in these JSON files
            metadata={"original_file": file_path, "raw_fullname": full_name}
        )

    def _collect_sample_values(self, column_names: List[str], sample_rows: object, max_rows: int = 20, max_values_per_col: int = 6):
        result = {str(c): [] for c in (column_names or [])}

        if not isinstance(sample_rows, list) or not sample_rows:
            return result

        lowered_keys = {str(c).lower(): str(c) for c in (column_names or [])}
        dedup_seen = defaultdict(set)

        for row in sample_rows[:max_rows]:
            if not isinstance(row, dict):
                continue

            # Build case-insensitive row lookup once per row.
            row_lc = {str(k).lower(): v for k, v in row.items()}

            for col_lc, canonical_name in lowered_keys.items():
                if len(result[canonical_name]) >= max_values_per_col:
                    continue

                value = row_lc.get(col_lc, None)
                if value is None:
                    continue

                marker = self._make_hashable_marker(value)
                if marker in dedup_seen[canonical_name]:
                    continue

                dedup_seen[canonical_name].add(marker)
                result[canonical_name].append(value)

        return result

    @staticmethod
    def _make_hashable_marker(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return (type(value).__name__, value)
        try:
            return ("json", json.dumps(value, sort_keys=True, ensure_ascii=False))
        except Exception:
            return ("repr", repr(value))
