from typing import Dict, Any
from src.core.domain import Database
from .base import ICompressionStrategy

class RawStrategy(ICompressionStrategy):
    """
    Baseline Strategy: No compression.
    Outputs the full list of tables and columns directly.
    """

    def analyze(self, database: Database) -> Dict[str, Any]:
        """
        No analysis needed for raw strategy.
        Just structure the data for generation.
        """
        tables_data = []
        for table in database.tables:
            tables_data.append({
                "name": table.full_name,
                "columns": [
                    {
                        "name": col.name,
                        "type": col.original_type,
                        "description": col.description,
                        "sample_values": (col.metadata or {}).get("sample_values", []),
                    }
                    for col in table.columns
                ]
            })
            
        return {
            "strategy": "raw",
            "tables": tables_data
        }

    def generate_prompt_context(self, artifacts: Dict[str, Any]) -> str:
        lines = []
        lines.append("[TABLES]")
        
        for table in artifacts.get("tables", []):
            lines.append(f"## {table['name']}")
            for col in table['columns']:
                lines.append(self.render_column_name_with_metadata(col))
            lines.append("")
            
        return "\n".join(lines)
