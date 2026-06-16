from abc import ABC, abstractmethod
from typing import Dict, Any
from src.core.domain import Database
from src.core.interfaces import ISchemaPreprocessor
from src.preprocessor.metadata_formatter import (
    MetadataRenderOptions,
    render_column_line,
)

class ICompressionStrategy(ISchemaPreprocessor, ABC):
    """
    Abstract Base Class for Schema Compression Strategies.
    Implements the ISchemaPreprocessor interface using the Template Method pattern.
    """
    
    def __init__(self, **kwargs):
        self.metadata_options = MetadataRenderOptions(
            enable_column_description=bool(kwargs.get("enable_column_description", False)),
            enable_column_type=bool(kwargs.get("enable_column_type", False)),
            enable_sample_values=bool(kwargs.get("enable_sample_values", False)),
            sample_values_max_items=max(0, int(kwargs.get("sample_values_max_items", 2))),
            sample_value_max_chars=max(16, int(kwargs.get("sample_value_max_chars", 120))),
        )

    def compress(self, database: Database) -> str:
        """
        Template method implementation of ISchemaPreprocessor.compress.
        Orchestrates the Analyze -> Generate pipeline.
        """
        artifacts = self.analyze(database)
        return self.generate_prompt_context(artifacts)

    def render_column_name_with_metadata(self, column: Dict[str, Any]) -> str:
        return render_column_line(
            name=column["name"],
            column_type=column.get("type"),
            description=column.get("desc") or column.get("description"),
            sample_values=column.get("sample_values"),
            options=self.metadata_options,
        )
    
    @abstractmethod
    def analyze(self, database: Database) -> Dict[str, Any]:
        """
        Analyzes the database schema to find compression opportunities.
        Returns a dictionary containing the compressed structure (Artifacts).
        """
        pass
        
    @abstractmethod
    def generate_prompt_context(self, artifacts: Dict[str, Any]) -> str:
        """
        Converts the compressed structural artifacts into a human/LLM-readable string 
        following the SIP (Schema Inheritance Protocol).
        """
        pass
