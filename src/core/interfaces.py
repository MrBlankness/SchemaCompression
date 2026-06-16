from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple
from .domain import Database, Table

class IDatabaseLoader(ABC):
    """
    Interface for loading database schemas from various sources (Spider2 files, BIRD format, SQL connection, etc.).
    """
    
    @abstractmethod
    def load_database(self, db_id: str) -> Database:
        """
        Loads a specific database schema into the domain model.
        
        Args:
            db_id: The identifier of the database to load.
            
        Returns:
            A populated Database object.
        """
        pass
        
    @abstractmethod
    def list_available_databases(self) -> List[str]:
        """
        Lists IDs of all available databases this loader can access.
        """
        pass

class ISchemaPreprocessor(ABC):
    """
    Interface for transforming/compressing a raw Database schema into a representation optimized for LLMs.
    """
    @abstractmethod
    def compress(self, database: Database) -> str:
        """
        Transforms the database schema into a compressed text format.
        """
        pass

class ILLMClient(ABC):
    """
    Interface for LLM interaction. 
    """
    @abstractmethod
    def generate(self, messages: List[dict], temperature: float = 0.0, **kwargs) -> Tuple[str, int]:
        """
        Standard chat completion interface.
        
        Returns:
            A tuple containing:
            - response_text (str): The generated text.
            - input_token_count (int): The number of tokens in the input prompt.
        """
        pass

class ISchemaLinker(ABC):
    """
    Interface for the Schema Linking Agent.
    """
    @abstractmethod
    def link_schema_details(
        self,
        db_id: str,
        question: str,
        strategy: str,
        external_knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extended version of link_schema.
        
        Returns: A dict containing:
            - "columns": List[str]
            - "token_cost": int (optional)
        """
        pass
    
    def link_schema(self, db_id: str, question: str, strategy: str = "inheritance") -> List[str]:
        """Backward compatibility wrapper"""
        res = self.link_schema_details(db_id, question, strategy, external_knowledge=None)
        return res.get("columns", [])