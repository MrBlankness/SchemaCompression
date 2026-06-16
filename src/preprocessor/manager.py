from typing import Dict, Type
from src.core.interfaces import ISchemaPreprocessor
from .strategies.base import ICompressionStrategy
from .strategies.factorizer import FactorizationStrategy
from .strategies.inheritance import InheritanceStrategy
from .strategies.raw import RawStrategy

class PreprocessorFactory:
    """
    Factory to create specific Schema Preprocessor instances.
    """
    
    _strategies: Dict[str, Type[ICompressionStrategy]] = {
        "factorization": FactorizationStrategy,
        "inheritance": InheritanceStrategy,
        "raw": RawStrategy,
    }

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: Type[ICompressionStrategy]):
        cls._strategies[name] = strategy_cls

    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> ISchemaPreprocessor:
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(cls._strategies.keys())}")
        
        # Instantiate strategy directly
        return cls._strategies[strategy_name](**kwargs)

