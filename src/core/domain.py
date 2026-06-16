from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Column:
    """
    Represents a single database column.
    """
    name: str
    original_type: str  # Database specific type description (e.g., 'VARCHAR(255)', 'INTEGER')
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> tuple:
        """
        Returns a normalized signature (name_lower, type_lower) for comparison/hashing.
        """
        # Handle cases where type might be None or complex
        t_str = str(self.original_type).lower() if self.original_type else "unknown"
        return (self.name.lower(), t_str)

@dataclass
class Table:
    """
    Represents a database table schema.
    """
    name: str # Just the table name (e.g., 'orders')
    columns: List[Column]
    description: Optional[str] = None # Table-level description
    schema_namespace: Optional[str] = None  # Database schema/namespace (e.g., 'public', 'sales_data')
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """
        Returns the fully qualified name if schema exists, else just table name.
        Example: 'sales_data.orders'
        """
        if self.schema_namespace:
            return f"{self.schema_namespace}.{self.name}"
        return self.name

@dataclass
class Database:
    """
    Represents an entire Database containing multiple tables.
    """
    id: str  # Unique identifier for the database instance (e.g., 'spider2_flight_dataset')
    tables: List[Table]
    description: Optional[str] = None
    source_path: str = "" # File system path or connection string
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_table_by_name(self, name: str) -> Optional[Table]:
        """
        Case-insensitive lookup for a table by its name or full_name.
        """
        target = name.lower()
        for t in self.tables:
            if t.name.lower() == target or t.full_name.lower() == target:
                return t
        return None
