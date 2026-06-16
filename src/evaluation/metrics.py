from typing import List, Tuple, Set

def normalize_schema(s: str) -> str:
    """
    Normalize schema string for comparison.
    We lower case and split by dots. 
    1. Lowercase
    2. Handle Schema.Table.Column vs Table.Column vs Column mismatch logic if needed.
    """
    parts = s.lower().strip().split('.')
    # Heuristic: keep last 3 parts if > 3 (DB.SCHEMA.TABLE.COL -> SCHEMA.TABLE.COL)
    # Usually it's SCHEMA.TABLE.COL or TABLE.COL
    # We will try to match primarily on the last 2 parts (TABLE.COL) as minimal unique identifier within a DB context usually,
    # but strictly checking full parts is safer if available.
    
    # Let's standardize to at most 3 parts: schema.table.column
    if len(parts) > 3:
        return ".".join(parts[-3:])
    return ".".join(parts)

def calculate_metrics(pred: List[str], gold: List[str]) -> Tuple[float, float, float]:
    """
    Calculate Precision, Recall, F1 for schema linking.
    
    Args:
        pred: List of predicted column strings.
        gold: List of gold standard column strings.
        
    Returns:
        (precision, recall, f1)
    """
    # Normalize
    p_set = set(normalize_schema(x) for x in pred)
    g_set = set(normalize_schema(x) for x in gold)
    
    # Calculate overlap
    tp = len(p_set.intersection(g_set))
    fp = len(p_set - g_set)
    fn = len(g_set - p_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1
