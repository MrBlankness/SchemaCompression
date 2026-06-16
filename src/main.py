import sys
import os
from typing import Dict, Any

# Add project root to path
# This allows us to use standard "from src.xxx import yyy" style
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig
from src.pipeline import PipelineOrchestrator

def main():
    # 1. Configuration
    # Note: Config loading now handles CLI > YAML > Defaults priority
    config = AppConfig.from_args()
    
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        
    # 2. Pipeline Execution
    try:
        orchestrator = PipelineOrchestrator(config)
        orchestrator.run()
    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
