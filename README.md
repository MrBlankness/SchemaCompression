# SchemaCompressor - CompressCore Module

A powerful schema compression and linking framework for database schema analysis and natural language query processing.

## Overview

CompressCore is the core module of the SchemaCompressor project, designed to compress complex database schemas into more manageable representations for natural language query processing. It supports multiple compression strategies and integrates with various LLM providers.

## Features

- **Multiple Compression Strategies**: Inheritance-based, factorization-based, and raw schema compression
- **LLM Integration**: Support for OpenAI-compatible APIs and local HuggingFace models
- **External Knowledge Integration**: Incorporate domain-specific knowledge for improved schema understanding
- **Flexible Configuration**: Environment-based configuration with YAML support
- **Evaluation Framework**: Built-in evaluation metrics for schema linking performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MrBlankness/SchemaCompression.git
cd SchemaCompression/CompressCore
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy and edit the environment template
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

## Configuration

Create a `config.yaml` file based on the template:

```yaml
# API Configuration
api_base: ${OPENAI_BASE_URL:-http://localhost:8000/v1}
api_key: ${OPENAI_API_KEY:-your-api-key-here}

# Model Settings
model_name: gpt-4o
max_retries: 3
max_new_tokens: 1024

# Compression Strategy
strategy: inheritance  # inheritance, factorization, or raw

# External Knowledge
enable_external_knowledge: false
external_knowledge_max_chars: 4000
```

Or use environment variables:
```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=your-api-base-url
```

## Usage

### Basic Usage

```bash
python src/main.py --input_file data/input/dev.json --db_root_path data/raw/resource/databases
```

### Advanced Options

```bash
# Use factorization strategy with external knowledge
python src/main.py \
    --input_file data/input/dev.json \
    --db_root_path data/raw/resource/databases \
    --strategy factorization \
    --enable_external_knowledge \
    --model_name gpt-4o

# Use local model
python src/main.py \
    --input_file data/input/dev.json \
    --db_root_path data/raw/resource/databases \
    --model_name local \
    --local_model_path /path/to/your/model
```

## Project Structure

```
CompressCore/
├── src/                    # Source code
│   ├── core/              # Core interfaces and domain models
│   ├── data_loader/       # Data loading utilities
│   ├── evaluation/        # Evaluation framework
│   ├── knowledge/         # External knowledge integration
│   ├── linker/            # Schema linking algorithms
│   ├── llm_service/       # LLM client implementations
│   ├── preprocessor/      # Schema preprocessing
│   └── utils/             # Utility functions
├── data/                  # Data directories
│   ├── input/            # Input files
│   ├── preprocess/       # Preprocessed data
│   └── output/           # Output results
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Compression Strategies

### 1. Inheritance-based Compression
Organizes schema elements using inheritance hierarchies to reduce redundancy.

### 2. Factorization-based Compression  
Factors common schema patterns into reusable components.

### 3. Raw Schema Processing
Processes schemas without compression for baseline comparison.

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SchemaCompressor in your research, please cite our work:

```bibtex
@software{schemacompression,
  title = {SchemaCompressor: A Framework for Database Schema Compression and Linking},
  author = {SchemaCompression Team},
  year = {2024},
  url = {https://github.com/MrBlankness/SchemaCompression}
}
```

## Support

For questions and support, please open an issue on GitHub or contact the development team.
