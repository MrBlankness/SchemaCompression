import logging
from typing import Optional

def setup_logger(name: str = "schema_linker", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a logger instance.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    if not log_file:
        raise ValueError("log_file is required for file-only logging setup")

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler only
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Attach file handler to root logger so all module loggers are captured.
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
        existing.close()
    root_logger.addHandler(file_handler)

    # Keep named logger for direct use; let messages propagate to root handlers.
    logger.propagate = True

    return logger
