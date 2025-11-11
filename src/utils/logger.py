import logging
import os


def configure_logging() -> None:
    """Configure root logging based on LOG_LEVEL env or default to INFO."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
