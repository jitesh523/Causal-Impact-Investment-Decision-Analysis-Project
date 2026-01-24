"""
Logging Configuration Module
Provides structured logging for the entire application
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    name: str = "causal_impact",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "causal_impact") -> logging.Logger:
    """Get or create a logger with the given name"""
    return logging.getLogger(name)


# Default application logger
logger = setup_logging(
    name="causal_impact",
    level=logging.INFO,
    log_file="logs/causal_impact.log"
)


# Convenience functions that use the default logger
def info(msg: str) -> None:
    """Log an info message"""
    logger.info(msg)


def warning(msg: str) -> None:
    """Log a warning message"""
    logger.warning(msg)


def error(msg: str) -> None:
    """Log an error message"""
    logger.error(msg)


def debug(msg: str) -> None:
    """Log a debug message"""
    logger.debug(msg)
