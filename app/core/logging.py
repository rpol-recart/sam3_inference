"""Logging configuration for the SAM3 Inference Server."""
import sys
import logging
from typing import Optional

from loguru import logger
from pydantic import BaseSettings


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = "INFO"
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    log_file: Optional[str] = None
    log_rotation: str = "100 MB"
    log_retention: str = "7 days"


def configure_logging(log_level: str = "INFO", log_format: str = None, log_file: Optional[str] = None):
    """Configure logging for the application."""
    # Remove default logger
    logger.remove()
    
    # Add handler to stdout
    logger.add(
        sys.stdout,
        format=log_format or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        backtrace=True,
        diagnose=True,
    )
    
    # Add handler to file if specified
    if log_file:
        logger.add(
            log_file,
            rotation="100 MB",
            retention="7 days",
            level=log_level,
            format=log_format or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            backtrace=True,
            diagnose=True,
        )
    
    return logger


# Create global logger instance
default_logger = configure_logging()