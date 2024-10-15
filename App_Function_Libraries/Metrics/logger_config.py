# logger_config.py
#
# Imports
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
import os
#
############################################################################################################
#
# Functions:

def setup_logger(log_file_path="tldw_app_logs.json"):
    """
    Sets up the logger with both StreamHandler and FileHandler, formatted in JSON.

    Parameters:
        log_file_path (str): Path to the JSON log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("tldw_app_logs")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

    # Prevent adding multiple handlers if the logger is already configured
    if not logger.handlers:
        # StreamHandler for console output
        stream_handler = logging.StreamHandler()
        stream_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # Ensure the directory for log_file_path exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # RotatingFileHandler for writing logs to a JSON file with rotation
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10*1024*1024, backupCount=5  # 10 MB per file, keep 5 backups
        )
        file_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize the logger
logger = setup_logger()

#
# End of Functions
############################################################################################################
