# logger_config.py
#
# Imports
import json
import sys
import os
#
# 3rd-Party Imports
from loguru import logger
# Local Imports
from App_Function_Libraries.Utils.Utils import load_and_log_configs
#
############################################################################################################
#
# Functions:

loaded_config_data = load_and_log_configs()
log_metrics_file = loaded_config_data['logging']['log_metrics_file']

def retention_function(files):
    """
    A retention function to mimic backupCount=5.

    Given a list of log file paths, this function sorts them by modification time
    and returns the list of files to be removed so that at most 5 are kept.
    """
    if len(files) > 5:
        # Sort files by modification time (oldest first)
        files.sort(key=lambda filename: os.path.getmtime(filename))
        # Remove all but the 5 most recent files.
        return files[:-5]
    return []


def json_formatter(record):
    """
    Custom JSON formatter for file logging.
    """
    # Format the log time as a string.
    dt = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")
    # Grab any extra data passed with the log.
    extra = record["extra"]
    log_record = {
        "time": dt,
        "levelname": record["level"].name,
        "name": record["name"],
        "message": record["message"],
        "event": extra.get("event"),
        "type": extra.get("type"),
        "value": extra.get("value"),
        "labels": extra.get("labels"),
        "timestamp": extra.get("timestamp"),
    }
    return json.dumps(log_record)


def setup_logger(args):
    """
    Sets up Loguru using command-line arguments (if provided)
    and configuration file settings.

    This function adds:
      - A console sink with a simple human‑readable format.
      - A file sink for standard logs.
      - Optionally, a file sink with JSON formatting for metrics.
    """
    # Remove any previously added sinks.
    logger.remove()

    # Determine the log level (from args; default to DEBUG)
    log_level = args.log_level.upper() if hasattr(args, "log_level") else "DEBUG"

    # Add a console sink.
    # (You could also choose to use JSON formatting on the console if desired.)
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}"
    )

    # Determine the file sink for standard logs.
    # Prefer the command-line argument if provided; otherwise, use the config.
    if hasattr(args, "log_file") and args.log_file:
        file_log_path = args.log_file
        logger.info(f"Log file created at: {file_log_path}")
    else:
        config = load_and_log_configs()
        file_log_path = config['logging']['log_file']
        logger.info(f"No logfile provided via command-line. Using default: {file_log_path}")

    # Ensure the directory for the file_log_path exists.
    log_dir = os.path.dirname(file_log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Add the standard file sink with human‑readable format.
    logger.add(
        file_log_path,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}"
    )

    # Optionally, add a metrics file sink with JSON formatting.
    # For example, if you want to have a separate file that rotates at 10 MB:
    config = load_and_log_configs()
    metrics_log_file = config['logging'].get('log_metrics_file')
    if metrics_log_file:
        metrics_dir = os.path.dirname(metrics_log_file)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

        logger.add(
            metrics_log_file,
            level="DEBUG",  # Or another level if you prefer
            format=json_formatter,
            rotation="10 MB",
            # Loguru’s built-in retention can be a simple number (e.g., 5) meaning “keep 5 files”
            retention=5,
        )
    return logger

# def setup_logger(log_file_path="tldw_app_logs.json"):
#     """
#     Sets up the logger with both StreamHandler and FileHandler, formatted in JSON.
#
#     Parameters:
#         log_file_path (str): Path to the JSON log file.
#
#     Returns:
#         logging.Logger: Configured logger instance.
#     """
#     logger = logging.getLogger("tldw_app_logs")
#     logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
#
#     # Prevent adding multiple handlers if the logger is already configured
#     if not logger.handlers:
#         # StreamHandler for console output
#         stream_handler = logging.StreamHandler()
#         stream_formatter = jsonlogger.JsonFormatter(
#             '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
#         )
#         stream_handler.setFormatter(stream_formatter)
#         logger.addHandler(stream_handler)
#
#         # Ensure the directory for log_file_path exists
#         log_dir = os.path.dirname(log_file_path)
#         if log_dir and not os.path.exists(log_dir):
#             os.makedirs(log_dir, exist_ok=True)
#
#         # RotatingFileHandler for writing logs to a JSON file with rotation
#         file_handler = RotatingFileHandler(
#             log_file_path, maxBytes=10*1024*1024, backupCount=5  # 10 MB per file, keep 5 backups
#         )
#         file_formatter = jsonlogger.JsonFormatter(
#             '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
#         )
#         file_handler.setFormatter(file_formatter)
#         logger.addHandler(file_handler)
#
#     return logger
#
# # Initialize the logger
# logger = setup_logger()


#
# End of Functions
############################################################################################################
