from .constants import BASE_LOGS_DIR


def get_logger_config(subdir: str = "") -> dict:

    # assert subdir is not None, subdir

    log_dir = BASE_LOGS_DIR / subdir if subdir else BASE_LOGS_DIR
    log_file = log_dir / f"app.log"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "file": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": log_file.as_posix(),
                "when": "midnight",
                "interval": 1,
                "backupCount": 7,
                "encoding": "utf-8",
                # 'suffix': '%Y-%m-%d',
            },
            "console": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": True,
            },
        },
    }

    return logging_config
