from ..logger import logger
from .config_manager import ConfigManager

cfg = ConfigManager()
logger.setLevel(cfg.general.log_level)

__all__ = ["ConfigManager", "cfg"]
