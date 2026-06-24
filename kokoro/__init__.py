__version__ = "0.9.4"

from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO",
)
logger.disable("kokoro")

from .model import (
    KModel,
    KokoroAcousticVocoder,
    KokoroInferenceBackend,
    KokoroTextDuration,
    expand_token_features,
)
from .pipeline import KPipeline
