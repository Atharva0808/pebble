"""
PEBBLE — A Mamba-2 Small Language Model built from scratch.

This package contains the complete architecture, training pipeline,
and inference utilities for the Pebble SLM.
"""

from .config import PebbleConfig
from .model import PebbleLMHeadModel

__version__ = "0.1.0"
__all__ = ["PebbleConfig", "PebbleLMHeadModel"]
