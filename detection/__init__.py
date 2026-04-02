"""Detection module — burst detection and Zadoff-Chu correlation."""

from .burst_detector import BurstDetector, BurstSegment
from .zc_correlator import ZadoffChuCorrelator

__all__ = ["BurstDetector", "BurstSegment", "ZadoffChuCorrelator"]
