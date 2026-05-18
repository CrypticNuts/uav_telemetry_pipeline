"""Preprocessor module — IQ data loading and conversion.

Only ``shift_and_filter`` is exported eagerly because it's used by the
runner. ``csv_to_iq`` and ``load_verified_iq`` have heavier optional
dependencies (pandas; sibling ``telemetry`` package); callers that want
them should import directly from the submodule.
"""

from .shift_and_filter import shift_and_filter

__all__ = ["shift_and_filter"]
