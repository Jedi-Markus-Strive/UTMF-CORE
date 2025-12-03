"""
UTMF-Core: Unified Temporal-Measurement Framework

Core package for multifractal measurement using a refined MFDFA implementation.
See Eversdijk (2025), "UTMF-Core: A Unified Temporal-Measurement Framework
for Heterogeneous Physical Time Series".
"""

from .core import run_utmf_core, build_utmf_summary, print_utmf_summary

__all__ = [
    "run_utmf_core",
    "build_utmf_summary",
    "print_utmf_summary",
]
