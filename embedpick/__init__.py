"""embedpick — kendi verinizde embedding modeli kıyaslama aracı."""

from .benchmark import run_benchmark
from .data import load_corpus, load_queries, validate

__version__ = "0.1.0"
__all__ = ["run_benchmark", "load_corpus", "load_queries", "validate"]
