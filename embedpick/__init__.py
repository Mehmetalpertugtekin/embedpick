"""embedpick — kendi verinizde embedding modeli kıyaslama aracı."""

from importlib.metadata import PackageNotFoundError, version

from .benchmark import run_benchmark
from .data import load_corpus, load_queries, validate

try:
    # Tek kaynak pyproject.toml. Sürümü burada ikinci kez yazmak, iki yerin
    # sessizce ayrı düşmesi demekti: paket 0.1.1 yayınlanmışken __init__
    # hâlâ 0.1.0 diyordu.
    __version__ = version("embedpick")
except PackageNotFoundError:  # pragma: no cover - sadece kurulum yokken
    # Depodan kurulmadan çalıştırılıyor (örn. repo kökünden import).
    __version__ = "0.0.0+unknown"
__all__ = ["run_benchmark", "load_corpus", "load_queries", "validate"]
