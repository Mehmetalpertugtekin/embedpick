"""embedpick komut satırı arayüzü."""

import argparse
import sys
from importlib import resources

from .benchmark import report, run_benchmark
from .data import load_corpus, load_queries, validate
from .retrievers import DEFAULT_MODELS


def sample_path(name):
    """Pakete gömülü örnek veri dosyasının yolu.

    Paket pip ile kurulduğunda çalışma dizininde data/ klasörü olmaz,
    o yüzden varsayılan veri kurulum dizininden okunuyor.
    """
    return str(resources.files("embedpick") / "sample_data" / name)


def build_parser():
    p = argparse.ArgumentParser(
        prog="embedpick",
        description=(
            "Kendi verinizde embedding modellerini kıyaslayın. "
            "Kalite, hız ve bellek maliyetini tek tabloda gösterir."
        ),
    )
    p.add_argument(
        "--corpus",
        help="Aranacak dokümanlar (sütunlar: id,text). "
        "Verilmezse pakete gömülü örnek veri kullanılır.",
    )
    p.add_argument(
        "--queries",
        help="Sorgular ve doğru cevaplar (sütunlar: query,relevant_ids). "
        "Verilmezse pakete gömülü örnek veri kullanılır.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Hugging Face model adları, boşlukla ayrılmış",
    )
    p.add_argument("-k", type=int, default=5, help="İlk kaç sonuç değerlendirilsin")
    p.add_argument(
        "--repeats", type=int, default=5, help="Hız ölçümü kaç kez tekrarlansın"
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="BM25 kelime bazlı baseline'ı atla (önerilmez)",
    )
    p.add_argument(
        "--no-presets",
        action="store_true",
        help=(
            "Bilinen model öneklerini uygulama. e5 gibi önek bekleyen modeller "
            "düşük skor alır; yanlış yapılandırmanın maliyetini ölçmek için."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Her sorgunun sonucunu tek tek yazdır",
    )
    p.add_argument("--out", help="Sonuç tablosunu CSV olarak kaydet")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    corpus_path = args.corpus or sample_path("corpus.csv")
    queries_path = args.queries or sample_path("queries.csv")

    if args.corpus is None and args.queries is None:
        print("Kendi veriniz verilmedi, pakete gömülü örnek veri kullanılıyor.")

    try:
        corpus = load_corpus(corpus_path)
        queries = load_queries(queries_path)
        validate(corpus, queries)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    print(f"{len(corpus)} doküman, {len(queries)} sorgu, k={args.k}")

    df, sorgu_bazli = run_benchmark(
        corpus,
        queries,
        model_names=args.models,
        k=args.k,
        repeats=args.repeats,
        include_baseline=not args.no_baseline,
        use_presets=not args.no_presets,
        verbose=args.verbose,
    )

    report(df, sorgu_bazli, args.k)

    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8")
        print(f"\nTablo kaydedildi: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
