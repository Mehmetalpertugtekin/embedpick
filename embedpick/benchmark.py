"""Benchmark akışını yürütür ve sonuç tablosunu üretir."""

import pandas as pd

from .metrics import bootstrap_ci, consensus_failures, recall_at_k, score_all
from .retrievers import BM25Retriever, EmbeddingRetriever


def _significant_digits(value, digits=3):
    """Ölçümü taşıyabileceği hassasiyete yuvarlar.

    docs_per_sec çalıştırmalar arasında kat kat oynuyor; tabloya 226486.3
    yazmak, o sayının son hanesine güvendiğimizi iddia etmek demek.
    Bir sayının hassasiyeti, ona ne kadar güvendiğimiz hakkında bir iddiadır.
    """
    if value <= 0:
        return 0.0
    from math import floor, log10

    basamak = digits - 1 - int(floor(log10(abs(value))))
    return round(value, basamak)


def run_benchmark(
    corpus,
    queries,
    model_names,
    k=5,
    repeats=5,
    include_baseline=True,
    use_presets=True,
    verbose=False,
    stemmer="none",
):
    retrievers = []
    if include_baseline:
        retrievers.append(BM25Retriever(stemmer=stemmer))

    for name in model_names:
        if use_presets:
            retrievers.append(EmbeddingRetriever(name))
        else:
            # Önekler kapalı: modelin belgelenmiş kullanım kuralı yok sayılıyor.
            # Yanlış yapılandırmanın maliyetini ölçmek için.
            retrievers.append(EmbeddingRetriever(name, query_prefix="", doc_prefix=""))

    satirlar = []
    sorgu_bazli = {}

    for retriever in retrievers:
        print(f"\n-> {retriever.label}")
        maliyet = retriever.index(corpus, repeats)
        bulunanlar = retriever.search(queries["query"].tolist(), k)

        recall, mrr = score_all(bulunanlar, queries, k, verbose=verbose)

        sorgu_bazli[retriever.label] = {
            q: recall_at_k(bulunan, dogru, k)
            for q, bulunan, dogru in zip(
                queries["query"], bulunanlar, queries["relevant_ids"]
            )
        }

        alt, ust = bootstrap_ci(sorgu_bazli[retriever.label].values())

        satirlar.append(
            {
                "model": retriever.label,
                f"recall@{k}": round(recall, 3),
                f"recall@{k}_ci": f"[{alt:.2f}, {ust:.2f}]",
                f"mrr@{k}": round(mrr, 3),
                "docs_per_sec": _significant_digits(maliyet["docs_per_sec"]),
                "dim": maliyet["dim"],
                "index_mb": round(maliyet["index_mb"], 2),
            }
        )

    _add_relative_speed(satirlar)
    return pd.DataFrame(satirlar), sorgu_bazli


def _add_relative_speed(satirlar):
    """Mutlak hız makineye ve o anki işlemci durumuna göre kat kat oynar.
    Aynı çalıştırma içindeki oranlar ise sabit kalır."""
    sinirsel = [r for r in satirlar if r["dim"] > 0]
    if not sinirsel:
        return

    en_hizli = max(r["docs_per_sec"] for r in sinirsel)
    for r in satirlar:
        r["rel_speed"] = f"{r['docs_per_sec'] / en_hizli:.2f}x" if r["dim"] > 0 else "-"


def report(df, sorgu_bazli, k):
    print()
    print(df.to_string(index=False))

    print(
        "\nNot: docs_per_sec makineye ve o anki işlemci durumuna göre değişir."
        "\n     Makineler arası kıyas için rel_speed sütununu kullanın"
        "\n     (en hızlı sinirsel model = 1.00x)."
        f"\n     recall@{k}_ci, {len(next(iter(sorgu_bazli.values()), {}))} sorgu"
        " üzerinden bootstrap ile hesaplanmış yüzde 95 güven aralığıdır."
        "\n     Aralıklar üst üste biniyorsa aradaki fark bu veri setinde"
        " gösterilememiştir."
    )

    # Tek yöntem varken "hiçbiri bulamadı" demek anlamsız; uyarının değeri
    # birden fazla yöntemin aynı sorguda batmasından geliyor.
    ortak = consensus_failures(sorgu_bazli) if len(sorgu_bazli) > 1 else []
    if ortak:
        print(
            f"\nHiçbir yöntemin ilk {k} sonuçta bulamadığı sorgular:"
        )
        for q in ortak:
            print(f"  - {q}")
        print(
            "  Bütün yöntemler aynı sorguda batıyorsa sorun genelde modelde değil,"
            "\n  etiketlerdedir. Bu sorguların relevant_ids değerlerini gözden geçirin."
        )
