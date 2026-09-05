"""Benchmark akışını yürütür ve sonuç tablosunu üretir."""

import pandas as pd

from .metrics import consensus_failures, recall_at_k, score_all
from .retrievers import BM25Retriever, EmbeddingRetriever


def run_benchmark(
    corpus,
    queries,
    model_names,
    k=5,
    repeats=5,
    include_baseline=True,
    verbose=False,
):
    retrievers = []
    if include_baseline:
        retrievers.append(BM25Retriever())
    retrievers += [EmbeddingRetriever(name) for name in model_names]

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

        satirlar.append(
            {
                "model": retriever.label,
                f"recall@{k}": round(recall, 3),
                f"mrr@{k}": round(mrr, 3),
                "docs_per_sec": round(maliyet["docs_per_sec"], 1),
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
    )

    ortak = consensus_failures(sorgu_bazli)
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
