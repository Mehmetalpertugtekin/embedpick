"""Benchmark akışını yürütür ve sonuç tablosunu üretir."""

import pandas as pd

from .metrics import (
    bootstrap_ci,
    consensus_failures,
    recall_at_k,
    score_all,
    significant_difference,
)
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
    # Doğrulama burada, CLI'da değil: Gradio arayüzü ve kütüphaneyi doğrudan
    # çağıran kod da aynı kapıdan geçiyor. repeats=0 ölçüm döngüsünü hiç
    # döndürmüyor ve sonrasında UnboundLocalError veriyordu; k=0 ise sessizce
    # boş sonuç listeleri üretip tüm skorları 0 gösteriyor ki bu daha kötü.
    if repeats < 1:
        raise ValueError(f"repeats en az 1 olmalı, verilen: {repeats}")
    if k < 1:
        raise ValueError(f"k en az 1 olmalı, verilen: {k}")

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


def baseline_comparison(sorgu_bazli):
    """Her modeli BM25 baseline'ı ile eşleştirilmiş olarak kıyaslar.

    Sütun bazlı güven aralıkları her yöntemin tek başına ne kadar oynadığını
    gösterir ve üst üste binmeleri "fark yok" anlamına GELMEZ: aynı sorgular
    üzerinde çalıştıkları için farkın belirsizliği, tek tek skorların
    belirsizliğinden küçüktür. Karar vermek isteyen kişinin bakması gereken
    sayı bu fonksiyonun döndürdüğü.
    """
    baseline = next((e for e in sorgu_bazli if e.startswith("BM25")), None)
    if baseline is None or len(sorgu_bazli) < 2:
        return None, []

    sorgular = list(sorgu_bazli[baseline])
    temel = [sorgu_bazli[baseline][q] for q in sorgular]

    sonuclar = [
        (etiket, significant_difference([skor[q] for q in sorgular], temel))
        for etiket, skor in sorgu_bazli.items()
        if etiket != baseline
    ]
    return baseline, sonuclar


def _report_baseline_comparison(sorgu_bazli):
    baseline, sonuclar = baseline_comparison(sorgu_bazli)
    if not sonuclar:
        return

    print(f"\n{baseline} ile eşleştirilmiş fark (yüzde 95 güven aralığı):")
    for etiket, s in sonuclar:
        alt, ust = s["ci"]
        karar = "anlamlı" if s["anlamli"] else "gösterilemedi"
        print(f"  {etiket:<40} {s['fark']:+.3f}  [{alt:+.3f}, {ust:+.3f}]  {karar}")

    if any(not s["anlamli"] for _, s in sonuclar):
        print(
            "  'gösterilemedi', farkın olmadığı anlamına gelmez; bu veri setinin"
            "\n  o farkı göstermeye yetmediği anlamına gelir. Daha çok sorgu ekleyin."
        )


def report(df, sorgu_bazli, k):
    print()
    print(df.to_string(index=False))

    print(
        "\nNot: docs_per_sec makineye ve o anki işlemci durumuna göre değişir."
        "\n     Makineler arası kıyas için rel_speed sütununu kullanın"
        "\n     (en hızlı sinirsel model = 1.00x)."
        f"\n     recall@{k}_ci, {len(next(iter(sorgu_bazli.values()), {}))} sorgu"
        " üzerinden bootstrap ile hesaplanmış yüzde 95 güven aralığıdır."
        "\n     Her sütun tek başına ne kadar oynadığını gösterir; iki yöntemin"
        "\n     FARKI için aşağıdaki eşleştirilmiş karşılaştırmaya bakın."
        "\n     Üst üste binen aralıklar tek başına 'fark yok' demek değildir."
    )

    _report_baseline_comparison(sorgu_bazli)

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
