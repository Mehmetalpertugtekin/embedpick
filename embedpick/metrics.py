"""Erişim metrikleri: recall@k ve MRR@k."""

import numpy as np


def recall_at_k(retrieved_ids, relevant_ids, k):
    """İlk k sonuçta doğru dokümanların ne kadarı yakalandı."""
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids, k):
    """İlk doğru sonucun sırasının tersi. 1. sıra -> 1.0, 2. sıra -> 0.5."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def score_all(all_retrieved, queries, k, verbose=False):
    """Tüm sorguları puanlar, ortalama recall ve MRR döndürür."""
    recalls = []
    rrs = []

    for q, retrieved_ids, relevant in zip(
        queries["query"], all_retrieved, queries["relevant_ids"]
    ):
        r = recall_at_k(retrieved_ids, relevant, k)
        rr = reciprocal_rank(retrieved_ids, relevant, k)
        recalls.append(r)
        rrs.append(rr)
        if verbose:
            print(
                f"   [r={r:.2f} rr={rr:.2f}] {q}"
                f"  -> {retrieved_ids}  | doğru: {relevant}"
            )

    return float(np.mean(recalls)), float(np.mean(rrs))


def bootstrap_ci(scores, n=2000, seed=0, alpha=0.05):
    """Ortalamanın yüzde 95 güven aralığı, bootstrap ile.

    26 sorguluk bir veri setinde 0.692 ile 0.750 arasındaki fark yaklaşık
    1.5 sorgu demek; bu, gürültüden ayırt edilemez. Tabloya aralık koymadan
    iki modeli kıyaslamak, olmayan bir farkı varmış gibi raporlamaktır.

    seed sabit: aynı veride aynı aralığı üretsin, tablo çalıştırmadan
    çalıştırmaya oynamasın.
    """
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    orneklemler = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
    alt = float(np.percentile(orneklemler, 100 * alpha / 2))
    ust = float(np.percentile(orneklemler, 100 * (1 - alpha / 2)))
    return alt, ust


def significant_difference(scores_a, scores_b, n=2000, seed=0, alpha=0.05):
    """İki yöntemin farkı sıfırdan ayırt edilebiliyor mu.

    Eşleştirilmiş bootstrap: aynı sorgular üzerinde fark alınıyor, çünkü
    sorguların zorluğu ortak. Aralık sıfırı içeriyorsa fark bu veri setinde
    gösterilememiş demektir — yokluğunun kanıtı değil, varlığının kanıtsızlığı.
    """
    fark = np.asarray(list(scores_a), dtype=float) - np.asarray(
        list(scores_b), dtype=float
    )
    alt, ust = bootstrap_ci(fark, n=n, seed=seed, alpha=alpha)
    return {"fark": float(fark.mean()), "ci": (alt, ust), "anlamli": alt > 0 or ust < 0}


def consensus_failures(per_model_scores, threshold=0.0):
    """Hiçbir modelin bulamadığı sorguları döndürür.

    Bütün modeller aynı sorguda batıyorsa suçlu genelde model değil, etikettir.
    """
    if not per_model_scores:
        return []

    sorgular = list(per_model_scores.values())[0].keys()
    return [
        q
        for q in sorgular
        if all(skorlar[q] <= threshold for skorlar in per_model_scores.values())
    ]
