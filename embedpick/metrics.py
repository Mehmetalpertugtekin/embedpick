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
