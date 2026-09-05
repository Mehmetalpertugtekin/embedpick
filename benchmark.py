import re
import time

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/corpus.csv"
QUERIES_PATH = "data/queries.csv"
K = 5
REPEATS = 5

MODELS = [
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "doc_prefix": "",
    },
    {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    {
        "name": "trmteb/turkish-embedding-model",
        "query_prefix": "",
        "doc_prefix": "",
    },
]


def load_data():
    corpus = pd.read_csv(CORPUS_PATH, encoding="utf-8")
    queries = pd.read_csv(QUERIES_PATH, encoding="utf-8")
    queries["relevant_ids"] = queries["relevant_ids"].apply(
        lambda s: [int(x) for x in str(s).split(";")]
    )
    return corpus, queries


def recall_at_k(retrieved_ids, relevant_ids, k):
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids, k):
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def score_run(all_retrieved, queries, k):
    """Sorgu bazlı sonuçları basar ve ortalama metrikleri döndürür."""
    recalls = []
    rrs = []
    for q, retrieved_ids, relevant in zip(
        queries["query"], all_retrieved, queries["relevant_ids"]
    ):
        r = recall_at_k(retrieved_ids, relevant, k)
        rr = reciprocal_rank(retrieved_ids, relevant, k)
        recalls.append(r)
        rrs.append(rr)
        print(f"   [r={r:.2f} rr={rr:.2f}] {q}  -> {retrieved_ids}  | doğru: {relevant}")
    return float(np.mean(recalls)), float(np.mean(rrs))


def tr_tokenize(text):
    # Python'un lower() metodu Türkçe bilmez: "I" -> "i" yapar, oysa "ı" olmalı.
    # "İ" ise "i" artı ayrı bir nokta karakterine bölünür. Önce elle düzeltiyoruz.
    text = text.replace("I", "ı").replace("İ", "i")
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def evaluate_bm25(corpus, queries, k=K):
    print("\n-> BM25 (kelime bazlı baseline)")

    texts = corpus["text"].tolist()

    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        tokenized = [tr_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        timings.append(time.perf_counter() - start)

    index_sec = float(np.median(timings))

    ids = corpus["id"].tolist()
    all_retrieved = []
    for q in queries["query"]:
        scores = bm25.get_scores(tr_tokenize(q))
        top = np.argsort(-scores)[:k]
        all_retrieved.append([ids[i] for i in top])

    recall, mrr = score_run(all_retrieved, queries, k)

    return {
        "model": "BM25 (baseline)",
        f"recall@{k}": round(recall, 3),
        f"mrr@{k}": round(mrr, 3),
        "docs_per_sec": round(len(texts) / index_sec, 1),
        "dim": 0,
        "index_mb": 0.0,
    }


def evaluate(cfg, corpus, queries, k=K):
    model_name = cfg["name"]
    print(f"\n-> {model_name}")
    model = SentenceTransformer(model_name)

    corpus_texts = [cfg["doc_prefix"] + t for t in corpus["text"].tolist()]
    query_texts = [cfg["query_prefix"] + q for q in queries["query"].tolist()]

    # Isınma: tüm korpusu bir kez geçiyoruz. Küçük bir ısınma turu işlemciyi
    # düşük frekanstan çıkarmaya yetmiyor ve ilk ölçümler yanıltıcı yavaş çıkıyor.
    model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)

    # Birden fazla ölçüm alıp medyanı kullanıyoruz; ortalama tek bir tepe değerden bozulur.
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        corpus_emb = model.encode(
            corpus_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        timings.append(time.perf_counter() - start)

    encode_sec = float(np.median(timings))

    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(np.asarray(corpus_emb, dtype="float32"))

    query_emb = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _, positions = index.search(np.asarray(query_emb, dtype="float32"), k)

    ids = corpus["id"].tolist()
    all_retrieved = [[ids[p] for p in row] for row in positions]

    recall, mrr = score_run(all_retrieved, queries, k)

    return {
        "model": model_name.split("/")[-1],
        f"recall@{k}": round(recall, 3),
        f"mrr@{k}": round(mrr, 3),
        "docs_per_sec": round(len(corpus_texts) / encode_sec, 1),
        "dim": int(corpus_emb.shape[1]),
        "index_mb": round(corpus_emb.nbytes / (1024 * 1024), 2),
    }


def add_relative_speed(results):
    """Mutlak hız makineye ve o anki işlemci durumuna göre kat kat oynayabiliyor.
    Aynı çalıştırma içindeki oranlar ise sabit kalıyor, o yüzden göreli sütun ekliyoruz."""
    neural = [r for r in results if r["dim"] > 0]
    if not neural:
        return results

    fastest = max(r["docs_per_sec"] for r in neural)
    for r in results:
        if r["dim"] > 0:
            r["rel_speed"] = f"{r['docs_per_sec'] / fastest:.2f}x"
        else:
            r["rel_speed"] = "-"
    return results


def main():
    corpus, queries = load_data()
    print(f"{len(corpus)} doküman, {len(queries)} sorgu")

    results = [evaluate_bm25(corpus, queries)]
    results += [evaluate(cfg, corpus, queries) for cfg in MODELS]
    results = add_relative_speed(results)

    print()
    print(pd.DataFrame(results).to_string(index=False))
    print(
        "\nNot: docs_per_sec makineye ve o anki işlemci durumuna göre değişir."
        "\n     Makineler arası kıyas için rel_speed sütununu kullanın"
        "\n     (en hızlı sinirsel model = 1.00x)."
    )


if __name__ == "__main__":
    main()
