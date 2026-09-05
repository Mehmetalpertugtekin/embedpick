"""Arama yöntemleri: kelime bazlı BM25 baseline ve embedding tabanlı arama."""

import re
import time

import numpy as np

# Bazı modeller metinlerin başına önek bekler. Önek konmazsa model kötü
# çalışır ve kullanıcı bunu "model kötüymüş" diye yorumlar. Bilinenleri
# burada tutuyoruz ki varsayılan davranış doğru olsun.
MODEL_PRESETS = {
    "intfloat/multilingual-e5-small": {"query": "query: ", "doc": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "doc": "passage: "},
    "intfloat/multilingual-e5-large": {"query": "query: ", "doc": "passage: "},
    "BAAI/bge-m3": {"query": "", "doc": ""},
}

DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "trmteb/turkish-embedding-model",
]


def get_prefixes(model_name):
    if model_name in MODEL_PRESETS:
        return MODEL_PRESETS[model_name]

    # Tanımadığımız bir e5 varyantı gelirse yine de uyaralım.
    if "e5" in model_name.lower():
        print(
            f"   uyarı: {model_name} bir e5 varyantı gibi görünüyor. "
            "e5 modelleri 'query: ' / 'passage: ' öneki bekler, "
            "önek olmadan skorlar düşük çıkar."
        )
    return {"query": "", "doc": ""}


def tr_tokenize(text):
    """Türkçe farkındalıklı basit kelime ayırıcı.

    Python'un lower() metodu Türkçe bilmez: "I" -> "i" yapar, oysa "ı" olmalı.
    "İ" ise "i" artı ayrı bir birleşen nokta karakterine bölünür.
    """
    text = text.replace("I", "ı").replace("İ", "i")
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    """Kelime bazlı baseline. Embedding modelleri bunu geçemiyorsa
    harcanan hesap gücü boşa gidiyor demektir."""

    label = "BM25 (baseline)"
    dim = 0

    def __init__(self):
        self._bm25 = None
        self._ids = None

    def index(self, corpus, repeats):
        from rank_bm25 import BM25Okapi

        texts = corpus["text"].tolist()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            tokenized = [tr_tokenize(t) for t in texts]
            bm25 = BM25Okapi(tokenized)
            timings.append(time.perf_counter() - start)

        self._bm25 = bm25
        self._ids = corpus["id"].tolist()

        index_sec = float(np.median(timings))
        return {
            "docs_per_sec": len(texts) / index_sec,
            "index_mb": 0.0,
            "dim": 0,
        }

    def search(self, queries, k):
        sonuclar = []
        for q in queries:
            scores = self._bm25.get_scores(tr_tokenize(q))
            top = np.argsort(-scores)[:k]
            sonuclar.append([self._ids[i] for i in top])
        return sonuclar


class EmbeddingRetriever:
    """sentence-transformers modeli + FAISS iç çarpım indeksi.

    Vektörler normalize edildiği için iç çarpım = kosinüs benzerliği.
    """

    def __init__(self, model_name, query_prefix=None, doc_prefix=None):
        self.model_name = model_name
        self.label = model_name.split("/")[-1]

        onekler = get_prefixes(model_name)
        self.query_prefix = onekler["query"] if query_prefix is None else query_prefix
        self.doc_prefix = onekler["doc"] if doc_prefix is None else doc_prefix

        self._model = None
        self._index = None
        self._ids = None
        self.dim = 0

    def index(self, corpus, repeats):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        texts = [self.doc_prefix + t for t in corpus["text"].tolist()]

        # Isınma: tüm korpusu bir kez geçiyoruz. Kısa bir ısınma turu
        # işlemciyi düşük frekanstan çıkarmaya yetmiyor ve ilk ölçümler
        # yanıltıcı derecede yavaş çıkıyor.
        self._encode(texts)

        # Medyan alıyoruz; ortalama tek bir tepe değerden bozulur.
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            emb = self._encode(texts)
            timings.append(time.perf_counter() - start)

        import faiss

        self.dim = int(emb.shape[1])
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(np.asarray(emb, dtype="float32"))
        self._ids = corpus["id"].tolist()

        encode_sec = float(np.median(timings))
        return {
            "docs_per_sec": len(texts) / encode_sec,
            "index_mb": emb.nbytes / (1024 * 1024),
            "dim": self.dim,
        }

    def search(self, queries, k):
        texts = [self.query_prefix + q for q in queries]
        emb = self._encode(texts)
        _, positions = self._index.search(np.asarray(emb, dtype="float32"), k)
        return [[self._ids[p] for p in row] for row in positions]

    def _encode(self, texts):
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
