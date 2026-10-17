"""Arama yöntemleri: kelime bazlı BM25 baseline ve embedding tabanlı arama."""

import re
import time

import numpy as np

from .stemmers import get_stemmer

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


def tr_lower(word):
    """Tek kelimeyi Türkçe kurallarına göre küçültür.

    Python'un lower() metodu Türkçe bilmez: "I" -> "i" yapar, oysa "ı" olmalı.
    "İ" ise "i" artı ayrı bir birleşen nokta karakterine bölünür.

    Ama kuralı koşulsuz uygularsak "IBAN" -> "ıban", "ISO" -> "ıso" olur ve
    Türkçe destek korpusları İngilizce kısaltma doludur. Tamamı büyük harf
    ve tamamı ASCII olan kelimeleri kısaltma sayıp düz lower() uyguluyoruz;
    "IŞIK" gibi Türkçe harf içerenler Türkçe kuralına gidiyor.
    """
    if word.isupper() and word.isascii():
        return word.lower()
    return word.replace("I", "ı").replace("İ", "i").lower()


def tr_tokenize(text, stemmer=None):
    """Türkçe farkındalıklı basit kelime ayırıcı.

    stemmer verilirse her terim gövdelenir; bkz. embedpick.stemmers.
    """
    tokens = (tr_lower(t) for t in re.findall(r"\w+", text, flags=re.UNICODE))
    if stemmer is None:
        return list(tokens)
    return [stemmer(t) for t in tokens]


class BM25Retriever:
    """Kelime bazlı baseline. Embedding modelleri bunu geçemiyorsa
    harcanan hesap gücü boşa gidiyor demektir."""

    dim = 0

    def __init__(self, stemmer="none"):
        self.stemmer_name = stemmer
        self._stemmer = get_stemmer(stemmer)
        self.label = (
            "BM25 (baseline)"
            if stemmer == "none"
            else f"BM25 (baseline, {stemmer} stem)"
        )
        self._bm25 = None
        self._ids = None

    def _tokenize(self, text):
        return tr_tokenize(text, self._stemmer)

    def index(self, corpus, repeats):
        from rank_bm25 import BM25Okapi

        texts = corpus["text"].tolist()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            tokenized = [self._tokenize(t) for t in texts]
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
            scores = self._bm25.get_scores(self._tokenize(q))
            top = np.argsort(-scores)[:k]
            sonuclar.append([self._ids[i] for i in top])
        return sonuclar


def _default_encoder(model_name):
    # Import burada: sentence-transformers ağır bir paket, sadece gerçekten
    # bir model yüklenecekse maliyetini ödüyoruz.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingRetriever:
    """sentence-transformers modeli + FAISS iç çarpım indeksi.

    Vektörler normalize edildiği için iç çarpım = kosinüs benzerliği.
    """

    def __init__(
        self, model_name, query_prefix=None, doc_prefix=None, encoder_factory=None
    ):
        self.model_name = model_name
        self.label = model_name.split("/")[-1]

        onekler = get_prefixes(model_name)
        self.query_prefix = onekler["query"] if query_prefix is None else query_prefix
        self.doc_prefix = onekler["doc"] if doc_prefix is None else doc_prefix

        # Model nesnesini dışarıdan verilebilir yaptık. Varsayılan davranış
        # değişmiyor; testler sahte bir encoder geçirerek önekleri, id
        # eşlemesini ve k sınırını model indirmeden doğrulayabiliyor.
        self._encoder_factory = encoder_factory or _default_encoder

        self._model = None
        self._index = None
        self._ids = None
        self.dim = 0

    def index(self, corpus, repeats):
        self._model = self._encoder_factory(self.model_name)
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
        # FAISS, isteneni dolduramadığı yerlere -1 koyar. self._ids[-1]
        # hata vermez, listenin son elemanını döndürür: 5 dokümanlık bir
        # korpusta k=10 istendiğinde tablo uydurma isabetlerle şişer ve
        # hiçbir uyarı çıkmaz. Ölçüm aracında sessiz yanlış, çökmekten
        # daha kötüdür. BM25 tarafı zaten korpustan fazlasını döndürmüyor;
        # iki yöntem aynı k'da aynı davranmalı.
        k_eff = min(k, self._index.ntotal)
        texts = [self.query_prefix + q for q in queries]
        emb = self._encode(texts)
        _, positions = self._index.search(np.asarray(emb, dtype="float32"), k_eff)
        return [[self._ids[p] for p in row if p != -1] for row in positions]

    def _encode(self, texts):
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
