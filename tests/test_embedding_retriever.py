"""EmbeddingRetriever testleri.

Gerçek model indirilmiyor: encoder dışarıdan verilebildiği için sahte bir
encoder geçiriyoruz. Böylece önek uygulaması, id eşlemesi ve k sınırı
saniyeler içinde, ağ erişimi olmadan doğrulanabiliyor.
"""

import numpy as np
import pandas as pd
import pytest

from embedpick.retrievers import BM25Retriever, EmbeddingRetriever

faiss = pytest.importorskip("faiss")


SOZLUK = ["kargo", "iade", "şifre", "fatura"]


class SahteEncoder:
    """Kelime varlığına bakan deterministik encoder.

    Gerçek bir modeli taklit etmiyor; sadece aynı kelimeyi içeren metinleri
    birbirine yakın yerleştiriyor, böylece sıralama iddiaları anlamlı oluyor.
    Ayrıca kendisine verilen metinleri saklıyor ki öneklerin gerçekten
    uygulandığını test edebilelim.
    """

    def __init__(self):
        self.gorulen_metinler = []

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        self.gorulen_metinler.extend(texts)

        vektorler = []
        for t in texts:
            dusuk = t.lower()
            v = np.array([1.0 if kelime in dusuk else 0.0 for kelime in SOZLUK])
            if not v.any():
                # Hiçbir anahtar kelime yoksa sıfır vektör kosinüsü tanımsız
                # yapar; sabit bir yöne koyuyoruz.
                v = np.ones(len(SOZLUK))
            if normalize_embeddings:
                v = v / np.linalg.norm(v)
            vektorler.append(v)

        return np.asarray(vektorler, dtype="float32")


def sahte_fabrika(encoder):
    return lambda model_name: encoder


def kucuk_korpus():
    return pd.DataFrame(
        {
            "id": [10, 20, 30],
            "text": [
                "Kargom hala elime ulaşmadı",
                "İade sürecini nasıl başlatabilirim",
                "Şifremi unuttum giriş yapamıyorum",
            ],
        }
    )


def kur(query_prefix="", doc_prefix=""):
    encoder = SahteEncoder()
    r = EmbeddingRetriever(
        "sahte/model",
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
        encoder_factory=sahte_fabrika(encoder),
    )
    r.index(kucuk_korpus(), repeats=1)
    return r, encoder


class TestKSinirlamasi:
    def test_k_korpustan_buyukse_uydurma_sonuc_donmez(self):
        # FAISS dolduramadığı yerlere -1 koyar ve self._ids[-1] hata vermeden
        # son id'yi döndürür. Düzeltme öncesi bu test 3 yerine 5 sonuç ve
        # tekrar eden id'ler görüyordu.
        r, _ = kur()
        bulunan = r.search(["kargo"], k=10)[0]

        assert len(bulunan) == 3
        assert sorted(bulunan) == [10, 20, 30]

    def test_tekrar_eden_id_yok(self):
        r, _ = kur()
        bulunan = r.search(["iade"], k=10)[0]
        assert len(bulunan) == len(set(bulunan))

    def test_k_korpustan_kucukse_tam_k_doner(self):
        r, _ = kur()
        assert len(r.search(["kargo"], k=2)[0]) == 2

    def test_baseline_ile_ayni_sayida_sonuc(self):
        # Aracın tüm amacı iki yöntemi kıyaslamak; aynı k'da farklı sayıda
        # sonuç döndürmeleri hatanın kendisinden daha kötü.
        r, _ = kur()
        bm25 = BM25Retriever()
        bm25.index(kucuk_korpus(), repeats=1)

        assert len(r.search(["kargo"], k=10)[0]) == len(bm25.search(["kargo"], k=10)[0])


class TestIdEslemesi:
    def test_pozisyon_degil_gercek_id_doner(self):
        r, _ = kur()
        bulunan = r.search(["şifre"], k=3)[0]
        assert set(bulunan) <= {10, 20, 30}

    def test_eslesen_dokuman_ilk_sirada(self):
        r, _ = kur()
        assert r.search(["şifre"], k=3)[0][0] == 30


class TestOnekler:
    def test_dokuman_oneki_uygulanir(self):
        _, encoder = kur(doc_prefix="passage: ")
        assert all(t.startswith("passage: ") for t in encoder.gorulen_metinler)

    def test_sorgu_oneki_uygulanir(self):
        r, encoder = kur(query_prefix="query: ")
        encoder.gorulen_metinler.clear()
        r.search(["kargo nerede"], k=3)
        assert encoder.gorulen_metinler == ["query: kargo nerede"]

    def test_onek_verilmezse_metin_degismez(self):
        r, encoder = kur()
        encoder.gorulen_metinler.clear()
        r.search(["kargo nerede"], k=3)
        assert encoder.gorulen_metinler == ["kargo nerede"]


class TestMaliyet:
    def test_maliyet_alanlari_dolu_doner(self):
        encoder = SahteEncoder()
        r = EmbeddingRetriever("sahte/model", encoder_factory=sahte_fabrika(encoder))
        maliyet = r.index(kucuk_korpus(), repeats=1)

        assert maliyet["dim"] == len(SOZLUK)
        assert maliyet["docs_per_sec"] > 0
        assert maliyet["index_mb"] > 0
