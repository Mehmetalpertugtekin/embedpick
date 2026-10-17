"""Gövdeleyici ve Türkçe küçültme testleri."""

import pytest

from embedpick.retrievers import BM25Retriever, tr_lower, tr_tokenize
from embedpick.stemmers import get_stemmer, simple


class TestSimpleStemmer:
    @pytest.mark.parametrize(
        "cekimli, yalin",
        [
            ("iadesi", "iade"),
            ("kargom", "kargo"),
            ("kargonun", "kargo"),
            ("şifremi", "şifre"),
            ("ürünleri", "ürün"),
            ("ürünler", "ürün"),
            ("faturamı", "fatura"),
        ],
    )
    def test_ayni_kokten_gelenler_ayni_terime_iner(self, cekimli, yalin):
        # Sözleşme bu: dilbilgisel olarak doğru gövdeyi bulmak değil, iki
        # yüzey biçimini BM25'in aynı terim sayacağı ortak bir dizeye
        # indirmek. "şifre" ve "şifremi" ikisi de "şifr" oluyor; fazla
        # soyuyor ama tutarlı soyuyor, erişim açısından istenen de bu.
        assert simple(cekimli) == simple(yalin)

    def test_ilgisiz_kelimeler_birbirine_karismaz(self):
        # Fazla soymanın sınırı: farklı kökler aynı dizeye düşmemeli.
        assert simple("kargo") != simple("şifre")
        assert simple("iade") != simple("fatura")

    def test_kisa_kelime_erimez(self):
        # Soyma gövdeyi 4 harften kısa bırakacaksa yapılmıyor.
        assert simple("iade") == "iade"
        assert simple("ev") == "ev"

    def test_eki_olmayan_kelime_degismez(self):
        assert simple("kargo") == "kargo"


class TestTurkceKucultme:
    def test_noktasiz_buyuk_I_turkce_kurala_gider(self):
        assert tr_lower("IŞIK") == "ışık"

    def test_noktali_buyuk_I(self):
        assert tr_lower("İade") == "iade"

    def test_ingilizce_kisaltma_bozulmaz(self):
        # Koşulsuz "I" -> "ı" kuralı IBAN'ı ıban yapıyordu; Türkçe destek
        # korpusları İngilizce kısaltma dolu.
        assert tr_lower("IBAN") == "iban"
        assert tr_lower("ISO") == "iso"

    def test_karisik_cumle(self):
        assert tr_tokenize("Iğdır ve İstanbul") == ["ığdır", "ve", "istanbul"]


class TestTokenizerStemmerIle:
    def test_stemmer_verilmezse_davranis_degismez(self):
        assert tr_tokenize("kargom gelmedi") == ["kargom", "gelmedi"]

    def test_stemmer_verilirse_terimler_govdelenir(self):
        stem = get_stemmer("simple")
        assert tr_tokenize("Kargom hala gelmedi", stem)[0] == "kargo"


class TestGetStemmer:
    def test_none_kelimeyi_degistirmez(self):
        assert get_stemmer("none")("iadesi") == "iadesi"

    def test_bilinmeyen_isim_hata_verir(self):
        with pytest.raises(ValueError, match="bilinmeyen stemmer"):
            get_stemmer("turkish-magic")


class TestBM25StemmerEntegrasyonu:
    def kur(self, stemmer):
        import pandas as pd

        corpus = pd.DataFrame(
            {
                "id": [10, 20, 30],
                "text": [
                    "Kargom hala elime ulaşmadı",
                    "İade sürecini nasıl başlatabilirim",
                    "Şifremi unuttum giriş yapamıyorum",
                ],
            }
        )
        r = BM25Retriever(stemmer=stemmer)
        r.index(corpus, repeats=1)
        return r

    def test_govdelemesiz_baseline_cekimli_sorguyu_kaciriyor(self):
        # "iadesi" korpusta yok, "İade" var. Gövdeleme olmadan BM25 için
        # bunlar ilgisiz iki terim; baseline'ın zayıf görünmesinin sebebi bu.
        r = self.kur("none")
        assert r._bm25.get_scores(r._tokenize("iadesi")).max() == 0

    def test_govdeleme_cekimli_sorguyu_yakaliyor(self):
        r = self.kur("simple")
        assert r.search(["iadesi"], k=3)[0][0] == 20

    def test_etiket_stemmeri_gosteriyor(self):
        # Tabloda hangi baseline'ın ölçüldüğü görünmeli.
        assert self.kur("none").label == "BM25 (baseline)"
        assert self.kur("simple").label == "BM25 (baseline, simple stem)"
