import pandas as pd

from embedpick.retrievers import BM25Retriever, get_prefixes, tr_tokenize


class TestTurkishTokenizer:
    def test_noktali_buyuk_I_dogru_kucultulur(self):
        # Python'un lower()'ı "İ" -> "i" + ayrı nokta karakteri üretir.
        assert tr_tokenize("İade") == ["iade"]

    def test_noktasiz_buyuk_I_dogru_kucultulur(self):
        # Python'un lower()'ı "I" -> "i" yapar, Türkçe'de "ı" olmalı.
        assert tr_tokenize("IŞIK") == ["ışık"]

    def test_karisik_kelime(self):
        assert tr_tokenize("Iğdır ve İstanbul") == ["ığdır", "ve", "istanbul"]

    def test_noktalama_ayirir(self):
        assert tr_tokenize("kargo, gelmedi!") == ["kargo", "gelmedi"]

    def test_turkce_harfler_korunur(self):
        assert tr_tokenize("çğöşü") == ["çğöşü"]

    def test_bos_metin(self):
        assert tr_tokenize("") == []


class TestPrefixPresets:
    def test_bilinen_e5_modeli_onek_alir(self):
        onek = get_prefixes("intfloat/multilingual-e5-small")
        assert onek["query"] == "query: "
        assert onek["doc"] == "passage: "

    def test_bilinmeyen_model_onek_almaz(self):
        onek = get_prefixes("sentence-transformers/all-MiniLM-L6-v2")
        assert onek == {"query": "", "doc": ""}

    def test_tanimsiz_e5_varyanti_uyari_basar(self, capsys):
        get_prefixes("someone/custom-e5-model")
        assert "uyarı" in capsys.readouterr().out


class TestBM25Retriever:
    def kur(self):
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
        r = BM25Retriever()
        maliyet = r.index(corpus, repeats=1)
        return r, maliyet

    def test_kelime_eslesmesi_birinci_sirada(self):
        r, _ = self.kur()
        assert r.search(["kargo ulaşmadı"], k=3)[0][0] == 10

    def test_buyuk_harfli_sorgu_da_eslesir(self):
        # "İade" korpusta büyük İ ile yazılı; sorgu küçük harfli.
        r, _ = self.kur()
        assert r.search(["iade"], k=3)[0][0] == 20

    def test_gercek_id_dondurur_pozisyon_degil(self):
        # FAISS ve BM25 pozisyon döndürür; id'ye çevirmeyi unutmak
        # sessiz ve yakalanması zor bir hatadır.
        r, _ = self.kur()
        assert set(r.search(["şifre"], k=3)[0]) <= {10, 20, 30}

    def test_k_kadar_sonuc_doner(self):
        r, _ = self.kur()
        assert len(r.search(["kargo"], k=2)[0]) == 2

    def test_baseline_maliyeti_sifir_bellek(self):
        _, maliyet = self.kur()
        assert maliyet["dim"] == 0
        assert maliyet["index_mb"] == 0.0
        assert maliyet["docs_per_sec"] > 0
