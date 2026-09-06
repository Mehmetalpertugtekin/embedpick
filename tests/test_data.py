import pytest

from embedpick.data import load_corpus, load_queries, validate


def yaz(tmp_path, ad, icerik):
    p = tmp_path / ad
    p.write_text(icerik, encoding="utf-8")
    return str(p)


class TestLoadCorpus:
    def test_gecerli_dosya(self, tmp_path):
        yol = yaz(tmp_path, "c.csv", "id,text\n1,Kargom ulaşmadı\n2,İade istiyorum\n")
        df = load_corpus(yol)
        assert len(df) == 2
        assert df["text"].iloc[1] == "İade istiyorum"

    def test_bosluklar_kirpilir(self, tmp_path):
        yol = yaz(tmp_path, "c.csv", "id,text\n1,   boşluklu   \n")
        assert load_corpus(yol)["text"].iloc[0] == "boşluklu"

    def test_eksik_sutun_hata_verir(self, tmp_path):
        yol = yaz(tmp_path, "c.csv", "id,icerik\n1,merhaba\n")
        with pytest.raises(ValueError, match="eksik sütun"):
            load_corpus(yol)

    def test_tekrar_eden_id_hata_verir(self, tmp_path):
        yol = yaz(tmp_path, "c.csv", "id,text\n1,bir\n1,iki\n")
        with pytest.raises(ValueError, match="tekrar eden id"):
            load_corpus(yol)


class TestLoadQueries:
    def test_tek_id(self, tmp_path):
        yol = yaz(tmp_path, "q.csv", "query,relevant_ids\nkargo,3\n")
        assert load_queries(yol)["relevant_ids"].iloc[0] == [3]

    def test_noktali_virgulle_coklu_id(self, tmp_path):
        yol = yaz(tmp_path, "q.csv", "query,relevant_ids\nkargo,3;11;42\n")
        assert load_queries(yol)["relevant_ids"].iloc[0] == [3, 11, 42]

    def test_bosluklu_ayirma_calisir(self, tmp_path):
        yol = yaz(tmp_path, "q.csv", "query,relevant_ids\nkargo,3 ; 11\n")
        assert load_queries(yol)["relevant_ids"].iloc[0] == [3, 11]

    def test_sayi_olmayan_id_hata_verir(self, tmp_path):
        yol = yaz(tmp_path, "q.csv", "query,relevant_ids\nkargo,abc\n")
        with pytest.raises(ValueError, match="sayı olmalı"):
            load_queries(yol)

    def test_eksik_sutun_hata_verir(self, tmp_path):
        yol = yaz(tmp_path, "q.csv", "query,cevap\nkargo,3\n")
        with pytest.raises(ValueError, match="eksik sütun"):
            load_queries(yol)


class TestValidate:
    def test_gecerli_etiketler_gecer(self, tmp_path):
        c = load_corpus(yaz(tmp_path, "c.csv", "id,text\n1,bir\n2,iki\n"))
        q = load_queries(yaz(tmp_path, "q.csv", "query,relevant_ids\na,1;2\n"))
        validate(c, q)  # hata fırlatmamalı

    def test_olmayan_id_yakalanir(self, tmp_path):
        c = load_corpus(yaz(tmp_path, "c.csv", "id,text\n1,bir\n"))
        q = load_queries(yaz(tmp_path, "q.csv", "query,relevant_ids\nsorgum,999\n"))
        with pytest.raises(ValueError) as exc:
            validate(c, q)
        # Hata mesajı hangi sorgunun bozuk olduğunu söylemeli.
        assert "sorgum" in str(exc.value)
        assert "999" in str(exc.value)
