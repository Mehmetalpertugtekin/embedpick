import pandas as pd
import pytest

from embedpick.metrics import (
    consensus_failures,
    reciprocal_rank,
    recall_at_k,
    score_all,
)


class TestRecallAtK:
    def test_hepsi_bulundu(self):
        assert recall_at_k([1, 2, 3], [1, 2], k=3) == 1.0

    def test_yarisi_bulundu(self):
        assert recall_at_k([1, 9, 8], [1, 2], k=3) == 0.5

    def test_hicbiri_bulunmadi(self):
        assert recall_at_k([7, 8, 9], [1, 2], k=3) == 0.0

    def test_k_disindakiler_sayilmaz(self):
        # Doğru cevap 4. sırada ama k=3, yani görünmüyor.
        assert recall_at_k([7, 8, 9, 1], [1], k=3) == 0.0

    def test_liste_k_dan_kisa_olabilir(self):
        assert recall_at_k([1], [1], k=5) == 1.0

    def test_tekrarli_sonuclar_iki_kez_sayilmaz(self):
        assert recall_at_k([1, 1, 1], [1, 2], k=3) == 0.5


class TestReciprocalRank:
    def test_birinci_sira(self):
        assert reciprocal_rank([1, 2, 3], [1], k=3) == 1.0

    def test_ikinci_sira(self):
        assert reciprocal_rank([9, 1, 3], [1], k=3) == 0.5

    def test_ucuncu_sira(self):
        assert reciprocal_rank([9, 8, 1], [1], k=3) == pytest.approx(1 / 3)

    def test_bulunamadi(self):
        assert reciprocal_rank([7, 8, 9], [1], k=3) == 0.0

    def test_ilk_isabet_sayilir(self):
        # 2 ve 3 doğru; ilki 1. sırada, MRR onu baz alır.
        assert reciprocal_rank([2, 3], [2, 3], k=2) == 1.0

    def test_k_disindaki_isabet_sayilmaz(self):
        assert reciprocal_rank([7, 8, 1], [1], k=2) == 0.0


class TestScoreAll:
    def test_sorgular_uzerinden_ortalama(self):
        queries = pd.DataFrame(
            {"query": ["a", "b"], "relevant_ids": [[1], [2]]}
        )
        # İlk sorgu 1. sırada bulundu, ikincisi hiç bulunamadı.
        recall, mrr = score_all([[1, 5], [8, 9]], queries, k=2)
        assert recall == 0.5
        assert mrr == 0.5


class TestConsensusFailures:
    def test_ortak_basarisizlik_yakalanir(self):
        skorlar = {
            "model_a": {"q1": 1.0, "q2": 0.0},
            "model_b": {"q1": 0.5, "q2": 0.0},
        }
        assert consensus_failures(skorlar) == ["q2"]

    def test_biri_bulduysa_isaretlenmez(self):
        skorlar = {
            "model_a": {"q1": 0.0},
            "model_b": {"q1": 0.5},
        }
        assert consensus_failures(skorlar) == []

    def test_bos_girdi(self):
        assert consensus_failures({}) == []
