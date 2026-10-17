"""Bootstrap güven aralığı testleri."""

import numpy as np

from embedpick.benchmark import _significant_digits
from embedpick.metrics import bootstrap_ci, significant_difference


class TestBootstrapCI:
    def test_aralik_ortalamayi_icerir(self):
        skorlar = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
        alt, ust = bootstrap_ci(skorlar)
        assert alt <= np.mean(skorlar) <= ust

    def test_sabit_skorlarda_aralik_sifir_genislikte(self):
        alt, ust = bootstrap_ci([1.0] * 20)
        assert alt == ust == 1.0

    def test_kucuk_orneklem_genis_aralik_verir(self):
        # 26 sorguda 0.058'lik bir fark gürültüden ayırt edilemez; aralığın
        # örneklem küçüldükçe genişlemesi tam olarak bunu gösteriyor.
        az = bootstrap_ci([1, 0] * 5)
        cok = bootstrap_ci([1, 0] * 100)
        assert (az[1] - az[0]) > (cok[1] - cok[0])

    def test_ayni_seed_ayni_sonuc(self):
        skorlar = [1, 0, 1, 1, 0, 1, 0, 0]
        assert bootstrap_ci(skorlar) == bootstrap_ci(skorlar)

    def test_bos_girdi_nan_doner(self):
        alt, ust = bootstrap_ci([])
        assert np.isnan(alt) and np.isnan(ust)


class TestSignificantDifference:
    def test_belirgin_fark_anlamli_bulunur(self):
        sonuc = significant_difference([1] * 30, [0] * 30)
        assert sonuc["anlamli"]
        assert sonuc["fark"] == 1.0

    def test_kucuk_fark_anlamli_bulunmaz(self):
        # 26 sorguda ~1.5 sorguluk fark: aralık sıfırı içermeli.
        a = [1] * 14 + [0] * 12
        b = [1] * 12 + [0] * 14
        sonuc = significant_difference(a, b)
        assert not sonuc["anlamli"]
        assert sonuc["ci"][0] <= 0 <= sonuc["ci"][1]


class TestAnlamliBasamak:
    def test_buyuk_sayi_kisaltilir(self):
        # 226486.3 yazmak son hanesine güvendiğimizi iddia etmek demek.
        assert _significant_digits(226486.3) == 226000.0

    def test_kucuk_sayi_korunur(self):
        assert _significant_digits(1.2345) == 1.23

    def test_sifir_cokmez(self):
        assert _significant_digits(0) == 0.0


class TestBaselineComparison:
    def test_baseline_yoksa_bos_doner(self):
        from embedpick.benchmark import baseline_comparison

        baseline, sonuclar = baseline_comparison({"model-a": {"q1": 1.0}})
        assert baseline is None and sonuclar == []

    def test_her_model_baseline_ile_kiyaslanir(self):
        from embedpick.benchmark import baseline_comparison

        sorgular = [f"q{i}" for i in range(20)]
        sorgu_bazli = {
            "BM25 (baseline)": {q: 0.0 for q in sorgular},
            "iyi-model": {q: 1.0 for q in sorgular},
            "ayni-model": {q: 0.0 for q in sorgular},
        }
        baseline, sonuclar = baseline_comparison(sorgu_bazli)

        assert baseline == "BM25 (baseline)"
        karar = dict((etiket, s["anlamli"]) for etiket, s in sonuclar)
        assert karar == {"iyi-model": True, "ayni-model": False}

    def test_eslestirilmis_fark_sutun_araliklarindan_dar(self):
        # Asıl mesele bu: iki yöntemin güven aralıkları üst üste binse bile
        # aynı sorgular üzerinde çalıştıkları için FARKIN aralığı daha dar
        # olabilir. "Aralıklar kesişiyor, demek ki fark yok" hatalı bir okuma.
        from embedpick.benchmark import baseline_comparison
        from embedpick.metrics import bootstrap_ci

        sorgular = [f"q{i}" for i in range(24)]
        # Her sorguda model baseline'dan bir tık iyi, ama sorgu zorlukları
        # birbirinden çok farklı: tek tek aralıklar geniş, fark tutarlı.
        temel = {q: (i % 2) * 1.0 for i, q in enumerate(sorgular)}
        model = {q: min(1.0, temel[q] + 0.5) for q in sorgular}
        baseline, sonuclar = baseline_comparison(
            {"BM25 (baseline)": temel, "model": model}
        )

        _, s = sonuclar[0]
        fark_genisligi = s["ci"][1] - s["ci"][0]
        temel_ci = bootstrap_ci(temel.values())
        assert fark_genisligi < (temel_ci[1] - temel_ci[0])
        assert s["anlamli"]
