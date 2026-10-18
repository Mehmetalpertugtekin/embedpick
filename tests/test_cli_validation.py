"""CLI girdi doğrulaması ve sürüm testleri."""

import subprocess
import sys

import pytest

import embedpick
from embedpick.benchmark import run_benchmark
from embedpick.cli import build_parser, pozitif_tamsayi


class TestPozitifTamsayi:
    def test_gecerli_deger_gecer(self):
        assert pozitif_tamsayi("3") == 3

    @pytest.mark.parametrize("deger", ["0", "-1"])
    def test_sifir_ve_negatif_reddedilir(self, deger):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="1 veya daha büyük"):
            pozitif_tamsayi(deger)

    def test_sayi_olmayan_reddedilir(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="tam sayı"):
            pozitif_tamsayi("bes")


class TestParserDogrulamasi:
    @pytest.mark.parametrize("arg", ["--repeats", "-k"])
    def test_sifir_parser_asamasinda_reddedilir(self, arg, capsys):
        # Eskiden parser bunu kabul ediyor, çökme ölçüm döngüsünde
        # UnboundLocalError olarak çıkıyordu — kullanıcının okuyamayacağı
        # bir hata. Girdi daha kapıda reddedilmeli.
        with pytest.raises(SystemExit):
            build_parser().parse_args([arg, "0"])
        assert "1 veya daha büyük" in capsys.readouterr().err

    def test_varsayilanlar_gecerli(self):
        args = build_parser().parse_args([])
        assert args.k == 5 and args.repeats == 5


class TestRunBenchmarkDogrulamasi:
    """Doğrulama kütüphanede de var: Gradio ve kütüphane kullanıcıları
    CLI'dan geçmiyor."""

    def test_repeats_sifir_hata_verir(self):
        with pytest.raises(ValueError, match="repeats en az 1"):
            run_benchmark(None, None, [], repeats=0)

    def test_k_sifir_hata_verir(self):
        with pytest.raises(ValueError, match="k en az 1"):
            run_benchmark(None, None, [], k=0)


class TestSurum:
    def test_tek_kaynaktan_okunuyor(self):
        # pyproject.toml tek kaynak; __init__.py'de elle yazılmış bir sürüm
        # sessizce ayrı düşüyordu (paket 0.1.1 iken __init__ 0.1.0 diyordu).
        from importlib.metadata import version

        assert embedpick.__version__ == version("embedpick")

    def test_version_bayragi_calisiyor(self):
        cikti = subprocess.run(
            [sys.executable, "-m", "embedpick", "--version"],
            capture_output=True,
            text=True,
        )
        assert cikti.returncode == 0
        assert embedpick.__version__ in cikti.stdout


class TestMainGuard:
    def test_main_import_edilince_surec_olmuyor(self):
        # Guard olmadan modül düzeyindeki raise SystemExit(main()) yüzünden
        # __main__.py'yi import eden her şey süreci öldürüyordu.
        cikti = subprocess.run(
            [sys.executable, "-c", "import embedpick.__main__; print('import ok')"],
            capture_output=True,
            text=True,
        )
        assert cikti.returncode == 0
        assert "import ok" in cikti.stdout
