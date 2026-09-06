"""embedpick web arayüzü.

Açılışta örnek veri setindeki hazır sonuçları gösterir, isteyen kendi
CSV dosyalarını yükleyip canlı çalıştırabilir.

Çalıştırmak için:
    pip install embedpick[ui]
    python app.py
"""

import pathlib
from importlib import resources

# ZeroGPU donanımında çalışırken bu import torch'tan ÖNCE gelmeli.
# Kendi makinenizde `spaces` kurulu olmayacaktır; o durumda sarmalayıcı
# hiçbir şey yapmayan bir dekoratöre dönüşür ve uygulama normal çalışır.
try:
    import spaces

    gpu_gorevi = spaces.GPU(duration=300)
except Exception:

    def gpu_gorevi(fn):
        return fn


import gradio as gr
import pandas as pd

from embedpick.benchmark import run_benchmark
from embedpick.data import load_corpus, load_queries, validate
from embedpick.metrics import consensus_failures
from embedpick.retrievers import DEFAULT_MODELS

REPO = "https://github.com/Mehmetalpertugtekin/embedpick"


def paket_dosyasi(ad):
    return resources.files("embedpick") / "sample_data" / ad


# Yayınlanmış paket sürümü bu dosyayı içermeyebilir, o yüzden gömülü bir
# yedek tutuyoruz. Sayılar 72 doküman / 26 sorgu ile tek bir makinede,
# yalnızca işlemci üzerinde ölçüldü.
YEDEK_SONUCLAR = pd.DataFrame(
    [
        ["BM25 (baseline)", 0.564, 0.564, 226486.3, 0, 0.00, "-"],
        ["paraphrase-multilingual-MiniLM-L12-v2", 0.615, 0.596, 340.4, 384, 0.11, "1.00x"],
        ["multilingual-e5-small", 0.692, 0.708, 304.1, 384, 0.11, "0.89x"],
        ["turkish-embedding-model", 0.846, 0.865, 103.0, 768, 0.21, "0.30x"],
    ],
    columns=[
        "model",
        "recall@5",
        "mrr@5",
        "docs_per_sec",
        "dim",
        "index_mb",
        "rel_speed",
    ],
)


def hazir_sonuclar():
    """Önce app.py'nin yanındaki dosyaya, sonra pakete, sonra gömülü yedeğe bakar."""
    yerel = pathlib.Path(__file__).parent / "precomputed_results.csv"
    if yerel.exists():
        try:
            return pd.read_csv(yerel)
        except Exception:
            pass

    try:
        return pd.read_csv(paket_dosyasi("precomputed_results.csv"))
    except Exception:
        return YEDEK_SONUCLAR


GIRIS = f"""
# embedpick

Embedding modellerini **kendi verinizde** kıyaslayın, başkasının sıralamasında değil.

Model kartlarındaki skorlar genel amaçlı İngilizce veri setlerinde ölçülür.
Sizin verinizde hangi modelin işe yaradığını ancak kendi verinizde ölçerek
bulabilirsiniz. embedpick kaliteyi, hızı ve belleği yan yana raporlar — ve
aynı tabloya düz kelime aramasını (BM25) da bir satır olarak koyar.

[Kaynak kod]({REPO}) · `pip install embedpick`
"""

HAZIR_ACIKLAMA = """
### Örnek veri setindeki sonuçlar

72 sentetik Türkçe müşteri destek mesajı, 26 etiketli sorgu, k=5.

İlk iki satıra dikkat: MiniLM, kelime aramasına göre 5 puanlık kazanç için
yaklaşık 600 kat indeksleme süresi ödüyor. Türkçe'ye özel model ise +28 puanla
bedelini hak ediyor. Yani çıkarım "embedding gereksiz" değil — **kötü seçilmiş
bir embedding modeli kelime aramasından iyi değil ve kimse bunu kontrol
etmiyor.**

`docs_per_sec` makineye göre değişir; makineler arası kıyas için `rel_speed`
sütununu kullanın.
"""

NASIL = """
### Kendi verinizle çalıştırın

İki CSV dosyası yükleyin:

**corpus.csv** — aranacak dokümanlar

```
id,text
1,Kargom hala elime ulaşmadı
2,Sipariş takip numaram sistemde görünmüyor
```

**queries.csv** — sorgular ve doğru cevaplar

```
query,relevant_ids
kargo gecikmesi,1;5
paketim nerede,1;2
```

Birden fazla doğru cevabı `;` ile ayırın. Dosyalar UTF-8 olmalı.

Gerçek farkları görmek için 20-30 sorgu genelde yeterli. Doğru cevabıyla hiç
ortak kelime paylaşmayan sorgular da koyun — anlamsal eşleştirmeyi asıl test
edenler onlar.
"""


@gpu_gorevi
def calistir(corpus_dosya, queries_dosya, secili, ek_modeller, k, baseline, presets):
    if corpus_dosya is None or queries_dosya is None:
        return None, None, "İki CSV dosyasını da yükleyin."

    modeller = list(secili or [])
    if ek_modeller and ek_modeller.strip():
        modeller += [m.strip() for m in ek_modeller.split(",") if m.strip()]

    if not modeller:
        return None, None, "En az bir model seçin veya yazın."

    try:
        corpus = load_corpus(corpus_dosya)
        queries = load_queries(queries_dosya)
        validate(corpus, queries)
    except (ValueError, FileNotFoundError) as exc:
        return None, None, f"Veri hatası:\n\n```\n{exc}\n```"

    try:
        df, sorgu_bazli = run_benchmark(
            corpus,
            queries,
            model_names=modeller,
            k=int(k),
            repeats=3,
            include_baseline=baseline,
            use_presets=presets,
        )
    except Exception as exc:
        return None, None, f"Çalıştırma hatası:\n\n```\n{type(exc).__name__}: {exc}\n```"

    detay = pd.DataFrame(sorgu_bazli)
    detay.insert(0, "sorgu", detay.index)
    detay = detay.reset_index(drop=True)

    notlar = [
        f"{len(corpus)} doküman, {len(queries)} sorgu, k={int(k)} ile çalıştırıldı."
    ]
    if not presets:
        notlar.append(
            "Model önekleri kapalı. e5 gibi önek bekleyen modeller düşük skor alır."
        )

    ortak = consensus_failures(sorgu_bazli) if len(sorgu_bazli) > 1 else []
    if ortak:
        liste = "\n".join(f"- {q}" for q in ortak)
        notlar.append(
            "**Hiçbir yöntemin bulamadığı sorgular:**\n"
            f"{liste}\n\n"
            "Bütün yöntemler aynı sorguda batıyorsa sorun genelde modelde değil, "
            "etiketlerdedir. Bu sorguların `relevant_ids` değerlerini gözden geçirin."
        )

    return df, detay, "\n\n".join(notlar)


with gr.Blocks(title="embedpick") as demo:
    gr.Markdown(GIRIS)

    gr.Markdown(HAZIR_ACIKLAMA)
    gr.Dataframe(value=hazir_sonuclar(), interactive=False, wrap=True)

    with gr.Accordion("Kendi verinizle çalıştırın", open=False):
        gr.Markdown(NASIL)

        with gr.Row():
            corpus_yukle = gr.File(
                label="corpus.csv", file_types=[".csv"], type="filepath"
            )
            queries_yukle = gr.File(
                label="queries.csv", file_types=[".csv"], type="filepath"
            )

        model_secimi = gr.CheckboxGroup(
            choices=DEFAULT_MODELS,
            value=DEFAULT_MODELS[:2],
            label="Modeller",
        )
        ek_model = gr.Textbox(
            label="Ek modeller (Hugging Face adı, virgülle ayırın)",
            placeholder="intfloat/multilingual-e5-base, BAAI/bge-m3",
        )

        with gr.Row():
            k_secimi = gr.Slider(1, 20, value=5, step=1, label="k (ilk kaç sonuç)")
            baseline_secimi = gr.Checkbox(
                value=True, label="BM25 baseline'ı dahil et"
            )
            preset_secimi = gr.Checkbox(
                value=True, label="Bilinen model öneklerini uygula"
            )

        gr.Markdown(
            "İlk çalıştırmada modeller indirilir, birkaç dakika sürebilir. "
            "Sonraki çalıştırmalar önbellekten okur."
        )

        calistir_dugmesi = gr.Button("Çalıştır", variant="primary")

        durum = gr.Markdown()
        sonuc_tablosu = gr.Dataframe(label="Sonuçlar", interactive=False, wrap=True)
        detay_tablosu = gr.Dataframe(
            label="Sorgu bazlı recall", interactive=False, wrap=True
        )

    calistir_dugmesi.click(
        fn=calistir,
        inputs=[
            corpus_yukle,
            queries_yukle,
            model_secimi,
            ek_model,
            k_secimi,
            baseline_secimi,
            preset_secimi,
        ],
        outputs=[sonuc_tablosu, detay_tablosu, durum],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
