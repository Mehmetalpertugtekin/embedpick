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
#
# duration bir rezervasyon talebi, üst sınır değil: hesap seviyesinin
# tavanını aşan bir talep işi hiç başlatmadan reddediliyor ve kullanıcı
# "requested GPU duration is larger than the maximum allowed" görüyor.
# Eskiden 300 isteniyordu; bu veri seti için fazlasıyla cömert, çünkü asıl
# maliyet model indirme (GPU'suz) ve 72 dokümanı kodlamak saniyeler sürüyor.
# Kısa talep ayrıca kuyrukta öne geçiriyor.
GPU_SURESI = 120

try:
    import spaces

    gpu_gorevi = spaces.GPU(duration=GPU_SURESI)
except Exception:

    def gpu_gorevi(fn):
        return fn


import gradio as gr
import pandas as pd

from embedpick.benchmark import run_benchmark
from embedpick.data import load_corpus, load_queries, validate
from embedpick.metrics import consensus_failures
from embedpick.retrievers import DEFAULT_MODELS
from embedpick.stemmers import CHOICES as STEMMER_CHOICES

REPO = "https://github.com/Mehmetalpertugtekin/embedpick"


def paket_dosyasi(ad):
    return resources.files("embedpick") / "sample_data" / ad


# Yayınlanmış paket sürümü bu dosyayı içermeyebilir, o yüzden gömülü bir
# yedek tutuyoruz. Sayılar 72 doküman / 26 sorgu ile tek bir makinede,
# yalnızca işlemci üzerinde ölçüldü.
YEDEK_SONUCLAR = pd.DataFrame(
    [
        # model, recall@5, GA, mrr@5, doküman/sn, boyut, MB, göreli hız
        ["BM25 (baseline, simple stem)",
         0.654, "[0.50, 0.81]", 0.622, 32000.0, 0, 0.00, "-"],
        ["paraphrase-multilingual-MiniLM-L12-v2",
         0.615, "[0.46, 0.77]", 0.596, 374.0, 384, 0.11, "1.00x"],
        ["multilingual-e5-small",
         0.692, "[0.56, 0.81]", 0.708, 322.0, 384, 0.11, "0.86x"],
        ["turkish-embedding-model",
         0.846, "[0.73, 0.94]", 0.865, 119.0, 768, 0.21, "0.32x"],
    ],
    columns=[
        "model",
        "recall@5",
        "recall@5_ci",
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
Baseline gövdelenmiş halde (`--stemmer simple`) çalıştırıldı.
"""

# Yorum kısmı katlanmış duruyor: tablo ve çalıştırma paneli ilk ekranda
# kalsın diye. Metin değerli ama aracı kullanmanın önüne geçmemeli.
YORUM = """
Baseline'a göre eşleştirilmiş fark, %95 güven aralığıyla:

| Model | fark | %95 GA | sonuç |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | −0.038 | [−0.192, +0.115] | gösterilemedi |
| multilingual-e5-small | +0.038 | [−0.115, +0.212] | gösterilemedi |
| trmteb/turkish-embedding-model | **+0.192** | [+0.058, +0.346] | **anlamlı** |

Üç modelden ikisinin bu veride kelime aramasını geçtiği gösterilemiyor;
MiniLM'in nokta tahmini sıfırın yanlış tarafında. Yani çıkarım "embedding
gereksiz" değil — **kötü seçilmiş bir embedding modeli kelime aramasından iyi
değil ve kimse bunu kontrol etmiyor.**

"Gösterilemedi", fark yok demek değil; 26 sorgunun o farkı göstermeye
yetmediği demek. `docs_per_sec` makineye göre değişir; makineler arası kıyas
için `rel_speed` sütununu kullanın.
"""

NASIL = """
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
def calistir(
    corpus_dosya, queries_dosya, secili, ek_modeller, k, baseline, presets, stemmer
):
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
            stemmer=stemmer,
        )
    except Exception as exc:
        hata = f"{type(exc).__name__}: {exc}"
        return None, None, f"Çalıştırma hatası:\n\n```\n{hata}\n```"

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
    if baseline and stemmer == "none":
        notlar.append(
            "BM25 gövdelemesiz çalıştı. Türkçe sondan eklemeli olduğu için "
            "baseline olduğundan zayıf görünür ve modellerle arasındaki fark "
            "olduğundan büyük çıkar. `simple` ile tekrar deneyin."
        )
    notlar.append(
        f"`recall@{int(k)}_ci`, {len(queries)} sorgu üzerinden bootstrap ile "
        "hesaplanmış yüzde 95 güven aralığıdır. Aralıklar üst üste biniyorsa "
        "aradaki fark bu veri setinde gösterilememiştir."
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


# Gradio'nun ana kabı varsayılan olarak görünür alan yüksekliğinde ve
# overflow-y: hidden geliyor; sayfa büyüdüğünde alt kısım kırpılıyor ve
# kullanıcı fareyle aşağı kaydıramıyor. Space bir iframe içinde gömülü
# gösterildiğinde özellikle can sıkıcı: Çalıştır butonu erişilemez oluyor.
# Kabı içeriğe göre büyümeye bırakıyoruz.
CSS = """
.gradio-container {
    height: auto !important;
    min-height: 100vh;
    overflow-y: visible !important;
}
"""

with gr.Blocks(title="embedpick", css=CSS) as demo:
    gr.Markdown(GIRIS)

    gr.Markdown(HAZIR_ACIKLAMA)
    gr.Dataframe(value=hazir_sonuclar(), interactive=False, wrap=True)

    with gr.Accordion("Bu tablo ne anlatıyor?", open=False):
        gr.Markdown(YORUM)

    with gr.Accordion("Kendi verinizle çalıştırın", open=False):
        # Düzen kasten sıkışık: iki yükleme kutusu, sonra doğrudan Çalıştır.
        # Önceki sürümde biçim açıklaması ve bütün ayarlar butonun üstündeydi,
        # buton ekranın altında kalıyordu ve kullanıcı aracı çalıştıramadan
        # vazgeçiyordu. Varsayılanlar zaten makul; ayarlar katlanmış duruyor.
        gr.Markdown(
            "İki CSV yükleyin ve **Çalıştır**'a basın. Varsayılan ayarlar makul; "
            "biçim örneği ve ayarlar aşağıdaki başlıklarda."
        )

        with gr.Row():
            corpus_yukle = gr.File(
                label="corpus.csv", file_types=[".csv"], type="filepath"
            )
            queries_yukle = gr.File(
                label="queries.csv", file_types=[".csv"], type="filepath"
            )

        with gr.Accordion("Ayarlar", open=False):
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

            stemmer_secimi = gr.Radio(
                choices=list(STEMMER_CHOICES),
                value="simple",
                label="BM25 baseline'ında Türkçe gövdeleme",
                info=(
                    "Gövdelemesiz baseline 'iade' ile 'iadesi'yi ilgisiz iki terim "
                    "sayar ve olduğundan zayıf görünür. zemberek ayrı kurulum ister."
                ),
            )

        with gr.Accordion("CSV biçimi", open=False):
            gr.Markdown(NASIL)

        calistir_dugmesi = gr.Button("Çalıştır", variant="primary")

        gr.Markdown(
            "İlk çalıştırmada modeller indirilir, birkaç dakika sürebilir; "
            "sonraki çalıştırmalar önbellekten okur. Her çalıştırma için "
            f"{GPU_SURESI} saniyelik bir GPU payı ayrılıyor — çok büyük bir "
            "korpusta süre dolarsa model sayısını azaltın ya da paketi kendi "
            "makinenizde çalıştırın: `pip install embedpick`."
        )

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
            stemmer_secimi,
        ],
        outputs=[sonuc_tablosu, detay_tablosu, durum],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
