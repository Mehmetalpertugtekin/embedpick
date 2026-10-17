# embedpick

Embedding modellerini **kendi verinizde** kıyaslayın, başkasının sıralamasında değil.

[English README](https://github.com/Mehmetalpertugtekin/embedpick/blob/main/README.md) · [Çevrimiçi deneyin](https://huggingface.co/spaces/tugtekinalper/embedpick)

## Neden

Embedding modeli seçmek genelde şöyle oluyor: Hugging Face'i aç, indirme
sayısına göre sırala, en üsttekini al. Model kartındaki skorlar genel amaçlı
İngilizce veri setlerinde ölçülmüş. Sizin veriniz o değil.

embedpick daha dar ama daha işe yarar bir soruya cevap veriyor: *benim
dokümanlarımda ve benim sorgularımda hangi model doğru sonucu buluyor, ve bunun
bana süre ve bellek olarak maliyeti ne?*

Kaliteyi, hızı ve belleği yan yana raporluyor. Çoğu benchmark aracından farklı
olarak, aynı tabloya düz kelime aramasını da bir satır olarak koyuyor.

## Asıl mesele baseline

Her çalıştırmada BM25 var — hiçbir sinir ağı içermeyen, klasik kelime
eşleştirme algoritması. Bir embedding modeli onu geçemiyorsa, o modeli
çalıştırmak için harcanan hesap gücü boşa gidiyor demektir.

Pakete dahil Türkçe müşteri destek veri setindeki sonuçlar (72 doküman,
26 sorgu, k=5), baseline gövdelenmiş halde (`--stemmer simple`):

| Model | recall@5 | %95 GA | MRR@5 | doküman/sn | boyut | göreli hız |
|---|---|---|---|---|---|---|
| BM25 (baseline, simple stem) | 0.654 | [0.50, 0.81] | 0.622 | ~32.000 | — | — |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.615 | [0.46, 0.77] | 0.596 | 374 | 384 | 1.00x |
| multilingual-e5-small | 0.692 | [0.56, 0.81] | 0.708 | 322 | 384 | 0.86x |
| trmteb/turkish-embedding-model | **0.846** | [0.73, 0.94] | **0.865** | 119 | 768 | 0.32x |

Aralıklar üst üste biniyor ama bunu "fark yok" diye okumak yanlış olur: iki
yöntem de aynı 26 sorguyu cevaplıyor, dolayısıyla *farkın* belirsizliği tek tek
sütunların belirsizliğinden küçük. Karar verirken bakılacak sayı eşleştirilmiş
karşılaştırma:

| Model | baseline'a göre fark | %95 GA | sonuç |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | −0.038 | [−0.192, +0.115] | gösterilemedi |
| multilingual-e5-small | +0.038 | [−0.115, +0.212] | gösterilemedi |
| trmteb/turkish-embedding-model | **+0.192** | [+0.058, +0.346] | **anlamlı** |

Bu satırlara dikkatli bakın. Üç embedding modelinden ikisinin bu veride kelime
aramasını geçtiği gösterilemiyor; MiniLM'in nokta tahmini ise sıfırın yanlış
tarafında — üstelik yaklaşık 90 kat indeksleme süresi ödeyerek. Buradaki tek
gerçek bulgu Türkçe'ye özel model: üstünlüğü hem düzgün kurulmuş bir baseline'a
hem de anlamlılık testine dayanıyor.

"Gösterilemedi", yöntemlerin eşit olduğu anlamına gelmez; veri setinin o farkı
göstermeye yetmediği anlamına gelir. 26 sorguyla bu eşik zaten alçak, asıl ders
de bu.

Yani çıkarım "embedding gereksiz" değil. Çıkarım şu: **kötü seçilmiş bir
embedding modeli kelime aramasından iyi değildir ve kimse bunu kontrol
etmiyor.**

### Baseline'a da karşısındaki model kadar özen gerekiyor

Bu tablonun daha eski bir sürümünde BM25 0.564 ve fark +28 puandı, çünkü
`tr_tokenize` sadece küçük harfe çeviriyordu. Türkçe sondan eklemeli bir dil:
`iade` ile `iadesi` birbiriyle ilgisiz iki terimdi ve baseline kendi
tokenizer'ı tarafından sakatlanıyordu. Gövdeleme eklemek onu 0.564'ten 0.654'e
çıkardı, manşetteki farkı +19 puana indirdi.

Sonuç ayakta kaldı, iddia değişti. Bir kıyaslamada en zayıf halka kendi
tarafınızdaki özensizliktir, çünkü onu yakalayacak konumda başka kimse yok.
`--stemmer none|simple|zemberek` tam da baseline'ınızın ne yaptığını
görebilmeniz için var — ve bu iyileştirmenin kendisi bile +0.090, güven aralığı
[−0.058, +0.231], yani bu örneklem büyüklüğünde gürültüden ayırt edilemiyor.

## Örnek veri setinden çıkan bulgular

**Belgelenmiş yapılandırma her zaman doğru yapılandırma değil.** e5 model
kartı, girdi metinlerinin başına `query: ` ve `passage: ` öneki konmasını
söylüyor. Bu veri setinde önekleri uygulamak recall@5'i 0.750'den 0.692'ye,
MRR'ı 0.792'den 0.708'e *düşürdü* — aynı model, aynı veri, aynı donanım.

Olası sebep: buradaki dokümanlar kısa, her biri beş-sekiz kelime, dolayısıyla
İngilizce önek her birinin kayda değer bir kısmını kaplıyor. Sebep ne olursa
olsun, önerilen ayar burada yaklaşık altı puana mal oldu ve bunu kendi verinizde
ölçmeden görmenin yolu yoktu.

embedpick bilinen önekleri, model yazarlarının talimatını izleyerek, varsayılan
olarak uyguluyor; `--no-presets` ile kapatıp kendiniz kontrol edebilirsiniz.
26 sorguda bu fark yaklaşık bir buçuk sorguya denk geliyor — tamamen gürültünün
içinde. Bir bulgu değil, kendi verinizde test etmeye değer bir soru olarak
okuyun; harekete geçmeden önce güven aralığıyla birlikte ölçün.

**BM25 ile embedding farklı sorgularda batıyor.** `ödeme yaparken sorun`
sorgusunda BM25 1.00, MiniLM 0.00 aldı. `sahte ürün şüphesi` sorgusunda tam
tersi oldu. Birbirlerini tamamlıyorlar — hibrit aramanın deneysel gerekçesi bu.

**Küçük korpuslar her şeyi gizler.** 16 dokümanlık pilot denemede üç model de
1.0 aldı ve birbirinin aynısı göründü. 72 dokümanda aralarında 23 puanlık fark
açıldı. Modelleri ayırt edemeyen bir benchmark, onlar hakkında hiçbir şey
söylemiyordur.

**Mutlak süreler güvenilmez, oranlar güvenilir.** Aynı makinede tekrarlanan
çalıştırmalarda `doküman/sn` 4 kata kadar oynadı, modeller arası oran ise
yaklaşık %20 içinde kaldı. Makineler arası kıyas için `rel_speed` sütununu
kullanın. Tablonun hızı üç anlamlı basamağa yuvarlaması da bilinçli: bir
sayının hassasiyeti, ona ne kadar güvendiğiniz hakkında bir iddiadır.

**Aralığı olmayan fark bir bulgu değil, bir hipotezdir.** Her kalite sütunu
bootstrap ile hesaplanmış %95 güven aralığı taşıyor ve her model baseline ile
eşleştirilmiş olarak kıyaslanıyor. 26 sorguda farkların çoğu bu eşiği
geçemiyor. Doğru hamle sütunu kaldırmak değil, bunu raporlayıp sorgu
eklemek.

## Kurulum

```bash
pip install embedpick
```

Kaynaktan kurmak için:

```bash
git clone https://github.com/Mehmetalpertugtekin/embedpick.git
cd embedpick
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e .
```

Python 3.9+.

## Kullanım

```bash
# pakete dahil örnek veri setiyle çalıştır
python -m embedpick

# kendi verinizle
python -m embedpick --corpus dokumanlarim.csv --queries sorgularim.csv

# model seçin, ilk 10 sonucu değerlendirin, sorgu detayını görün
python -m embedpick --models intfloat/multilingual-e5-base -k 10 --verbose

# tabloyu kaydedin
python -m embedpick --out sonuclar.csv
```

Tüm parametreler için: `python -m embedpick --help`

### Baseline için Türkçe gövdeleme

`--stemmer`, BM25 baseline'ının metni nasıl böldüğünü belirler:

| değer | ne yapar |
|---|---|
| `none` | sadece küçük harfe çevirir — baseline olduğundan zayıf görünür |
| `simple` | `embedpick/stemmers.py` içinde açıkça yazılı, bağımlılıksız ek ayıklayıcı |
| `zemberek` | gerçek morfolojik çözümleme; `pip install embedpick[zemberek]`, Java gerektirir |

`simple` zaman zaman fazla soyuyor (`fatura` → `fatur`) ama tutarlı soyuyor
(`faturamı` da `fatur` oluyor), erişim açısından istenen de bu. Zekice değil,
kasten okunabilir olacak şekilde yazıldı: baseline'ınızın ne yaptığını
görebilmelisiniz.

## Geliştirme

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Testler model indirmiyor, saniyeler içinde bitiyor. Önce sanal ortamınızın
aktif olduğundan emin olun — test çıktısında `rank_bm25` bulunamadı hatası
görüyorsanız genelde sebep paketlerin sistem Python'ına kurulmuş olmasıdır.

## Veri formatı

**corpus.csv** — aranacak dokümanlar:

```csv
id,text
1,Kargom hala elime ulaşmadı
2,Sipariş takip numaram sistemde görünmüyor
```

**queries.csv** — sorgular ve doğru cevaplar:

```csv
query,relevant_ids
kargo gecikmesi,1;5
paketim nerede,1;2
```

Birden fazla doğru cevabı `;` ile ayırın, sayı sınırı yok. Dosyalar UTF-8
olmalı.

Gerçek farkları görmek için genelde 20-30 sorgu yeterli. Doğru cevabıyla hiç
ortak kelime paylaşmayan sorgular da koyun — anlamsal eşleştirmeyi asıl test
edenler onlar, diğerleri kelime çakışmasıyla da çözülebilir.

## Metrikler

**recall@k** — doğru dokümanların ne kadarı ilk k sonuçta çıktı. Kullanıcıya
sonuç listesi gösteriyorsanız bu önemli.

**MRR@k** — ilk doğru cevap 1. sıradaysa 1.0, 2. sıradaysa 0.5, 3. sıradaysa
0.33; sorgular üzerinden ortalaması. Tek bir cevap gösteriyorsanız ya da ilk
sonucu bir dil modeline veriyorsanız bu önemli.

İkisi çelişebilir ve çelişmesi bilgi taşır. Önceki bir çalıştırmada e5,
MiniLM'den yüksek recall ama düşük MRR aldı: daha fazlasını buldu ama daha kötü
sıraladı. Hangi modelin "daha iyi" olduğu, ne inşa ettiğinize bağlı.

**doküman/sn** — indeksleme hızı. Makineye bağlı, yukarıdaki uyarıya bakın.

**dim** ve **index_mb** — vektör genişliği ve indeks boyutu. 768 boyutlu bir
model, aynı korpusta 384 boyutlunun iki katı bellek ister.

## Web arayüzü

Pakete bir Gradio arayüzü dahil. Açılışta örnek veri setindeki sonuçları
gösteriyor, isteyen kendi CSV dosyalarını yükleyip canlı çalıştırabiliyor.

```bash
pip install embedpick[ui]
python app.py
```

Yayındaki sürüm:
[huggingface.co/spaces/tugtekinalper/embedpick](https://huggingface.co/spaces/tugtekinalper/embedpick)

## Etiket kontrolü

İki şey kendiliğinden yapılıyor:

- `relevant_ids` korpusta olmayan bir id'ye işaret ediyorsa çalıştırma durur ve
  hangi sorgu olduğunu söyler. Aksi halde o sorgu sonsuza kadar sessizce sıfır
  alır ve siz modeli suçlarsınız.
- BM25 dahil *hiçbir* yöntem bir sorgunun cevabını bulamıyorsa embedpick bunu
  işaretler. Her şey aynı sorguda batıyorsa etiket genelde yanlıştır.
  Geliştirme sırasında bu, yanlış etiketlenmiş bir sorguyu yakaladı.

## Örnek veri seti

`embedpick/sample_data/` klasöründe dokuz temaya yayılmış 72 sentetik Türkçe müşteri destek
mesajı (kargo, iade, ödeme, hesap, ürün kalitesi, kampanya, garanti, sipariş
yönetimi, destek) ve 26 etiketli sorgu var. Kazınmadı, yazıldı; bu yüzden
lisans veya gizlilik kısıtı taşımıyor.

Bazı sorgular cevaplarıyla kasten hiç ortak kelime paylaşmıyor — örneğin
`güvenlik ihlali şüphesi` sorgusunun doğru cevabı *"Hesabıma başkası girmiş
olabilir"*. Kelime araması bunları çözemez, amaç da bu.

## Sınırlar

- 72 dokümanlık bir korpusta süre ölçümü, özellikle BM25 için, ölçüm gürültüsü
  sınırında. Hız sütununu gösterge olarak okuyun.
- BM25 ile sinirsel modeller farklı ölçekleniyor. 72 dokümandaki oran,
  100.000 dokümandaki oran değildir.
- Sonuçlar tek bir makineden, yalnızca işlemci üzerinde, GPU kullanılmadan.

## Yol haritası

- Hibrit arama (BM25 + embedding skor birleştirme) dördüncü satır olarak
- Hub tespiti: neredeyse her sorguda çıkan dokümanları işaretleme
- Markdown rapor çıktısı
- İsteğe bağlı Gradio arayüzü

## Lisans

MIT
