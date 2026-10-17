"""BM25 baseline'ı için Türkçe gövdeleyiciler.

Türkçe sondan eklemeli bir dil: "iade" ve "iadesi" BM25 için birbiriyle
ilgisiz iki terim. Gövdeleme yapmayan bir baseline, kıyasladığı embedding
modellerinin aleyhine değil lehine hata yapar ve ölçülen fark gerçekte
olduğundan büyük çıkar. Baseline'a da karşısındaki model kadar özen
göstermek gerekiyor, bu modül onun için.
"""


def identity(word):
    return word


# Yaygın çekim ekleri, uzundan kısaya. Sıra önemli: "leri" önce denenmezse
# "i" soyulur ve geriye "ler" kalır.
_SUFFIXES = (
    "larımız", "lerimiz", "larınız", "leriniz",
    "ımız", "imiz", "umuz", "ümüz",
    "ınız", "iniz", "unuz", "ünüz",
    "ları", "leri", "lar", "ler",
    "nin", "nın", "nun", "nün",
    "den", "dan", "ten", "tan",
    "yla", "yle",
    "de", "da", "te", "ta",
    "ya", "ye", "la", "le",
    "yı", "yi", "yu", "yü",
    "sı", "si", "su", "sü",
    "nı", "ni", "nu", "nü",
    "ım", "im", "um", "üm",
    "ın", "in", "un", "ün",
    "a", "e", "ı", "i", "u", "ü", "m", "n",
)

# Bundan kısa bir gövde bırakacak soyma yapılmıyor. Aksi halde "iade"
# "iad"e, oradan tek harfe kadar erir ve farklı kelimeler birbirine karışır.
_MIN_STEM = 4


def simple(word):
    """Bağımlılıksız, kasten kaba bir ek ayıklayıcı.

    Dilbilgisel olarak doğru gövde üretmeye çalışmıyor; amacı aynı kökten
    gelen yüzey biçimlerini BM25'in aynı terim sayacağı ortak bir dizeye
    indirmek. Zaman zaman fazla soyuyor ("fatura" -> "fatur"), ama tutarlı
    soyduğu için erişim açısından sorun değil: "faturamı" da "fatur" oluyor.

    Gerçek morfolojik çözümleme için --stemmer zemberek kullanın.
    """
    degisti = True
    while degisti:
        degisti = False
        for ek in _SUFFIXES:
            if word.endswith(ek) and len(word) - len(ek) >= _MIN_STEM:
                word = word[: -len(ek)]
                degisti = True
                break
    return word


def _zemberek():
    """Zemberek'i tembel yükler ve kelime -> lemma fonksiyonu döndürür.

    Zemberek JVM üzerinde çalışır ve opsiyonel bir bağımlılıktır; projenin
    geri kalanı onsuz çalışsın diye import burada.
    """
    try:
        from zemberek import TurkishMorphology
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise ImportError(
            "--stemmer zemberek için zemberek-python gerekiyor "
            "(pip install embedpick[zemberek]) ve sistemde Java kurulu olmalı."
        ) from exc

    morphology = TurkishMorphology.create_with_defaults()
    onbellek = {}

    def stem(word):
        if word not in onbellek:
            analiz = morphology.analyze_and_disambiguate(word).best_analysis()
            # Çözümlenemeyen kelimede lemma yerine kelimenin kendisi kalsın;
            # sessizce boş dize döndürmek terimi tamamen kaybettirir.
            onbellek[word] = analiz[0].item.lemma if analiz else word
        return onbellek[word]

    return stem


CHOICES = ("none", "simple", "zemberek")


def get_stemmer(name):
    """İsimden gövdeleyici fonksiyon üretir."""
    if name == "none":
        return identity
    if name == "simple":
        return simple
    if name == "zemberek":
        return _zemberek()
    raise ValueError(f"bilinmeyen stemmer: {name!r}. seçenekler: {', '.join(CHOICES)}")
