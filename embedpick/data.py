"""Korpus ve sorgu dosyalarını okur, formatlarını doğrular."""

import pandas as pd


def load_corpus(path):
    df = pd.read_csv(path, encoding="utf-8")

    missing = {"id", "text"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} dosyasında eksik sütun: {sorted(missing)}. "
            "Beklenen başlık satırı: id,text"
        )

    if df["id"].duplicated().any():
        tekrar = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ValueError(f"{path} içinde tekrar eden id değerleri var: {tekrar}")

    df["text"] = df["text"].astype(str).str.strip()
    return df


def load_queries(path):
    df = pd.read_csv(path, encoding="utf-8")

    missing = {"query", "relevant_ids"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} dosyasında eksik sütun: {sorted(missing)}. "
            "Beklenen başlık satırı: query,relevant_ids"
        )

    df["query"] = df["query"].astype(str).str.strip()
    df["relevant_ids"] = df["relevant_ids"].apply(_parse_ids)
    return df


def _parse_ids(value):
    parts = [p.strip() for p in str(value).split(";") if p.strip()]
    if not parts:
        raise ValueError(f"Boş relevant_ids değeri bulundu: {value!r}")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"relevant_ids sayı olmalı, alınan: {value!r}. "
            "Birden fazla id noktalı virgülle ayrılır, örnek: 3;11"
        ) from exc


def validate(corpus, queries):
    """Sorgu etiketlerinin korpusta gerçekten var olduğunu kontrol eder.

    Etiket hataları benchmark sonuçlarını sessizce bozar: olmayan bir id'yi
    hiçbir model bulamaz, sen de modeli suçlarsın. Baştan yakalamak daha iyi.
    """
    bilinen = set(corpus["id"])
    sorunlar = []

    for q, ids in zip(queries["query"], queries["relevant_ids"]):
        eksik = [i for i in ids if i not in bilinen]
        if eksik:
            sorunlar.append(f'  "{q}" -> korpusta olmayan id: {eksik}')

    if sorunlar:
        raise ValueError(
            "Sorgu etiketleri korpusla uyuşmuyor:\n" + "\n".join(sorunlar)
        )
