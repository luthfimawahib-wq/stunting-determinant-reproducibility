from __future__ import annotations
import logging
import re
import numpy as np
import pandas as pd

import konfigurasi as K


def get_logger(nama: str = "p1") -> logging.Logger:
    logger = logging.getLogger(nama)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


def muat_parquet(path: str) -> pd.DataFrame:
    log.info("Memuat Parquet: %s", path)
    df = pd.read_parquet(path)
    log.info("  -> %s baris, %s kolom", f"{len(df):,}", df.shape[1])
    return df


def subset_sel(df: pd.DataFrame, sumber: str, kohort: str) -> pd.DataFrame:
    mask = (df[K.SOURCE_COL].astype(str) == sumber) &           (df[K.COHORT_COL].astype(str) == kohort)
    return df.loc[mask].copy()


def _norm(s: str) -> str:
    return str(s).strip().lower()


def kolom_indikator_missing(df: pd.DataFrame) -> set[str]:
    kandidat = set()
    pola_regex = [p.replace("{f}", r".+") for p in K.MISSING_INDICATOR_PATTERNS]
    pola_regex = [re.compile("^" + p + "$", re.IGNORECASE) for p in pola_regex]
    for c in df.columns:
        if any(p.match(str(c)) for p in pola_regex):
            kandidat.add(c)
    return kandidat


def peta_indikator(df: pd.DataFrame, fitur: list[str]) -> dict[str, str]:
    kolom = set(df.columns)
    peta = {}
    for f in fitur:
        for pola in K.MISSING_INDICATOR_PATTERNS:
            kandidat = pola.format(f=f)
            if kandidat in kolom:
                peta[f] = kandidat
                break
    return peta


import json


def muat_skema_encoding() -> dict | None:
    if not K.SKEMA_ENCODING_PATH:
        return None
    try:
        with open(K.SKEMA_ENCODING_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        log.warning("skema_encoding.json tidak ditemukan/terbaca; pakai heuristik nama.")
        return None


_TIPE_FITUR = ("nominal", "ordinal", "biner", "binary", "kontinu", "continuous", "numerik")
_TIPE_BUKAN_FITUR = ("meta", "drop", "buang", "hapus", "target")


def _kumpulkan_kolom(skema: dict, kunci_cocok) -> set[str]:
    out = set()
    for kunci, nilai in skema.items():
        k = _norm(kunci)
        if any(p in k for p in kunci_cocok):
            if isinstance(nilai, dict):
                out |= set(nilai.keys())
            elif isinstance(nilai, (list, tuple)):
                out |= set(nilai)
    return out


def muat_fitur_universal() -> set[str] | None:
    if not K.UNIVERSAL_FEATURES_PATH:
        return None
    try:
        txt = open(K.UNIVERSAL_FEATURES_PATH, encoding="utf-8").read()
        return {ln.strip() for ln in txt.splitlines() if ln.strip()}
    except OSError:
        log.warning("Berkas fitur universal tidak terbaca: %s", K.UNIVERSAL_FEATURES_PATH)
        return None


def kolom_fitur(df: pd.DataFrame) -> list[str]:
    kol = set(df.columns)
    skema = muat_skema_encoding()

    if skema:
        kandidat = _kumpulkan_kolom(skema, _TIPE_FITUR) & kol
        bukan = _kumpulkan_kolom(skema, _TIPE_BUKAN_FITUR)
        if not kandidat:
            skema = None
    if not skema:
        kandidat = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
        bukan = set()

    excl = set(map(_norm, K.TARGET_COLS)) | set(map(_norm, K.EXTRA_EXCLUDE_COLS))
    excl |= set(map(_norm, K.LEAKAGE_EXCLUDE_COLS))
    excl |= set(map(_norm, getattr(K, "KONKUREN_ANTRO_EXCLUDE", [])))
    excl |= {_norm(K.SOURCE_COL), _norm(K.COHORT_COL)}
    bukan_norm = set(map(_norm, bukan))
    ind = kolom_indikator_missing(df) if not K.INCLUDE_MISSING_INDICATORS else set()

    fitur = []
    for c in kandidat:
        nc = _norm(c)
        if c in ind:
            continue
        if nc in excl or nc in bukan_norm:
            continue
        if any(nc.startswith(p) for p in K.META_PREFIXES):
            continue
        if re.search(r"(^|_)id($|_)", nc):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        fitur.append(c)

    if K.FEATURE_SET == "universal":
        uni = muat_fitur_universal()
        if uni is None:
            log.warning("FEATURE_SET=universal tetapi daftar universal tak ada; pakai kaya.")
        else:
            sebelum = len(fitur)
            fitur = [f for f in fitur if f in uni]
            log.info("Mode universal: %d -> %d fitur (irisan dgn daftar 45).",
                     sebelum, len(fitur))
    return sorted(fitur)


def tutup_balik_teramati(df: pd.DataFrame, fitur: list[str]) -> pd.DataFrame:
    peta = peta_indikator(df, fitur)
    out = df[fitur].copy()
    tanpa_indikator = []
    for f in fitur:
        ind = peta.get(f)
        if ind is None:
            tanpa_indikator.append(f)
            continue
        mask_missing = pd.to_numeric(df[ind], errors="coerce").fillna(0) == 1
        out.loc[mask_missing, f] = np.nan
    if tanpa_indikator:
        log.warning("  %d fitur tanpa indikator missing, dipakai apa adanya: %s%s",
                    len(tanpa_indikator), tanpa_indikator[:5],
                    " ..." if len(tanpa_indikator) > 5 else "")
    return out


def pilih_fitur_terisi(df_obs: pd.DataFrame, fitur: list[str]) -> tuple[list[str], pd.DataFrame]:
    baris = []
    lolos = []
    n = len(df_obs)
    for f in fitur:
        s = df_obs[f]
        n_obs = int(s.notna().sum())
        komplit = n_obs / n if n else 0.0
        n_unik = int(s.nunique(dropna=True))
        ok = (komplit >= K.MIN_COMPLETENESS) and (n_unik > 1)
        baris.append(dict(fitur=f, n_observasi=n_obs, kelengkapan=round(komplit, 4),
                          n_unik=n_unik, lolos=ok))
        if ok:
            lolos.append(f)
    cakupan = pd.DataFrame(baris).sort_values("kelengkapan", ascending=False)
    return lolos, cakupan


def is_kategorikal(s: pd.Series) -> bool:
    return s.nunique(dropna=True) <= K.MAX_UNIQUE_CATEGORICAL


def bin_fitur(s: pd.Series) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce")
    out = np.full(len(s), np.nan, dtype=float)
    mask = s.notna().values
    if mask.sum() == 0:
        return out
    vals = s.values[mask]
    if is_kategorikal(s):
        kode = pd.Categorical(vals).codes.astype(float)
    else:
        try:
            kode = pd.qcut(vals, q=K.N_BINS, labels=False, duplicates="drop").astype(float)
        except (ValueError, IndexError):
            kode = pd.Categorical(vals).codes.astype(float)
    out[mask] = kode
    return out


def ambil_target_biner(df: pd.DataFrame) -> pd.Series | None:
    col = K.TARGET_FOR_MODEL
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def subsample(df: pd.DataFrame, n: int | None, seed: int = K.SEED) -> pd.DataFrame:
    if not n or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def muat_kategori_fitur() -> dict | None:
    path = getattr(K, "MATRIKS_KATEGORI_PATH", None)
    if not path:
        return None
    try:
        m = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    except (FileNotFoundError, OSError):
        log.warning("Matriks kategori tidak ditemukan: %s (pakai grouping awalan).", path)
        return None
    kol_var = next((c for c in m.columns if c.lower() in ("variabel", "fitur", "variable")), None)
    kol_kat = next((c for c in m.columns if "kategori" in c.lower() or c.lower() == "category"), None)
    if not kol_var or not kol_kat:
        return None
    return dict(zip(m[kol_var].astype(str), m[kol_kat].astype(str)))
