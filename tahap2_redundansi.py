from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import normalized_mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import konfigurasi as K
import utilitas as U

log = U.get_logger()


def _matriks_nmi(df_obs: pd.DataFrame, fitur: list[str]) -> pd.DataFrame:
    sub = U.subsample(df_obs[fitur], K.MAX_ROWS_MI)
    binned = {f: U.bin_fitur(sub[f]) for f in fitur}
    p = len(fitur)
    M = np.eye(p)
    for i, j in combinations(range(p), 2):
        a, b = binned[fitur[i]], binned[fitur[j]]
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 30:
            val = np.nan
        else:
            val = normalized_mutual_info_score(a[mask].astype(int), b[mask].astype(int))
        M[i, j] = M[j, i] = val
    return pd.DataFrame(M, index=fitur, columns=fitur)


def _nmi_dengan_target(df_obs: pd.DataFrame, y: pd.Series, fitur: list[str]) -> pd.Series:
    if y is None:
        return pd.Series(0.0, index=fitur)
    yb = U.bin_fitur(y)
    out = {}
    for f in fitur:
        a = U.bin_fitur(df_obs[f])
        mask = ~np.isnan(a) & ~np.isnan(yb)
        out[f] = (normalized_mutual_info_score(a[mask].astype(int), yb[mask].astype(int))
                  if mask.sum() >= 30 else 0.0)
    return pd.Series(out)


def _vif(df_obs: pd.DataFrame, fitur: list[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor


    komposit_set = set(getattr(K, "KOMPOSIT_KONSTITUEN", {}).keys())
    fitur_indep = [f for f in fitur if f not in komposit_set]
    X = df_obs[fitur_indep].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] < 2:
        return pd.DataFrame(columns=["fitur", "VIF", "tinggi", "n_baris"])

    corr = X.corr().abs()
    drop = set()
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] > 0.999 and cols[j] not in drop:
                drop.add(cols[j])
    X = X.drop(columns=list(drop))
    Xs = (X - X.mean()) / X.std(ddof=0)
    Xs = Xs.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    n = len(Xs)
    baris = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, col in enumerate(Xs.columns):
            try:
                v = float(variance_inflation_factor(Xs.values, k))
            except Exception:
                v = np.inf
            baris.append(dict(fitur=col, VIF=round(v, 3), n_baris=n))
    for c in drop:
        baris.append(dict(fitur=c, VIF=np.inf, n_baris=n))
    return (pd.DataFrame(baris).sort_values("VIF", ascending=False)
            .assign(tinggi=lambda d: d.VIF >= K.VIF_THRESHOLD))


def _redundansi_komposit(fitur: list[str], nmi: pd.DataFrame, spear: pd.DataFrame) -> pd.DataFrame:
    fset = set(fitur)
    baris = []
    for komposit, konst in getattr(K, "KOMPOSIT_KONSTITUEN", {}).items():
        if komposit not in fset:
            continue
        hadir = [c for c in konst if c in fset]
        if not hadir:
            continue
        for c in hadir:
            nmi_v = float(nmi.loc[komposit, c]) if c in nmi.columns and komposit in nmi.index else np.nan
            sp_v = float(spear.loc[komposit, c]) if c in spear.columns and komposit in spear.index else np.nan
            baris.append(dict(komposit=komposit, konstituen=c,
                              NMI=round(nmi_v, 4) if nmi_v == nmi_v else None,
                              spearman=round(sp_v, 4) if sp_v == sp_v else None))
        baris.append(dict(komposit=komposit, konstituen=f"[{len(hadir)} konstituen hadir]",
                          NMI=None, spearman=None))
    return pd.DataFrame(baris)


def _grup_fitur(fitur: list[str]) -> dict:
    kat = U.muat_kategori_fitur()
    komp = getattr(K, "KOMPOSIT_KONSTITUEN", {})
    grup = {}

    if kat:

        for komposit, konst in komp.items():
            kats = [kat[c] for c in konst if c in kat]
            if komposit not in kat and kats:
                kat[komposit] = max(set(kats), key=kats.count)
        for f in fitur:
            grup[f] = kat.get(f)

    for komposit, konst in komp.items():
        if grup.get(komposit) is None:
            grup[komposit] = komposit
        for k in konst:
            if grup.get(k) is None:
                grup[k] = komposit
    prefiks = {"percep_": "persepsi", "bansos_": "bansos", "imm_": "imunisasi",
               "food_": "pangan", "asset_": "aset", "anc_freq": "anc"}
    for f in fitur:
        if grup.get(f) is None:
            g = next((lab for p, lab in prefiks.items() if f.startswith(p)), None)
            grup[f] = g if g else f
    return grup


def _konstituen_dari(komposit: str) -> set:
    return set(getattr(K, "KOMPOSIT_KONSTITUEN", {}).get(komposit, []))


def _tipe_pasangan(fi, fj, komposit_set, grup, ukuran_grup) -> str:
    if (fi in komposit_set and fj in _konstituen_dari(fi)) or       (fj in komposit_set and fi in _konstituen_dari(fj)):
        return "rekayasa (komposit-konstituen)"

    gi, gj = grup.get(fi), grup.get(fj)
    if gi == gj and ukuran_grup.get(gi, 1) > 1:
        return "diharapkan (sekategori/semodul)"
    return "lintas-kategori (kandidat wawasan)"


def _pasangan_beranotasi(nmi: pd.DataFrame, spear: pd.DataFrame, fitur: list[str]) -> pd.DataFrame:
    komposit_set = set(getattr(K, "KOMPOSIT_KONSTITUEN", {}).keys()) & set(fitur)
    grup = _grup_fitur(fitur)
    ukuran = pd.Series(grup).value_counts().to_dict()
    f = list(nmi.columns)
    baris = []
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            sp = spear.iloc[i, j]
            baris.append(dict(
                fitur_i=f[i], fitur_j=f[j],
                NMI=round(float(nmi.iloc[i, j]), 4),
                spearman=round(float(sp), 4) if not pd.isna(sp) else np.nan,
                tipe=_tipe_pasangan(f[i], f[j], komposit_set, grup, ukuran),
            ))
    df = pd.DataFrame(baris)
    df["abs_spearman"] = df["spearman"].abs()
    df["skor_redundansi"] = df[["NMI", "abs_spearman"]].max(axis=1)
    return df.sort_values("skor_redundansi", ascending=False).reset_index(drop=True)


def _klaster(nmi: pd.DataFrame) -> pd.Series:
    fitur = list(nmi.columns)
    if len(fitur) < 2:
        return pd.Series({f: i + 1 for i, f in enumerate(fitur)})
    D = 1.0 - nmi.values
    D = np.clip(D, 0, 1)
    np.fill_diagonal(D, 0.0)
    D = np.nan_to_num(D, nan=1.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    label = fcluster(Z, t=K.REDUNDANCY_DIST_THRESHOLD, criterion="distance")
    return pd.Series(label, index=fitur)


def _kandidat_pemangkasan(klaster: pd.Series, nmi_target: pd.Series,
                          cakupan: pd.Series) -> pd.DataFrame:
    baris = []
    for cid, anggota in klaster.groupby(klaster):
        feats = list(anggota.index)
        if len(feats) == 1:
            f = feats[0]
            baris.append(dict(fitur=f, klaster=cid, anggota_klaster=1,
                              wakil=True, alasan="tunggal",
                              nmi_target=round(float(nmi_target.get(f, 0)), 4),
                              kelengkapan=round(float(cakupan.get(f, 0)), 4)))
            continue

        skor = pd.DataFrame({"nmi": nmi_target.reindex(feats).fillna(0),
                             "komplit": cakupan.reindex(feats).fillna(0)})
        wakil = skor.sort_values(["nmi", "komplit"], ascending=False).index[0]
        for f in feats:
            adalah_wakil = (f == wakil)
            baris.append(dict(
                fitur=f, klaster=cid, anggota_klaster=len(feats),
                wakil=adalah_wakil,
                alasan="dipertahankan (wakil)" if adalah_wakil else "kandidat dipangkas",
                nmi_target=round(float(nmi_target.get(f, 0)), 4),
                kelengkapan=round(float(cakupan.get(f, 0)), 4)))
    return pd.DataFrame(baris).sort_values(["klaster", "wakil"], ascending=[True, False])


def _heatmap(nmi: pd.DataFrame, path, judul: str):
    n = len(nmi)
    fig, ax = plt.subplots(figsize=(min(0.35 * n + 2, 20), min(0.35 * n + 2, 20)))
    im = ax.imshow(nmi.values, vmin=0, vmax=1, cmap="viridis")
    if n <= 40:
        ax.set_xticks(range(n)); ax.set_xticklabels(nmi.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(nmi.index, fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(judul, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NMI")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def jalankan_sel(nama: str, info: dict, df_imp: pd.DataFrame,
                 df_obs_src: pd.DataFrame | None) -> dict:
    fitur = info["fitur"]
    sumber, kohort = info["sumber"], info["kohort"]
    sub_imp = U.subset_sel(df_imp, sumber, kohort)
    if df_obs_src is not None:
        raw = U.subset_sel(df_obs_src, sumber, kohort)
        df_obs = raw[[c for c in fitur if c in raw]].copy()
    else:
        df_obs = U.tutup_balik_teramati(sub_imp, fitur)

    log.info("[Tahap 2] %s: %d fitur", nama, len(fitur))
    nmi = _matriks_nmi(df_obs, fitur)
    spear = df_obs[fitur].corr(method="spearman")
    y = U.ambil_target_biner(sub_imp)
    nmi_t = _nmi_dengan_target(df_obs, y, fitur)
    vif = _vif(df_obs, fitur)
    klaster = _klaster(nmi)
    komplit = df_obs[fitur].notna().mean()
    pangkas = _kandidat_pemangkasan(klaster, nmi_t, komplit)
    komposit = _redundansi_komposit(fitur, nmi, spear)


    semua = _pasangan_beranotasi(nmi, spear, fitur)

    teratas = semua.head(40)


    strukt = set(getattr(K, "STRUKTURAL_FITUR", []))
    bukan_umur = ~(semua.fitur_i.isin(strukt) | semua.fitur_j.isin(strukt))
    novel = semua[(semua.tipe == "lintas-kategori (kandidat wawasan)") & bukan_umur].head(25)

    cdir = K.cell_dir(nama)
    nmi.to_csv(cdir / "matriks_nmi.csv")
    spear.to_csv(cdir / "matriks_spearman.csv")
    vif.to_csv(cdir / "vif.csv", index=False)
    klaster.rename("klaster").to_csv(cdir / "klaster_redundansi.csv")
    pangkas.to_csv(cdir / "kandidat_pemangkasan.csv", index=False)
    teratas.to_csv(cdir / "pasangan_redundan_teratas.csv", index=False)
    novel.to_csv(cdir / "redundansi_novel_lintas_modul.csv", index=False)
    if len(komposit):
        komposit.to_csv(cdir / "redundansi_komposit.csv", index=False)
    _heatmap(nmi, cdir / "heatmap_nmi.png", f"NMI antar fitur - {nama}")

    n_pangkas = int((~pangkas.wakil).sum())
    n_komp = komposit["komposit"].nunique() if len(komposit) else 0
    top_novel = novel.iloc[0] if len(novel) else None
    log.info("  VIF tinggi(tanpa komposit)=%d/%d, komposit redundan=%d, novel lintas-kategori (skor>=0,3)=%d",
             int(vif.get("tinggi", pd.Series(dtype=bool)).sum()), len(vif), n_komp,
             int((novel.skor_redundansi >= 0.3).sum()) if len(novel) else 0)
    log.info("  redundansi NOVEL teratas: %s",
             f"{top_novel.fitur_i} x {top_novel.fitur_j} (skor={round(top_novel.skor_redundansi,3)}, NMI={top_novel.NMI}, rho={top_novel.spearman})"
             if top_novel is not None else "tidak ada")
    return dict(nmi=nmi, klaster=klaster, pangkas=pangkas, teratas=teratas,
                novel=novel, komposit=komposit)


def jalankan(hasil_tahap1: dict | None = None,
             df_imp: pd.DataFrame | None = None,
             df_obs_src: pd.DataFrame | None = None) -> dict:
    if df_imp is None:
        df_imp = U.muat_parquet(K.PARQUET_PATH)
    if df_obs_src is None and K.OBSERVED_PARQUET_PATH:
        df_obs_src = U.muat_parquet(K.OBSERVED_PARQUET_PATH)
    if hasil_tahap1 is None:
        import tahap1_pemuatan
        hasil_tahap1 = tahap1_pemuatan.jalankan(df_imp, df_obs_src)

    out = {}
    for nama, info in hasil_tahap1.items():
        out[nama] = jalankan_sel(nama, info, df_imp, df_obs_src)
    log.info("Tahap 2 selesai.")
    return out


if __name__ == "__main__":
    jalankan()
