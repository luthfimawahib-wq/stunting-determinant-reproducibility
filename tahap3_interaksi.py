from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import konfigurasi as K
import utilitas as U

log = U.get_logger()


def _latih_model(X: pd.DataFrame, y: pd.Series):
    import xgboost as xgb
    params = dict(K.XGB_PARAMS)
    if K.USE_SCALE_POS_WEIGHT:
        n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
        params["scale_pos_weight"] = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model, params.get("scale_pos_weight", 1.0)


def _matriks_interaksi(model, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb
    Xs = U.subsample(X, K.MAX_ROWS_SHAP_INT)
    try:
        booster = model.get_booster()
        dm = xgb.DMatrix(Xs, missing=np.nan, feature_names=list(Xs.columns))
        inter = np.asarray(booster.predict(dm, pred_interactions=True))

        if inter.ndim == 4:
            inter = inter[:, -1, :, :]

        if inter.ndim == 3 and inter.shape[1] == Xs.shape[1] + 1:
            inter = inter[:, :-1, :-1]
        return np.abs(inter).mean(axis=0)
    except Exception as e:
        log.warning("pred_interactions natif gagal (%s); fallback ke shap.TreeExplainer.", e)
        import shap
        si = np.asarray(shap.TreeExplainer(model).shap_interaction_values(Xs))
        if si.ndim == 4:
            si = si[..., -1]
        return np.abs(si).mean(axis=0)


def _pasangan_teratas(M: np.ndarray, fitur: list[str]) -> pd.DataFrame:
    p = len(fitur)
    baris = []
    for i in range(p):
        for j in range(i + 1, p):
            baris.append(dict(fitur_i=fitur[i], fitur_j=fitur[j],
                              interaksi=float(M[i, j])))
    df = pd.DataFrame(baris).sort_values("interaksi", ascending=False).reset_index(drop=True)
    df.insert(0, "peringkat", df.index + 1)
    return df


def _heatmap(M: np.ndarray, fitur: list[str], path, judul: str):
    Mo = M.copy(); np.fill_diagonal(Mo, 0.0)
    n = len(fitur)
    fig, ax = plt.subplots(figsize=(min(0.35 * n + 2, 20), min(0.35 * n + 2, 20)))
    im = ax.imshow(Mo, cmap="magma")
    if n <= 40:
        ax.set_xticks(range(n)); ax.set_xticklabels(fitur, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(fitur, fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(judul, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rata-rata |interaksi SHAP|")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def jalankan_sel(nama: str, info: dict, df_imp: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    fitur = info["fitur"]
    sub = U.subset_sel(df_imp, info["sumber"], info["kohort"])
    y = U.ambil_target_biner(sub)
    X = sub[fitur].apply(pd.to_numeric, errors="coerce")


    kosong_total = [c for c in X.columns if X[c].notna().sum() == 0]
    if kosong_total:
        X[kosong_total] = 0.0
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    log.info("[Tahap 3] %s: latih XGBoost instrumen (n=%s, p=%d)",
             nama, f"{len(X):,}", len(fitur))
    model, spw = _latih_model(X, y)
    try:
        auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    except Exception:
        auc = float("nan")

    M = _matriks_interaksi(model, X)
    Mdf = pd.DataFrame(M, index=fitur, columns=fitur)
    pasangan = _pasangan_teratas(M, fitur)
    efek_utama = pd.DataFrame({"fitur": fitur, "efek_utama": np.diag(M)})        .sort_values("efek_utama", ascending=False)

    cdir = K.cell_dir(nama)
    Mdf.to_csv(cdir / "matriks_interaksi.csv")
    pasangan.to_csv(cdir / "pasangan_interaksi_teratas.csv", index=False)
    efek_utama.to_csv(cdir / "efek_utama.csv", index=False)
    _heatmap(M, fitur, cdir / "heatmap_interaksi.png", f"Interaksi SHAP - {nama}")
    (cdir / "model_meta.json").write_text(json.dumps(dict(
        n_latih=int(len(X)), p=len(fitur), scale_pos_weight=round(float(spw), 3),
        auc_latih_cek_waras=round(auc, 4),
        n_sampel_shap=min(K.MAX_ROWS_SHAP_INT or len(X), len(X)),
        catatan="model adalah instrumen pengukur interaksi, bukan klaim prediktif"
    ), indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("  AUC latih (cek waras)=%.3f, pasangan teratas: %s x %s (%.4g)",
             auc, pasangan.iloc[0].fitur_i, pasangan.iloc[0].fitur_j,
             pasangan.iloc[0].interaksi)
    return dict(matriks=Mdf, pasangan=pasangan)


def jalankan(hasil_tahap1: dict | None = None,
             df_imp: pd.DataFrame | None = None) -> dict:
    if df_imp is None:
        df_imp = U.muat_parquet(K.PARQUET_PATH)
    if hasil_tahap1 is None:
        import tahap1_pemuatan
        hasil_tahap1 = tahap1_pemuatan.jalankan(df_imp)
    out = {}
    for nama, info in hasil_tahap1.items():
        out[nama] = jalankan_sel(nama, info, df_imp)
    log.info("Tahap 3 selesai.")
    return out


if __name__ == "__main__":
    jalankan()
