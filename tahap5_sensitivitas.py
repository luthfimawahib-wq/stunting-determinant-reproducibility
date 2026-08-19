from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import konfigurasi as K
import utilitas as U
import tahap3_interaksi as T3

log = U.get_logger()


def _seri_interaksi(X: pd.DataFrame, y: pd.Series, fitur: list[str]) -> pd.Series:
    import tahap4_kontras as T4
    model, _ = T3._latih_model(X[fitur], y)
    M = T3._matriks_interaksi(model, X[fitur])
    return T4._pasangan_panjang(pd.DataFrame(M, index=fitur, columns=fitur))


def _siapkan_xy(sub: pd.DataFrame, fitur: list[str]):
    y = U.ambil_target_biner(sub)
    X = sub[fitur].apply(pd.to_numeric, errors="coerce")
    kosong = [c for c in X.columns if X[c].notna().sum() == 0]
    if kosong:
        X[kosong] = 0.0
    m = y.notna()
    return X.loc[m], y.loc[m].astype(int)


def _stabilitas(seri: list[pd.Series], label: list[str]) -> pd.DataFrame:
    mat = pd.concat([s.rename(l) for s, l in zip(seri, label)], axis=1)
    peringkat = mat.rank(ascending=False)
    topk = K.TOP_K_PAIRS
    out = pd.DataFrame({
        "interaksi_mean": mat.mean(axis=1),
        "interaksi_sd": mat.std(axis=1, ddof=0),
        "peringkat_mean": peringkat.mean(axis=1),
        "peringkat_sd": peringkat.std(axis=1, ddof=0),
        "freq_dalam_topK": (peringkat <= topk).sum(axis=1),
        "n_dataset": mat.notna().sum(axis=1),
    })
    out["proporsi_topK"] = out["freq_dalam_topK"] / len(label)
    return out.sort_values("peringkat_mean").reset_index().rename(columns={"index": "pasangan"})


def _mode_observed_vs_imputed(sub_imp, df_obs, fitur):
    Xi, yi = _siapkan_xy(sub_imp, fitur)
    s_imp = _seri_interaksi(Xi, yi, fitur)

    obs = df_obs[fitur].copy()
    obs[K.TARGET_FOR_MODEL] = sub_imp[K.TARGET_FOR_MODEL].values
    Xo, yo = _siapkan_xy(obs, fitur)
    s_obs = _seri_interaksi(Xo, yo, fitur)
    stab = _stabilitas([s_obs, s_imp], ["teramati", "terimputasi"])
    rho, _ = spearmanr(s_obs.reindex(stab.pasangan), s_imp.reindex(stab.pasangan),
                       nan_policy="omit")
    topk = min(K.TOP_K_PAIRS, len(stab))
    jacc_a = set(s_obs.sort_values(ascending=False).head(topk).index)
    jacc_b = set(s_imp.sort_values(ascending=False).head(topk).index)
    jacc = len(jacc_a & jacc_b) / len(jacc_a | jacc_b) if (jacc_a | jacc_b) else float("nan")
    extra = dict(spearman_obs_vs_imp=round(float(rho), 4),
                 jaccard_topK_obs_vs_imp=round(float(jacc), 4))
    return stab, extra


def _mode_mice_draws(df_obs, sub_imp, fitur):
    import miceforest as mf
    draws, label = [], []
    base = df_obs[fitur].copy().reset_index(drop=True)
    y_target = pd.to_numeric(sub_imp[K.TARGET_FOR_MODEL].values, errors="coerce")
    for d in range(K.N_MICE_DRAWS):
        kernel = mf.ImputationKernel(base, num_datasets=1, random_state=K.SEED + d)
        kernel.mice(K.MICE_ITERATIONS)
        Xc = kernel.complete_data(dataset=0)
        Xc[K.TARGET_FOR_MODEL] = y_target
        Xc, yc = _siapkan_xy(Xc, fitur)
        draws.append(_seri_interaksi(Xc, yc, fitur)); label.append(f"draw{d+1}")
    return _stabilitas(draws, label), {}


def _mode_bootstrap(sub_imp, fitur):
    seri, label = [], []
    for b in range(K.N_BOOTSTRAP):
        boot = sub_imp.sample(n=len(sub_imp), replace=True, random_state=K.SEED + b)
        Xb, yb = _siapkan_xy(boot, fitur)
        seri.append(_seri_interaksi(Xb, yb, fitur)); label.append(f"boot{b+1}")
    return _stabilitas(seri, label), {}


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

    sdir = K.output_dir() / "sensitivitas"; sdir.mkdir(parents=True, exist_ok=True)
    daftar = getattr(K, "SENSITIVITY_CELLS", None)
    if daftar in (None, "all"):
        daftar = list(K.CELLS.keys())
    elif isinstance(daftar, str):
        daftar = [daftar]

    rekap = []
    for sel in daftar:
        info = hasil_tahap1[sel]
        fitur = info["fitur"]
        sub_imp = U.subset_sel(df_imp, info["sumber"], info["kohort"])
        if df_obs_src is not None:
            raw = U.subset_sel(df_obs_src, info["sumber"], info["kohort"])
            df_obs = raw[[c for c in fitur if c in raw]].copy()
        else:
            df_obs = U.tutup_balik_teramati(sub_imp, fitur)

        mode = K.SENSITIVITY_MODE
        extra = {}
        log.info("[Tahap 5] Mode=%s pada sel %s (p=%d)", mode, sel, len(fitur))
        if mode == "mice_draws_on_cell":
            try:
                stab, extra = _mode_mice_draws(df_obs, sub_imp, fitur); n_ds = K.N_MICE_DRAWS
            except Exception as e:
                log.warning("  mice_draws gagal (%s); jatuh ke observed_vs_imputed.", e)
                mode = "observed_vs_imputed"; stab, extra = _mode_observed_vs_imputed(sub_imp, df_obs, fitur); n_ds = 2
        elif mode == "bootstrap":
            stab, _ = _mode_bootstrap(sub_imp, fitur); n_ds = K.N_BOOTSTRAP
        else:
            mode = "observed_vs_imputed"
            stab, extra = _mode_observed_vs_imputed(sub_imp, df_obs, fitur); n_ds = 2

        stab.to_csv(sdir / f"stabilitas_pasangan_{sel}.csv", index=False)
        top = stab.head(K.TOP_K_PAIRS)
        ringkas = dict(
            sel=sel, mode=mode, n_dataset=int(n_ds), top_k=K.TOP_K_PAIRS,
            rata_proporsi_topK_pada_top=round(float(top["proporsi_topK"].mean()), 3),
            rata_peringkat_sd_pada_top=round(float(top["peringkat_sd"].mean()), 3),
            **extra,
        )
        rekap.append(ringkas)
        log.info("[Tahap 5] %s selesai (%s). %s", sel, mode,
                 {k: v for k, v in ringkas.items() if k not in ("sel", "mode", "top_k")})

    rekap_df = pd.DataFrame(rekap)
    rekap_df.to_csv(sdir / "ringkasan_sensitivitas_semua_sel.csv", index=False)
    (sdir / "ringkasan_sensitivitas.json").write_text(
        json.dumps(dict(per_sel=rekap,
                        catatan=("observed_vs_imputed menilai apakah imputasi mengubah "
                                 "struktur interaksi. Diuji di semua sel; Sel D paling "
                                 "rapuh (fitur paling sedikit, instrumen terlemah).")),
                   indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[Tahap 5] Selesai semua sel. Ringkasan:\n%s", rekap_df.to_string(index=False))
    return rekap


if __name__ == "__main__":
    jalankan()
