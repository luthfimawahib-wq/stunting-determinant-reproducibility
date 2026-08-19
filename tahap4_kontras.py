from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import konfigurasi as K
import utilitas as U

log = U.get_logger()


def _pasangan_panjang(M: pd.DataFrame) -> pd.Series:
    fitur = list(M.columns)
    data = {}
    for i in range(len(fitur)):
        for j in range(i + 1, len(fitur)):
            data[f"{fitur[i]}||{fitur[j]}"] = float(M.iloc[i, j])
    return pd.Series(data, name="interaksi")


def _efek_utama(M: pd.DataFrame) -> pd.Series:
    return pd.Series(np.diag(M.values), index=list(M.columns), name="efek_utama")


def _perm_null_spearman(x: pd.Series, y: pd.Series, n: int = 1000, seed: int = 0):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 5:
        return (np.nan, np.nan, np.nan, np.nan)
    rho = spearmanr(d.x, d.y).correlation
    rng = np.random.default_rng(seed)
    yv = d.y.values
    null = np.empty(n)
    for i in range(n):
        null[i] = spearmanr(d.x.values, rng.permutation(yv)).correlation
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n + 1))
    return (float(rho), p, float(np.mean(null)), float(np.percentile(null, 95)))


def _boot_ci_spearman(x: pd.Series, y: pd.Series, n: int = 2000, seed: int = 7):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(d))
    xs, ys = d.x.values, d.y.values
    bs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        r = spearmanr(xs[s], ys[s]).correlation
        if r == r:
            bs.append(r)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan, np.nan)


def _jaccard_acak(k: int, n_total: int) -> float:
    if n_total <= 0 or k <= 0:
        return float("nan")
    inter = k * k / n_total
    union = 2 * k - inter
    return float(inter / union) if union > 0 else float("nan")


def _delta_interaksi(Ma: pd.DataFrame, Mb: pd.DataFrame, label_a: str, label_b: str):
    fitur_bersama = [f for f in Ma.columns if f in Mb.columns]
    Ma2 = Ma.loc[fitur_bersama, fitur_bersama]
    Mb2 = Mb.loc[fitur_bersama, fitur_bersama]
    sa = _pasangan_panjang(Ma2).rename(label_a)
    sb = _pasangan_panjang(Mb2).rename(label_b)
    df = pd.concat([sa, sb], axis=1).dropna()
    df["delta"] = df[label_a] - df[label_b]
    df["abs_delta"] = df["delta"].abs()
    df = df.sort_values("abs_delta", ascending=False).reset_index()
    df = df.rename(columns={"index": "pasangan"})

    if len(df) >= 3:
        rho, _ = spearmanr(df[label_a], df[label_b])
    else:
        rho = np.nan

    topk = min(K.TOP_K_PAIRS, len(df))
    set_a = set(_pasangan_panjang(Ma2).sort_values(ascending=False).head(topk).index)
    set_b = set(_pasangan_panjang(Mb2).sort_values(ascending=False).head(topk).index)
    jacc = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else np.nan
    return df, fitur_bersama, rho, jacc


def _fitur_eksklusif(fitur_a, fitur_b, kohort_a, kohort_b):
    eks_a = sorted(set(fitur_a) - set(fitur_b))
    eks_b = sorted(set(fitur_b) - set(fitur_a))
    baris = [dict(fitur=f, hanya_di=kohort_a) for f in eks_a] +            [dict(fitur=f, hanya_di=kohort_b) for f in eks_b]
    return pd.DataFrame(baris)


def _scatter(df, label_a, label_b, path, judul):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[label_b], df[label_a], s=10, alpha=0.5)
    lim = max(df[label_a].max(), df[label_b].max()) * 1.05 + 1e-9
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    for _, r in df.head(8).iterrows():
        ax.annotate(r["pasangan"], (r[label_b], r[label_a]), fontsize=5)
    ax.set_xlabel(f"interaksi ({label_b})"); ax.set_ylabel(f"interaksi ({label_a})")
    ax.set_title(judul, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def jalankan(hasil_tahap2: dict, hasil_tahap3: dict) -> dict:
    kdir = K.output_dir() / "kontras_kohort"
    kdir.mkdir(parents=True, exist_ok=True)

    ringkas = []
    delta_simpan = {}
    for nama_kontras, (sel_a, sel_b) in K.CONTRASTS.items():
        Ma = hasil_tahap3[sel_a]["matriks"]; Mb = hasil_tahap3[sel_b]["matriks"]
        kohort_a = K.CELLS[sel_a][1]; kohort_b = K.CELLS[sel_b][1]
        label_a, label_b = kohort_a, kohort_b

        df, bersama, rho, jacc = _delta_interaksi(Ma, Mb, label_a, label_b)
        eks = _fitur_eksklusif(list(Ma.columns), list(Mb.columns), kohort_a, kohort_b)

        df.to_csv(kdir / f"{nama_kontras}_delta_interaksi.csv", index=False)
        eks.to_csv(kdir / f"{nama_kontras}_fitur_eksklusif.csv", index=False)
        _scatter(df, label_a, label_b, kdir / f"{nama_kontras}_scatter_interaksi.png",
                 f"Interaksi per pasangan: {kohort_a} vs {kohort_b} ({nama_kontras})")

        delta_simpan[nama_kontras] = df.set_index("pasangan")["delta"]
        ringkas.append(dict(kontras=nama_kontras, sel_a=sel_a, sel_b=sel_b,
                            n_fitur_bersama=len(bersama),
                            spearman_peringkat=round(float(rho), 4) if rho == rho else None,
                            jaccard_top=round(float(jacc), 4) if jacc == jacc else None,
                            n_fitur_eksklusif=int(len(eks))))
        log.info("[Tahap 4] %s: fitur bersama=%d, Spearman=%.3f, Jaccard top-%d=%.3f",
                 nama_kontras, len(bersama), rho if rho == rho else float("nan"),
                 K.TOP_K_PAIRS, jacc if jacc == jacc else float("nan"))

    pd.DataFrame(ringkas).to_csv(kdir / "ringkasan_kontras.csv", index=False)


    kunci = list(delta_simpan.keys())
    if len(kunci) >= 2:
        d1 = delta_simpan[kunci[0]].rename(kunci[0])
        d2 = delta_simpan[kunci[1]].rename(kunci[1])
        rep = pd.concat([d1, d2], axis=1).dropna()
        if len(rep) >= 3:


            mags = []
            for sel in K.CELLS:
                if sel in hasil_tahap3:
                    mags.append(_pasangan_panjang(hasil_tahap3[sel]["matriks"]))
            mag = pd.concat(mags, axis=1).mean(axis=1) if mags else pd.Series(dtype=float)
            rep["magnitudo"] = mag.reindex(rep.index)
            rep["tanda_sama"] = np.sign(rep[kunci[0]]) == np.sign(rep[kunci[1]])

            rho_rep, _ = spearmanr(rep[kunci[0]], rep[kunci[1]])
            tanda_polos = 100 * rep["tanda_sama"].mean()


            topk = min(K.TOP_K_PAIRS, len(rep))
            rep_top = rep.nlargest(topk, "magnitudo")
            rho_top, _ = spearmanr(rep_top[kunci[0]], rep_top[kunci[1]]) if len(rep_top) >= 3 else (np.nan, None)
            tanda_top = 100 * rep_top["tanda_sama"].mean()


            w = rep["magnitudo"].fillna(0)
            tanda_bobot = 100 * (rep["tanda_sama"] * w).sum() / w.sum() if w.sum() > 0 else np.nan


            strukt = set(getattr(K, "STRUKTURAL_FITUR", []))
            def _tanpa_strukt(idx):
                a, b = idx.split("||")
                return (a not in strukt) and (b not in strukt)
            mask_det = rep.index.map(_tanpa_strukt)
            rep_det = rep[mask_det]
            rho_det, _ = spearmanr(rep_det[kunci[0]], rep_det[kunci[1]]) if len(rep_det) >= 3 else (np.nan, None)
            tanda_det = 100 * rep_det["tanda_sama"].mean() if len(rep_det) else np.nan
            rep_det_top = rep_det.nlargest(min(K.TOP_K_PAIRS, len(rep_det)), "magnitudo")
            rho_det_top, _ = spearmanr(rep_det_top[kunci[0]], rep_det_top[kunci[1]]) if len(rep_det_top) >= 3 else (np.nan, None)
            tanda_det_top = 100 * rep_det_top["tanda_sama"].mean() if len(rep_det_top) else np.nan

            rep.sort_values("magnitudo", ascending=False).to_csv(kdir / "replikasi_delta.csv")
            pd.DataFrame([dict(
                spearman_semua=round(float(rho_rep), 4),
                tanda_sama_semua_pct=round(float(tanda_polos), 1),
                spearman_topK_bermakna=round(float(rho_top), 4) if rho_top == rho_top else None,
                tanda_sama_topK_pct=round(float(tanda_top), 1),
                tanda_sama_dibobot_pct=round(float(tanda_bobot), 1) if tanda_bobot == tanda_bobot else None,
                spearman_determinan=round(float(rho_det), 4) if rho_det == rho_det else None,
                tanda_sama_determinan_pct=round(float(tanda_det), 1) if tanda_det == tanda_det else None,
                spearman_determinan_topK=round(float(rho_det_top), 4) if rho_det_top == rho_det_top else None,
                tanda_sama_determinan_topK_pct=round(float(tanda_det_top), 1) if tanda_det_top == tanda_det_top else None,
                n_pasangan=len(rep), n_pasangan_determinan=int(len(rep_det)), top_k=topk,
            )]).to_csv(kdir / "replikasi_ringkasan.csv", index=False)

            log.info("[Tahap 4] Replikasi kohort 2024 vs 2022:")
            log.info("   semua pasangan      : Spearman=%.3f, tanda sama=%.1f%%", rho_rep, tanda_polos)
            log.info("   top-%d bermakna      : Spearman=%.3f, tanda sama=%.1f%%",
                     topk, rho_top if rho_top == rho_top else float("nan"), tanda_top)
            log.info("   tanda dibobot magnitudo=%.1f%%", tanda_bobot if tanda_bobot == tanda_bobot else float("nan"))
            log.info("   antar-determinan     : Spearman=%.3f, tanda sama=%.1f%% (top-K tanda=%.1f%%)",
                     rho_det if rho_det == rho_det else float("nan"),
                     tanda_det if tanda_det == tanda_det else float("nan"),
                     tanda_det_top if tanda_det_top == tanda_det_top else float("nan"))


    def _cari(src, koh):
        return next((s for s, (a, b) in K.CELLS.items() if a == src and b == koh), None)
    selA, selB = _cari("ssgi24", "baduta"), _cari("ssgi24", "balita_tua")
    selC, selD = _cari("ssgi22", "baduta"), _cari("ssgi22", "balita_tua")

    if all(s in hasil_tahap3 for s in (selA, selB, selC, selD)):
        ME = {s: _efek_utama(hasil_tahap3[s]["matriks"]) for s in (selA, selB, selC, selD)}
        IX = {s: _pasangan_panjang(hasil_tahap3[s]["matriks"]) for s in (selA, selB, selC, selD)}

        def _align_spear(x, y):
            d = pd.concat([x, y], axis=1).dropna()
            return (spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation, len(d)) if len(d) >= 5 else (np.nan, len(d))

        baris = []

        for lap, dat in (("marginal", ME), ("interaksi", IX)):
            rho_bad, p_bad, nm_bad, _ = _perm_null_spearman(dat[selA], dat[selC])
            lo_bad, hi_bad = _boot_ci_spearman(dat[selA], dat[selC])
            rho_bal, p_bal, nm_bal, _ = _perm_null_spearman(dat[selB], dat[selD])
            lo_bal, hi_bal = _boot_ci_spearman(dat[selB], dat[selD])
            baris.append(dict(lapisan=lap, ukuran="transfer_baduta_2024v2022",
                              spearman=round(rho_bad, 4) if rho_bad == rho_bad else None,
                              ci_lo=round(lo_bad, 4) if lo_bad == lo_bad else None,
                              ci_hi=round(hi_bad, 4) if hi_bad == hi_bad else None,
                              p_permutasi=round(p_bad, 4) if p_bad == p_bad else None,
                              null_mean=round(nm_bad, 4) if nm_bad == nm_bad else None))
            baris.append(dict(lapisan=lap, ukuran="transfer_balita_2024v2022",
                              spearman=round(rho_bal, 4) if rho_bal == rho_bal else None,
                              ci_lo=round(lo_bal, 4) if lo_bal == lo_bal else None,
                              ci_hi=round(hi_bal, 4) if hi_bal == hi_bal else None,
                              p_permutasi=round(p_bal, 4) if p_bal == p_bal else None,
                              null_mean=round(nm_bal, 4) if nm_bal == nm_bal else None))

        for lap, dat in (("marginal", ME), ("interaksi", IX)):
            shared_24 = dat[selA].index.intersection(dat[selB].index)
            shared_22 = dat[selC].index.intersection(dat[selD].index)
            d24 = (dat[selA].reindex(shared_24) - dat[selB].reindex(shared_24)).rename("d24")
            d22 = (dat[selC].reindex(shared_22) - dat[selD].reindex(shared_22)).rename("d22")
            both = pd.concat([d24, d22], axis=1).dropna()
            rho, p, nm, _ = _perm_null_spearman(both["d24"], both["d22"]) if len(both) >= 5 else (np.nan, np.nan, np.nan, None)
            lo, hi = _boot_ci_spearman(both["d24"], both["d22"]) if len(both) >= 5 else (np.nan, np.nan)
            tanda = 100 * (np.sign(both["d24"]) == np.sign(both["d22"])).mean() if len(both) else np.nan
            baris.append(dict(lapisan=lap, ukuran="replikasi_delta_kohort",
                              spearman=round(rho, 4) if rho == rho else None,
                              ci_lo=round(lo, 4) if lo == lo else None,
                              ci_hi=round(hi, 4) if hi == hi else None,
                              p_permutasi=round(p, 4) if p == p else None,
                              null_mean=round(nm, 4) if nm == nm else None,
                              tanda_sama_pct=round(float(tanda), 1) if tanda == tanda else None))

        stab = pd.DataFrame(baris)
        stab.to_csv(kdir / "kontras_stabilitas_marginal_vs_interaksi.csv", index=False)
        log.info("[Tahap 4] KONTRAS STABILITAS (kaki unit pengetahuan), marginal vs interaksi:")
        for _, r in stab.iterrows():
            log.info("   %-10s %-26s Spearman=%s CI[%s, %s] (p_perm=%s)%s",
                     r["lapisan"], r["ukuran"], r["spearman"], r.get("ci_lo"), r.get("ci_hi"),
                     r["p_permutasi"],
                     f", tanda={r.get('tanda_sama_pct')}%" if "tanda_sama_pct" in r and pd.notna(r.get("tanda_sama_pct")) else "")

    log.info("Tahap 4 selesai.")
    return dict(ringkasan=ringkas)


if __name__ == "__main__":
    import tahap1_pemuatan, tahap2_redundansi, tahap3_interaksi
    df_imp = U.muat_parquet(K.PARQUET_PATH)
    h1 = tahap1_pemuatan.jalankan(df_imp)
    h2 = tahap2_redundansi.jalankan(h1, df_imp)
    h3 = tahap3_interaksi.jalankan(h1, df_imp)
    jalankan(h2, h3)
