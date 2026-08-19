from __future__ import annotations
import pandas as pd

import konfigurasi as K
import utilitas as U

log = U.get_logger()


def jalankan(df_imp: pd.DataFrame | None = None,
             df_obs_src: pd.DataFrame | None = None) -> dict:
    if df_imp is None:
        df_imp = U.muat_parquet(K.PARQUET_PATH)
    if df_obs_src is None and K.OBSERVED_PARQUET_PATH:
        df_obs_src = U.muat_parquet(K.OBSERVED_PARQUET_PATH)

    fitur_kandidat = U.kolom_fitur(df_imp)
    log.info("Fitur kandidat global: %d", len(fitur_kandidat))

    hasil = {}
    rekap = []
    for nama, (sumber, kohort) in K.CELLS.items():
        sub_imp = U.subset_sel(df_imp, sumber, kohort)
        if df_obs_src is not None:
            sub_obs_raw = U.subset_sel(df_obs_src, sumber, kohort)
            df_obs = sub_obs_raw[[c for c in fitur_kandidat if c in sub_obs_raw]].copy()
        else:
            df_obs = U.tutup_balik_teramati(sub_imp, fitur_kandidat)

        lolos, cakupan = U.pilih_fitur_terisi(df_obs, fitur_kandidat)
        log.info("Sel %-22s n=%-8s fitur lolos=%d", nama, f"{len(sub_imp):,}", len(lolos))

        cdir = K.cell_dir(nama)
        cakupan.to_csv(cdir / "cakupan_fitur.csv", index=False)
        (cdir / "fitur_lolos.txt").write_text("\n".join(lolos), encoding="utf-8")

        hasil[nama] = dict(sumber=sumber, kohort=kohort, n=len(sub_imp),
                           fitur=lolos)
        c = cakupan.copy()
        c.insert(0, "sel", nama)
        rekap.append(c)

    rekap_df = pd.concat(rekap, ignore_index=True)
    rekap_df.to_csv(K.output_dir() / "cakupan_fitur_per_sel.csv", index=False)


    ringkas = pd.DataFrame([
        dict(sel=k, sumber=v["sumber"], kohort=v["kohort"], n_baris=v["n"],
             n_fitur_lolos=len(v["fitur"]))
        for k, v in hasil.items()
    ])
    ringkas.to_csv(K.output_dir() / "ringkasan_fitur_per_sel.csv", index=False)
    log.info("Tahap 1 selesai. Ringkasan:\n%s", ringkas.to_string(index=False))
    return hasil


if __name__ == "__main__":
    jalankan()
