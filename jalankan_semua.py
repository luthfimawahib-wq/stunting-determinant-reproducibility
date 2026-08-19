from __future__ import annotations
import time

import konfigurasi as K
import utilitas as U
import tahap1_pemuatan
import tahap2_redundansi
import tahap3_interaksi
import tahap4_kontras
import tahap5_sensitivitas

log = U.get_logger()


def main():
    t0 = time.time()
    df_imp = U.muat_parquet(K.PARQUET_PATH)
    df_obs = U.muat_parquet(K.OBSERVED_PARQUET_PATH) if K.OBSERVED_PARQUET_PATH else None

    log.info("=== TAHAP 1: pemuatan per-sel ===")
    h1 = tahap1_pemuatan.jalankan(df_imp, df_obs)

    log.info("=== TAHAP 2: redundansi ===")
    h2 = tahap2_redundansi.jalankan(h1, df_imp, df_obs)

    log.info("=== TAHAP 3: interaksi ===")
    h3 = tahap3_interaksi.jalankan(h1, df_imp)

    log.info("=== TAHAP 4: kontras kohort ===")
    tahap4_kontras.jalankan(h2, h3)

    log.info("=== TAHAP 5: sensitivitas ===")
    tahap5_sensitivitas.jalankan(h1, df_imp)

    log.info("SELESAI seluruh Pekerjaan A dalam %.1f menit. Output di: %s/",
             (time.time() - t0) / 60, K.OUTPUT_DIR)


if __name__ == "__main__":
    main()
