# Reproducibility of Stunting Determinant Rankings Across Age Cohorts and National Survey Waves

Analysis code for a measurement-validity study of model-derived determinant
rankings for childhood stunting, using two harmonized waves of the Indonesia
Nutritional Status Survey (SSGI 2022 and SSGI 2024).

This repository does not develop or validate a deployable prediction model. A
gradient-boosted tree is used solely as a measuring instrument, and the object
of measurement is the determinant ranking itself: how far the recovered
determinant-ranking structure survives when the population, the survey wave, or
the imputation changes. Reproducibility is summarized by a single named
estimand, the rank-based reproducibility coefficient, defined as the Spearman
rank correlation between two importance profiles.

## Data foundation

This repository operates on a harmonized master dataset produced by a separate
foundation repository:

    Stunting Harmonization Pipeline
    https://github.com/luthfimawahib-wq/stunting-harmonization
    DOI: 10.5281/zenodo.22015038

The SSGI microdata are held by the Ministry of Health of the Republic of
Indonesia and are not publicly redistributable. The harmonized master Parquet is
a derivative of that microdata and is never committed to this repository.

## Requirements

    pip install -r requirements.txt

Python 3.10.4. Library versions are pinned to those reported in the manuscript,
because version agreement is required to reproduce the reported values.

## Input

Place the harmonized master dataset at:

    output_harmonisasi/stunting_harmonized.parquet

Two supporting files from the harmonization repository are read when present and
improve feature typing and redundancy labelling:

    output_harmonisasi/skema_encoding.json        encoding schema, source of truth for column types
    output_harmonisasi/matriks_ketersediaan.csv   feature availability metadata with domain categories

Open `konfigurasi.py` and adjust paths if your layout differs.

## Running

Run the whole pipeline:

    python jalankan_semua.py

Or run stages individually, in order:

    python tahap1_pemuatan.py
    python tahap2_redundansi.py
    python tahap3_interaksi.py
    python tahap4_kontras.py
    python tahap5_sensitivitas.py
    python lantai_kebisingan_P1.py

All results are written to `output_p1/`.

## Running without the restricted microdata

The harmonization repository includes a synthetic-data generator that produces a
Parquet file with the same schema. Generate it there, copy it to
`output_harmonisasi/`, and the full pipeline in this repository can be executed
and verified without access to the microdata.

## Design

Four analysis cells are formed by crossing two survey waves with two age
cohorts. Each cell is analyzed on the feature set populated in that cell, without
concatenating waves into one matrix, so that structural missingness between
waves does not create artifacts.

| Cell | Source    | Age cohort    |
|------|-----------|---------------|
| A    | SSGI 2024 | 0-23 months   |
| B    | SSGI 2024 | 24-59 months  |
| C    | SSGI 2022 | 0-23 months   |
| D    | SSGI 2022 | 24-59 months  |

An anti-leakage protocol excludes outcome-forming anthropometry, so the
instrument cannot reconstruct the outcome from its constituents. Age and sex are
retained, with age treated as a conditioned structural axis rather than as an
actionable determinant.

## Stages and outputs

| Stage | Script | Main outputs in `output_p1/` |
|-------|--------|------------------------------|
| 1. Per-cell loading | `tahap1_pemuatan.py` | `cakupan_fitur_per_sel.csv`, `ringkasan_fitur_per_sel.csv`, `<cell>/fitur_lolos.txt`, `<cell>/cakupan_fitur.csv` |
| 2. Redundancy | `tahap2_redundansi.py` | `<cell>/matriks_nmi.csv`, `matriks_spearman.csv`, `pasangan_redundan_teratas.csv`, `redundansi_komposit.csv`, `redundansi_novel_lintas_modul.csv`, `klaster_redundansi.csv`, `kandidat_pemangkasan.csv`, `vif.csv`, `heatmap_nmi.png` |
| 3. Main effects and interactions | `tahap3_interaksi.py` | `<cell>/matriks_interaksi.csv`, `pasangan_interaksi_teratas.csv`, `efek_utama.csv`, `heatmap_interaksi.png`, `model_meta.json` |
| 4. Cohort contrast and replication | `tahap4_kontras.py` | `kontras_kohort/<contrast>_delta_interaksi.csv`, `<contrast>_fitur_eksklusif.csv`, `<contrast>_scatter_interaksi.png`, `ringkasan_kontras.csv`, `replikasi_delta.csv`, `replikasi_ringkasan.csv`, `kontras_stabilitas_marginal_vs_interaksi.csv` |
| 5. Imputation sensitivity | `tahap5_sensitivitas.py` | `sensitivitas/stabilitas_pasangan_<cell>.csv`, `ringkasan_sensitivitas_semua_sel.csv`, `ringkasan_sensitivitas.json` |
| 6. Measurement-noise floor | `lantai_kebisingan_P1.py` | `lantai_kebisingan_hasil.csv` |

## Configuration

Key settings in `konfigurasi.py`:

- `FEATURE_SET`: `"kaya"` uses every substantive feature populated in each cell;
  `"universal"` restricts to the canonical universal feature list in
  `fitur_universal_45.txt`. Outcome-forming anthropometry is removed in either
  mode, so the universal mode yields 43 effective predictors.
- `SENSITIVITY_MODE`: `"observed_vs_imputed"`, `"mice_draws_on_cell"`, or
  `"bootstrap"`.
- `MIN_COMPLETENESS`, `MAX_ROWS_SHAP_INT`, `TOP_K_PAIRS`, and `SEED` control
  feature screening, the interaction subsample, ranking depth, and determinism.

## Citation

If you use this software, cite the archived release listed in `CITATION.cff` and `.zenodo.json`.
The harmonization pipeline is archived separately and should be cited alongside
it when the full workflow is reproduced.

## License

MIT. See `LICENSE`.
