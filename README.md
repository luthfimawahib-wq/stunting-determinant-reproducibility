# Reproducibility of Model-Derived Predictor Importance Rankings for Childhood Stunting Across Age Cohorts and National Survey Waves

Analysis code for a measurement-validity study of model-derived predictor
importance rankings for childhood stunting, using two harmonized waves of the
Indonesia Nutritional Status Survey (SSGI 2022 and SSGI 2024).

This repository does not develop or validate a deployable prediction model. A
gradient-boosted tree is used solely as a measuring instrument, and the object of
measurement is the predictor importance ranking itself: how far the recovered
ranking structure survives when the population, the survey wave, the imputation,
or the model specification changes. Reproducibility is summarized by a single
named estimand, the rank-based reproducibility coefficient, defined as the
Spearman rank correlation between two importance profiles on the exact set of
items shared by the two cells compared.

Throughout this repository, *predictor* refers to a model input and to
model-derived variable importance. The term *determinant* is reserved for the
epidemiologic literature and is not used for model output. Some legacy column
names in the stored outputs still carry the older wording; the mapping table
below states which reported quantity each column corresponds to.

## Data foundation

This repository operates on a harmonized master dataset produced by a separate
foundation repository:

```
Stunting Harmonization Pipeline
https://github.com/luthfimawahib-wq/stunting-harmonization
DOI: 10.5281/zenodo.22015038
```

The SSGI microdata are held by the Ministry of Health of the Republic of
Indonesia and are not publicly redistributable. The harmonized master Parquet is
a derivative of that microdata and is never committed to this repository.

## Source dataset and analyzed population

The harmonized master dataset holds **721,385 records** from three national
sources (SSGI 2022, SSGI 2024, and SKI 2023). This study analyzes the two SSGI waves only:

```
harmonized master                                721,385 records
  |
  +-- SKI 2023, reserved as an independent
  |   external validator for a separate study      86,364 records
  |
  +-- SSGI 2022 and SSGI 2024                     635,021 records
        |
        +-- no observed stunting outcome            1,283 records
        |
        +-- analyzed here                         633,738 records
              |
              +-- A  SSGI 2024, 0-23 months       105,216
              +-- B  SSGI 2024, 24-59 months      193,694
              +-- C  SSGI 2022, 0-23 months       124,752
              +-- D  SSGI 2022, 24-59 months      210,076
```

SKI (Indonesian Health Survey) 2023 is outside the two-wave design of this study and is never loaded by
this pipeline. Of the 635,021 SSGI records, 1,283 have no observed stunting
outcome; these are dropped where the instrument is fitted, in
`tahap3_interaksi.py`, and `n_latih` in each `<cell>/model_meta.json` records the
resulting per-cell count. Stage 1 removes no rows.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10.4. Library versions are pinned to those reported in the manuscript,
because version agreement is required to reproduce the reported values.

## Input

Place the harmonized master dataset at:

```
output_harmonisasi/stunting_harmonized.parquet
```

Two supporting files from the harmonization repository are read when present and
improve feature typing and redundancy labelling:

```
output_harmonisasi/skema_encoding.json        encoding schema, source of truth for column types
output_harmonisasi/matriks_ketersediaan.csv   feature availability metadata with domain categories
```

Open `konfigurasi.py` and adjust paths if your layout differs.

## Running

Run the whole pipeline:

```
python jalankan_semua.py
```

Or run stages individually, in order:

```
python tahap1_pemuatan.py
python tahap2_redundansi.py
python tahap3_interaksi.py
python tahap4_kontras.py
python tahap5_sensitivitas.py
python lantai_kebisingan_P1.py
```

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

| Cell | Source    | Age cohort   | Analyzed records | Screened predictors |
| ---- | --------- | ------------ | ------- | ------------------- |
| A    | SSGI 2024 | 0-23 months  | 105,216 | 75 |
| B    | SSGI 2024 | 24-59 months | 193,694 | 77 |
| C    | SSGI 2022 | 0-23 months  | 124,752 | 82 |
| D    | SSGI 2022 | 24-59 months | 210,076 | 46 |

An anti-leakage protocol excludes outcome-forming anthropometry, so the
instrument cannot reconstruct the outcome from its constituents. Age and sex are
retained, with age treated as a conditioned structural axis rather than as an
actionable determinant.

## Stages and outputs

| Stage | Script | Main outputs in `output_p1/` |
| ----- | ------ | ---------------------------- |
| 1. Per-cell loading | `tahap1_pemuatan.py` | `cakupan_fitur_per_sel.csv`, `ringkasan_fitur_per_sel.csv`, `<cell>/fitur_lolos.txt`, `<cell>/cakupan_fitur.csv` |
| 2. Redundancy | `tahap2_redundansi.py` | `<cell>/matriks_nmi.csv`, `matriks_spearman.csv`, `pasangan_redundan_teratas.csv`, `redundansi_komposit.csv`, `redundansi_novel_lintas_modul.csv`, `klaster_redundansi.csv`, `kandidat_pemangkasan.csv`, `vif.csv`, `heatmap_nmi.png` |
| 3. Main effects and interactions | `tahap3_interaksi.py` | `<cell>/matriks_interaksi.csv`, `pasangan_interaksi_teratas.csv`, `efek_utama.csv`, `heatmap_interaksi.png`, `model_meta.json` |
| 4. Cohort contrast and replication | `tahap4_kontras.py` | `kontras_kohort/kontras_stabilitas_marginal_vs_interaksi.csv`, `replikasi_ringkasan.csv`, `replikasi_delta.csv`, `ringkasan_kontras.csv`, `<contrast>_delta_interaksi.csv`, `<contrast>_fitur_eksklusif.csv`, `<contrast>_scatter_interaksi.png` |
| 5. Imputation sensitivity | `tahap5_sensitivitas.py` | `sensitivitas/ringkasan_sensitivitas_semua_sel.csv`, `ringkasan_sensitivitas.json`, `stabilitas_pasangan_<cell>.csv` |
| 6. Robustness analyses | `lantai_kebisingan_P1.py` | `lantai_kebisingan_hasil.csv`, `lantai_vektor/` |

## Robustness analyses

`lantai_kebisingan_P1.py` carries two complementary analyses plus the cross-wave
comparison they are judged against. All three are computed in one re-estimation
under a single unified pipeline, separate from the primary analysis, so its point
values are not those of the primary analysis; the comparison rests on the
ordering rather than on the values themselves.

**Subsample-seed noise floor.** Holding the data and the fitted model fixed, five
random 2,000-record subsamples are drawn and the interaction profile recomputed
for each. The floor is the mean pairwise Spearman correlation between the
resulting profiles, bounded by the 2.5th and 97.5th percentiles. This varies
nuisance only and is therefore a measurement-noise floor.

**Model-specification sensitivity.** Holding the imputation and the subsample
seed fixed, the instrument is refitted under seven specifications and agreement
is summarized as the mean pairwise Spearman correlation across arms, with the
Jaccard overlap of the top-ranked interaction pairs. This varies the model
deliberately and is a sensitivity analysis, not noise.

Anchor specification (`SPEC_BASE_PARAMS`): 300 trees, `max_depth` 4,
`learning_rate` 0.1, `subsample` 0.8, `colsample_bytree` 0.8, `reg_lambda` 1.0,
`random_state` 42, class weighting active.

| Arm | Change from the anchor |
| --- | ---------------------- |
| `baseline` | none |
| `shallow_depth2` | `max_depth` 2 |
| `deep_depth6` | `max_depth` 6 |
| `slow_learning` | `learning_rate` 0.05, 600 trees |
| `unweighted` | class weighting off |
| `alt_seed` | `random_state` 20260719 |
| `reestimation_spec` | 400 trees, `learning_rate` 0.05, `reg_lambda` 0.0, class weighting off |

**Cross-wave comparison.** The same pipeline recomputes cross-wave
reproducibility for each cohort, which is then compared with the lower bound of
each analysis above.

**Imputation floor.** The script also computes an imputation floor over five
draws. It is written to the output file but is **not reported in the
manuscript**: the only missing values remaining in the harmonized dataset belong
to the antenatal-care block, which the primary analysis deliberately leaves
unimputed so that absence of antenatal contact is carried as signal, so varying
its imputation would not reflect the procedure actually applied.

Other settings: interaction subsample 2,000 records, top-ranked pairs 200, base
seed 20260718, instrument for the seed and imputation floors 400 trees,
`max_depth` 4, `learning_rate` 0.05.

`lantai_kebisingan_hasil.csv` distinguishes the analyses through its
`floor_type` column: `subsample_seed`, `model_spec_sensitivity`,
`spec_vs_baseline` (one row per arm per layer per cell), `imputation`, and
`CROSS_WAVE`. `lantai_vektor/` holds the underlying importance vectors, one file
per cell, layer, and condition, so every summary value in the output file can be
recomputed from the vectors it was derived from.

## Mapping from computation to reported tables and figures

| Reported item | Script | Output file | Field or rows |
| ------------- | ------ | ----------- | ------------- |
| Table 1, analyzed records | `tahap3_interaksi.py` | `<cell>/model_meta.json` | `n_latih` |
| Table 1, screened predictors | `tahap1_pemuatan.py` | `ringkasan_fitur_per_sel.csv` | `n_fitur_lolos` |
| Table 1, in-sample AUC | `tahap3_interaksi.py` | `<cell>/model_meta.json` | `auc_latih_cek_waras` |
| Table 2, rows 1 to 6 | `tahap4_kontras.py` | `kontras_kohort/kontras_stabilitas_marginal_vs_interaksi.csv` | `spearman`, `p_permutasi` |
| Table 2, age-excluded row | `tahap4_kontras.py` | `kontras_kohort/replikasi_ringkasan.csv` | `spearman_determinan`, `n_pasangan_determinan` |
| Table 3 | `tahap4_kontras.py` | `kontras_kohort/ringkasan_kontras.csv` | `n_fitur_bersama`, `jaccard_top` |
| Table 4 | `tahap5_sensitivitas.py` | `sensitivitas/ringkasan_sensitivitas_semua_sel.csv` | `spearman_obs_vs_imp`, `jaccard_topK_obs_vs_imp` |
| Table 5 | `lantai_kebisingan_P1.py` | `lantai_kebisingan_hasil.csv` | `floor_type` = `subsample_seed`, `model_spec_sensitivity` |
| Table 6 | `lantai_kebisingan_P1.py` | `lantai_kebisingan_hasil.csv` | `floor_type` = `CROSS_WAVE`, against the `lo` of the rows above |
| Table 7 | `tahap3_interaksi.py` | `<cell>/matriks_interaksi.csv` | diagonal against off-diagonal, each pair counted once |
| Table 8 | `tahap2_redundansi.py` | `<cell>/redundansi_novel_lintas_modul.csv` | highest-ranked cross-module pair |
| Figure 1 | `tahap4_kontras.py` | `kontras_kohort/kontras_stabilitas_marginal_vs_interaksi.csv` | `spearman`, `p_permutasi` |
| Figure 2 | `tahap5_sensitivitas.py` | `sensitivitas/ringkasan_sensitivitas_semua_sel.csv` | `spearman_obs_vs_imp`, `jaccard_topK_obs_vs_imp` |
| Figure 3 | `tahap3_interaksi.py` | `<cell>/matriks_interaksi.csv` | diagonal against off-diagonal share |
| Supplementary Table S1, rows 1 to 6 | `tahap4_kontras.py` | `kontras_kohort/kontras_stabilitas_marginal_vs_interaksi.csv` | `ci_lo`, `ci_hi` |
| Supplementary Tables S2 and S3 | `tahap1_pemuatan.py` with `utilitas.py` | `ringkasan_fitur_per_sel.csv`, `<cell>/cakupan_fitur.csv`, `<cell>/fitur_lolos.txt` | global candidate set and per-cell screening counts |
| Supplementary Table S4 | `lantai_kebisingan_P1.py` | `konfigurasi` block in the script | `SPEC_BASE_PARAMS`, `SPEC_ARMS` |
| Supplementary Table S5 | `lantai_kebisingan_P1.py` | `lantai_kebisingan_hasil.csv` | `floor_type` = `spec_vs_baseline`, one row per arm |

Notes on legacy column names. In `replikasi_ringkasan.csv`, the fields
`spearman_determinan` and `n_pasangan_determinan` carry the age-excluded
interaction comparison reported in the manuscript as the age-excluded row of
Table 2; the name predates the terminology used in the manuscript. In
`kontras_stabilitas_marginal_vs_interaksi.csv`, the columns `ci_lo` and `ci_hi`
hold the item-resampling sensitivity range reported in Supplementary Table S1,
not a confidence interval for a population parameter; the manuscript states why
no interval estimate is reported. In both files, `marginal` denotes the
main-effect layer.

The resampling sensitivity range for the age-excluded row of Supplementary
Table S1 is computed by the same procedure but is not written to the output
file by this version of the pipeline.

## Configuration

Key settings in `konfigurasi.py`:

- `FEATURE_SET`: `"kaya"` uses every substantive feature populated in each cell;
  `"universal"` restricts to the canonical universal feature list in
  `fitur_universal_45.txt`. This study reports the `"kaya"` setting. Outcome-forming
  anthropometry is removed in either mode.
- `SENSITIVITY_MODE`: this study reports `"observed_vs_imputed"`. The
  `"mice_draws_on_cell"` and `"bootstrap"` modes are alternative settings that
  were not used for the reported results.
- `MIN_COMPLETENESS`, `MAX_ROWS_SHAP_INT`, `TOP_K_PAIRS`, and `SEED` control
  feature screening, the interaction subsample, ranking depth, and determinism.

Settings for the robustness analyses live in the `CONFIG` block at the top of
`lantai_kebisingan_P1.py` and are listed in the section above.

## Citation

If you use this software, cite the archived release listed in `CITATION.cff` and
`.zenodo.json`. The harmonization pipeline is archived separately and should be
cited alongside it when the full workflow is reproduced.

## License

MIT. See `LICENSE`.
