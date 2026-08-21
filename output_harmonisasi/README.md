# Master Parquet location (input)

Place `stunting_harmonized.parquet` in this folder before running the pipeline
pipeline. The file is NOT committed to Git (see `.gitignore`).

There are two ways to obtain it:

1. Run the harmonization repository on the real microdata, then copy the master
   Parquet here.
2. Run the synthetic test-data generator in the harmonization repository, then
   copy the synthetic Parquet here to exercise the pipeline without real data.

Adjust the file name and paths in `konfigurasi.py` if your layout differs.
