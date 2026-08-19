from pathlib import Path

PARQUET_PATH = "output_harmonisasi/stunting_harmonized.parquet"


SKEMA_ENCODING_PATH = "output_harmonisasi/skema_encoding.json"


MATRIKS_KATEGORI_PATH = "output_harmonisasi/matriks_ketersediaan.csv"


OBSERVED_PARQUET_PATH = None

OUTPUT_DIR = "output_p1"


SOURCE_COL = "source_flag"
COHORT_COL = "kohort"

CELLS = {
    "A_ssgi24_baduta":     ("ssgi24", "baduta"),
    "B_ssgi24_balita_tua": ("ssgi24", "balita_tua"),
    "C_ssgi22_baduta":     ("ssgi22", "baduta"),
    "D_ssgi22_balita_tua": ("ssgi22", "balita_tua"),
}


CONTRASTS = {
    "kontras_2024_AvsB": ("A_ssgi24_baduta", "B_ssgi24_balita_tua"),
    "kontras_2022_CvsD": ("C_ssgi22_baduta", "D_ssgi22_balita_tua"),
}


SENSITIVITY_CELL = "A_ssgi24_baduta"


SENSITIVITY_CELLS = "all"


TARGET_COLS = [
    "haz", "waz", "whz", "haz_score", "waz_score", "whz_score",
    "stunting_binary", "zscore_tb_u",
]
TARGET_FOR_MODEL = "stunting_binary"


LEAKAGE_EXCLUDE_COLS = [
    "height_child_cm", "weight_child_kg",
    "tinggi_anak_cm", "berat_anak_kg", "tinggi_badan", "berat_badan",
]


KONKUREN_ANTRO_EXCLUDE = ["lila_child_cm", "weight_gain_trend"]


EXTRA_EXCLUDE_COLS = [
    "svy_weight", "svy_psu", "svy_strata", "id_ruta",
    "measure_position", "posisi_ukur",
    "provinsi", "province", "kabupaten", "kecamatan", "desa", "kelurahan",
    "puskesmas", "posyandu", "tahun", "year", "nik",
]

META_PREFIXES = ["svy_"]


MISSING_INDICATOR_PATTERNS = ["{f}_missing", "mi_{f}", "{f}_mi", "miss_{f}", "{f}_isna"]
INCLUDE_MISSING_INDICATORS = False


FEATURE_SET = "kaya"


UNIVERSAL_FEATURES_PATH = "output_harmonisasi/fitur_universal_45.txt"


KOMPOSIT_KONSTITUEN = {
    "wealth_index": ["asset_gas", "asset_washing_machine", "asset_fridge",
                     "asset_phone", "asset_computer", "asset_tv",
                     "asset_motorcycle", "asset_car", "asset_gold",
                     "asset_land", "asset_livestock"],
    "ANC_total_visit": ["anc_freq_doc_t1", "anc_freq_doc_t2", "anc_freq_doc_t3",
                        "anc_freq_mid_t1", "anc_freq_mid_t2", "anc_freq_mid_t3"],
    "food_diversity_score": ["food_water", "food_formula", "food_cereal",
                             "food_vit_a_veg", "food_green_veg", "food_vit_a_fruit",
                             "food_organ_meat", "food_meat", "food_egg",
                             "food_fish", "food_legume"],
    "TTD_compliance": ["ttd_count", "ttd_received"],
    "imunisasi_lengkap": ["imm_hepb0", "imm_bcg", "imm_dpt1", "imm_dpt2",
                          "imm_dpt3", "imm_dpt_boost", "imm_pcv", "imm_polio",
                          "imm_measles_9mo", "imm_measles_boost"],
}


STRUKTURAL_FITUR = ["age_child_months"]


MIN_COMPLETENESS = 0.30
MAX_UNIQUE_CATEGORICAL = 15
N_BINS = 10
MAX_ROWS_MI = 50000
MAX_ROWS_SHAP_INT = 2000


REDUNDANCY_DIST_THRESHOLD = 0.7
VIF_THRESHOLD = 10.0
TOP_K_PAIRS = 30
SEED = 42


SENSITIVITY_MODE = "observed_vs_imputed"
N_MICE_DRAWS = 5
MICE_ITERATIONS = 5
N_BOOTSTRAP = 20

DRAW_PARQUET_PATHS = []


XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    eval_metric="logloss", n_jobs=-1, random_state=SEED, tree_method="hist",
)
USE_SCALE_POS_WEIGHT = True


def output_dir() -> Path:
    p = Path(OUTPUT_DIR); p.mkdir(parents=True, exist_ok=True); return p

def cell_dir(cell_name: str) -> Path:
    p = output_dir() / cell_name; p.mkdir(parents=True, exist_ok=True); return p
