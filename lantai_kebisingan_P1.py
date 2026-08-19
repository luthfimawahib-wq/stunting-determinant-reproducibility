import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr


CONFIG = dict(
    PARQUET_PATH   = "output_harmonisasi/stunting_harmonized.parquet",
    SOURCE_COL     = "source_flag",
    SOURCE_VALUES  = {"2024": "ssgi24", "2022": "ssgi22"},
    COHORT_COL     = "cohort",
    COHORT_VALUES  = {"younger": "baduta", "older": "balita_tua"},
    OUTCOME_COL    = "stunting_binary",

    EXCLUDE_COLS   = [

        "id_ruta", "svy_weight", "svy_psu", "svy_strata", "source_flag", "kohort",

        "height_child_cm", "weight_child_kg", "lila_child_cm",

        "haz_score", "waz_score", "whz_score", "stunting_binary",
    ],
    KEEP_AGE_SEX   = ["age_child_months", "sex_child"],

    N_IMPUTATIONS      = 5,
    N_SUBSAMPLE_SEEDS  = 5,
    SUBSAMPLE_N        = 2000,
    TOP_PAIRS          = 200,
    RANDOM_STATE       = 20260718,

    XGB_PARAMS = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        tree_method="hist", n_jobs=-1,
    ),
    OUT_CSV = "output_p1/lantai_kebisingan_hasil.csv",
)


def impute_once(X, seed):
    try:
        import miceforest as mf
        Xr = X.reset_index(drop=True)
        try:
            kernel = mf.ImputationKernel(Xr, num_datasets=1, random_state=seed)
        except TypeError:
            kernel = mf.ImputationKernel(Xr, datasets=1, random_state=seed)
        kernel.mice(5)
        out = kernel.complete_data(dataset=0)
        return out
    except Exception as e:
        print(f"    [peringatan] miceforest gagal ({type(e).__name__}: {e}); "
              f"memakai IterativeImputer sklearn (posterior sampling) sebagai cadangan.", flush=True)
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(sample_posterior=True, max_iter=5, random_state=seed)
        arr = imp.fit_transform(X)
        return pd.DataFrame(arr, columns=X.columns, index=X.index)


def fit_model(Xc, y):
    from xgboost import XGBClassifier
    m = XGBClassifier(**CONFIG["XGB_PARAMS"], random_state=CONFIG["RANDOM_STATE"])
    m.fit(Xc, y)
    return m

def _dmatrix(Xc):
    import xgboost as xgb
    return xgb.DMatrix(np.asarray(Xc, dtype=float))

def main_effect_vector(model, Xc):
    contribs = model.get_booster().predict(_dmatrix(Xc), pred_contribs=True,
                                            validate_features=False)
    imp = np.abs(contribs[:, :-1]).mean(axis=0)
    return pd.Series(imp, index=list(Xc.columns))

def interaction_vector(model, Xc, seed, subN, top_pairs):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xc), size=min(subN, len(Xc)), replace=False)
    Xs = Xc.iloc[idx]
    inter = model.get_booster().predict(_dmatrix(Xs), pred_interactions=True,
                                        validate_features=False)
    A = np.abs(inter).mean(axis=0)[:-1, :-1]
    cols = list(Xc.columns); pairs = {}
    for i, j in combinations(range(len(cols)), 2):
        pairs[(cols[i], cols[j])] = A[i, j]
    s = pd.Series(pairs)
    return s.sort_values(ascending=False).head(top_pairs)


def pairwise_spearman(vectors):
    keys = set(vectors[0].index)
    for v in vectors[1:]:
        keys &= set(v.index)
    keys = sorted(keys)
    if len(keys) < 3:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n_pairs=0, n_items=len(keys))
    aligned = [v.reindex(keys).values for v in vectors]
    rs = []
    for a, b in combinations(range(len(aligned)), 2):
        rho, _ = spearmanr(aligned[a], aligned[b])
        rs.append(rho)
    rs = np.array(rs)
    return dict(mean=float(np.mean(rs)), lo=float(np.percentile(rs, 2.5)),
                hi=float(np.percentile(rs, 97.5)), n_pairs=len(rs), n_items=len(keys))


def load_cell(df, source_val, cohort_val):
    m = (df[CONFIG["SOURCE_COL"]].astype(str).str.lower() == str(source_val).lower()) &        (df[CONFIG["COHORT_COL"]].astype(str).str.lower() == str(cohort_val).lower())
    sub = df.loc[m].copy()
    sub = sub[sub[CONFIG["OUTCOME_COL"]].notna()]
    y = pd.to_numeric(sub[CONFIG["OUTCOME_COL"]], errors="coerce").astype(int).values
    drop = set(CONFIG["EXCLUDE_COLS"]) | {CONFIG["SOURCE_COL"], CONFIG["COHORT_COL"], CONFIG["OUTCOME_COL"]}
    feat = [c for c in sub.columns if c not in drop]

    X = sub[feat].dropna(axis=1, how="all").copy()

    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")


    X = X.replace([np.inf, -np.inf], np.nan)
    nun = X.nunique(dropna=True)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
    return X, y

def run_cell(df, source_val, cohort_val, label):
    print(f"\n=== SEL {label}  (source={source_val}, cohort={cohort_val}) ===", flush=True)
    X, y = load_cell(df, source_val, cohort_val)
    print(f"    n={len(X)}, fitur kandidat={X.shape[1]}, prevalensi={y.mean():.4f}", flush=True)


    me_vecs, ix_vecs = [], []
    base_seed = CONFIG["RANDOM_STATE"]
    for d in range(CONFIG["N_IMPUTATIONS"]):
        Xc = impute_once(X, seed=base_seed + d)
        model = fit_model(Xc, y)
        me_vecs.append(main_effect_vector(model, Xc))
        ix_vecs.append(interaction_vector(model, Xc, seed=base_seed,
                                          subN=CONFIG["SUBSAMPLE_N"],
                                          top_pairs=CONFIG["TOP_PAIRS"]))
        print(f"    [imputasi draw {d+1}/{CONFIG['N_IMPUTATIONS']}] selesai", flush=True)
    floor_imp_me = pairwise_spearman(me_vecs)
    floor_imp_ix = pairwise_spearman(ix_vecs)


    Xc0 = impute_once(X, seed=base_seed)
    model0 = fit_model(Xc0, y)
    seed_vecs = []
    for s in range(CONFIG["N_SUBSAMPLE_SEEDS"]):
        seed_vecs.append(interaction_vector(model0, Xc0, seed=base_seed + 1000 + s,
                                            subN=CONFIG["SUBSAMPLE_N"],
                                            top_pairs=CONFIG["TOP_PAIRS"]))
        print(f"    [benih subsampel {s+1}/{CONFIG['N_SUBSAMPLE_SEEDS']}] selesai", flush=True)
    floor_sub_ix = pairwise_spearman(seed_vecs)

    return ([
        dict(sel=label, layer="main_effect", floor_type="imputation",
             **floor_imp_me),
        dict(sel=label, layer="interaction", floor_type="imputation",
             **floor_imp_ix),
        dict(sel=label, layer="interaction", floor_type="subsample_seed",
             **floor_sub_ix),
    ], me_vecs[0], ix_vecs[0])

def resolve_schema(df):
    cols = list(df.columns)
    print(f"    Parquet shape: {df.shape}", flush=True)
    print(f"    Kolom (semua {len(cols)}): {cols}", flush=True)

    def col_with_values(target_vals):

        for c in cols:
            try:
                vals = set(pd.Series(df[c].dropna().unique()).astype(str).str.lower())
            except Exception:
                continue
            if vals & target_vals:
                return c
        return None


    src_vals = {"ssgi22", "ssgi24", "ski23"}
    source_col = CONFIG["SOURCE_COL"]
    if (source_col not in df.columns) or not (
        set(df[source_col].astype(str).str.lower().unique()) & src_vals):
        source_col = col_with_values(src_vals)
    if source_col is None:
        raise SystemExit("Tidak menemukan kolom sumber (nilai ssgi22/ssgi24/ski23). "
                         "Set CONFIG['SOURCE_COL'] manual dari daftar kolom di atas.")
    CONFIG["SOURCE_COL"] = source_col
    print(f"    -> kolom sumber : '{source_col}' | nilai={sorted(set(df[source_col].astype(str).unique()))[:8]}", flush=True)


    coh_vals = {"baduta", "balita_tua"}
    cohort_col = CONFIG["COHORT_COL"]
    if (cohort_col not in df.columns) or not (
        set(df[cohort_col].astype(str).str.lower().unique()) & coh_vals):
        cohort_col = col_with_values(coh_vals)
    if cohort_col is None:
        age_col = None
        for cand in ["age_months", "umur_bulan", "usia_bulan", "age_month", "age", "umur", "usia"]:
            if cand in df.columns:
                age_col = cand; break
        if age_col is None:
            for c in cols:
                lc = c.lower()
                if ("age" in lc or "umur" in lc or "usia" in lc or "bulan" in lc or "month" in lc)                   and pd.api.types.is_numeric_dtype(df[c]):
                    mx = pd.to_numeric(df[c], errors="coerce").max()
                    if pd.notna(mx) and mx <= 72:
                        age_col = c; break
        if age_col is None:
            raise SystemExit("Tidak menemukan kolom kohort (baduta/balita_tua) maupun kolom umur "
                             "untuk menurunkannya. Set CONFIG['COHORT_COL'] manual.")
        age = pd.to_numeric(df[age_col], errors="coerce")
        df = df.copy()
        df["cohort"] = np.where(age < 24, "baduta", np.where(age <= 59, "balita_tua", np.nan))
        cohort_col = "cohort"
        print(f"    -> kohort diturunkan dari umur '{age_col}' (<24=baduta, 24-59=balita_tua)", flush=True)
    CONFIG["COHORT_COL"] = cohort_col
    print(f"    -> kolom kohort : '{cohort_col}' | nilai={sorted(set(df[cohort_col].dropna().astype(str).unique()))[:8]}", flush=True)


    if CONFIG["OUTCOME_COL"] not in df.columns:
        found = None
        for cand in ["stunting_binary", "stunting", "stunted", "stunting_bin", "is_stunting"]:
            if cand in df.columns:
                found = cand; break
        if found is None:
            haz = None
            for cand in ["haz", "zscore_haz", "haz_who", "z_haz", "haz2006"]:
                if cand in df.columns:
                    haz = cand; break
            if haz is None:
                raise SystemExit("Tidak menemukan kolom outcome stunting maupun HAZ. "
                                 "Set CONFIG['OUTCOME_COL'] manual.")
            df = df.copy()
            df["stunting_binary"] = (pd.to_numeric(df[haz], errors="coerce") < -2).astype(int)
            found = "stunting_binary"
            print(f"    -> outcome diturunkan dari '{haz}' (< -2 SD)", flush=True)
        CONFIG["OUTCOME_COL"] = found
    print(f"    -> kolom outcome: '{CONFIG['OUTCOME_COL']}'", flush=True)
    return df


def main():
    import os
    print("Memuat parquet:", CONFIG["PARQUET_PATH"], flush=True)
    df = pd.read_parquet(CONFIG["PARQUET_PATH"])
    df = resolve_schema(df)
    cells = [
        (CONFIG["SOURCE_VALUES"]["2024"], CONFIG["COHORT_VALUES"]["younger"], "A"),
        (CONFIG["SOURCE_VALUES"]["2024"], CONFIG["COHORT_VALUES"]["older"],   "B"),
        (CONFIG["SOURCE_VALUES"]["2022"], CONFIG["COHORT_VALUES"]["younger"], "C"),
        (CONFIG["SOURCE_VALUES"]["2022"], CONFIG["COHORT_VALUES"]["older"],   "D"),
    ]
    rows = []; me0 = {}; ix0 = {}
    for sv, cv, lab in cells:
        r, me, ix = run_cell(df, sv, cv, lab)
        rows.extend(r); me0[lab] = me; ix0[lab] = ix


    def _sp(a, b):
        keys = sorted(set(a.index) & set(b.index))
        if len(keys) < 3:
            return np.nan, len(keys)
        rho, _ = spearmanr(a.reindex(keys).values, b.reindex(keys).values)
        return float(rho), len(keys)
    for pair, (w24, w22) in {"younger (A vs C)": ("A", "C"),
                             "older (B vs D)":   ("B", "D")}.items():
        rme, nme = _sp(me0[w24], me0[w22])
        rix, nix = _sp(ix0[w24], ix0[w22])
        rows.append(dict(sel=pair, layer="main_effect", floor_type="CROSS_WAVE",
                         mean=rme, lo=np.nan, hi=np.nan, n_items=nme, n_pairs=1))
        rows.append(dict(sel=pair, layer="interaction", floor_type="CROSS_WAVE",
                         mean=rix, lo=np.nan, hi=np.nan, n_items=nix, n_pairs=1))

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CONFIG["OUT_CSV"]) or ".", exist_ok=True)
    res.to_csv(CONFIG["OUT_CSV"], index=False)

    pd.set_option("display.width", 160)
    print("\n========== LANTAI KEBISINGAN vs LINTAS-GELOMBANG (Spearman) ==========")
    print(res[["sel","layer","floor_type","mean","lo","hi","n_items","n_pairs"]].round(3).to_string(index=False))
    print(f"\nDisimpan ke: {CONFIG['OUT_CSV']}")

if __name__ == "__main__":
    main()
