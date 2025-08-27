import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors


os.environ["ENVIRONMENT"] = "PRED"
# os.environ["ENVIRONMENT"] = "TEST"

SEED = 42
rng = np.random.default_rng(SEED)

# --- Load raw (keep a copy for display) ---
csv_path = "startUpAgent_Backend/statistic_search/data/growjo_desc.csv"
df_raw = pd.read_csv(csv_path)

# Columns we want to show in outputs (if present)
display_cols = [c for c in ["company_name","Industry","Industry_Group","country",
                            "ranking","founded","current_employees",
                            "total_funding","valuation","description"]
                if c in df_raw.columns]

# --- Build feature frame (drop only IDs / text you don't want as features) ---
df = df_raw.copy()
for c in ["id","company_name","valuation_as_of","ranking","Points"]:  # remove only from features
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# Optional: year from founded
if "founded" in df.columns:
    df["founded"] = pd.to_datetime(df["founded"], errors="coerce", format="%Y").dt.year

df["country"] = df["country"].apply(lambda x : str(x).lower())

# Split types
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

# Rare-category bucketing
# RARE = 5
# for c in cat_cols:
#     if c != "Industry_Group":
#         vc = df[c].value_counts(dropna=False)
#         rare = vc[vc < RARE].index
#         df[c] = df[c].where(~df[c].isin(rare), "__OTHER__")

# --- IMPORTANT: exclude free-text from features (keeps OHE sane) ---
TEXT_COLS = ["description"]  # add any other long text cols here
X_df = df.drop(columns=[c for c in TEXT_COLS if c in df.columns])

# Preprocessor (fit only on TRAIN later to avoid leakage)
num_cols_use = X_df.select_dtypes(include=["number"]).columns.tolist()
cat_cols_use = [c for c in X_df.columns if c not in num_cols_use]
pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols_use),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols_use),
    ],
    remainder="drop",
)

N = len(X_df)
idx = np.arange(N)
rng.shuffle(idx)

# --- Train/val split by row indices ---
if os.environ["ENVIRONMENT"] == "TEST":

    n_val = max(50, int(0.2 * N))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    pre.fit(X_df.iloc[train_idx])
    X_train = pre.transform(X_df.iloc[train_idx])
    X_val   = pre.transform(X_df.iloc[val_idx])
    # print(f"Data size after Feature Engineering : {X_train.shape}")
else:
    train_idx = idx
    pre.fit(X_df.iloc[train_idx])
    X_train = pre.transform(X_df.iloc[train_idx])
    # print(f"Data size after Feature Engineering : {X_train.shape}")

# Fit ONLY on train to avoid leakage





# # kNN index on train (cosine works well with mixed/scaled features)
# nn = NearestNeighbors(metric="cosine", n_neighbors=min(50, len(X_df)), algorithm="brute")
# nn.fit(X_train)

# ========== Helpers ==========

def _apply_constraints(df_base: pd.DataFrame, ids: np.ndarray, constraints: dict | None):
    """Filter candidate ids by constraints:
       - equality: {"Industry":"AI"}
       - in-list: {"country":["US","CA"]}
       - numeric range: {"current_employees": (50, 300)}"""
    if not constraints:
        return ids
    mask = np.ones(len(df_base), dtype=bool)
    for col, cond in constraints.items():
        if col not in df_base.columns:
            continue
        s = df_base[col]
        if isinstance(cond, tuple) and len(cond) == 2 and pd.api.types.is_numeric_dtype(s):
            lo, hi = cond
            mask &= s.between(lo, hi, inclusive="both")
        elif isinstance(cond, (list, set, tuple)):
            mask &= s.isin(list(cond))
        else:
            mask &= (s == cond)
    return ids[mask[ids]]

def _mini_preprocessor_for(use_cols: list[str], train_ids: np.ndarray):
    """Build a preprocessor on a subset of columns (fit on TRAIN only)."""
    use_cols = [c for c in use_cols if c in X_df.columns]
    use_num = [c for c in use_cols if pd.api.types.is_numeric_dtype(X_df[c])]
    use_cat = [c for c in use_cols if c not in use_num]
    pre_use = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), use_num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))]), use_cat),
        ],
        remainder="drop",
    )

    pre_use.fit(X_df.iloc[train_ids][use_cols])

    return pre_use, use_cols


def recommend_topk_for_input(
    user_input: dict,
    k: int = 10,
    constraints: dict | None = None,   # filter candidate pool
    use_cols: list[str] | None = None, # restrict similarity columns
    show_cols: list[str] | None = None # columns to display
) -> pd.DataFrame:
    """Return top-k similar TRAIN items to a one-row user input."""
    # Candidate pool after constraints
    cand_ids = _apply_constraints(df_raw, train_idx, constraints)

    if cand_ids.size == 0:
        return pd.DataFrame({"msg": [f"No candidates match constraints: {constraints}"]})
    
    if "country" in user_input.keys():
        user_input["country"] = user_input["country"].lower()

    # Preprocessor choice
    if use_cols is None:
        pre_use = pre
        cols_used = X_df.columns
        X_cand = pre_use.transform(X_df.iloc[cand_ids])
        # Build a one-row DataFrame covering all cols_used
        q_row = pd.DataFrame([{col: user_input.get(col, np.nan) for col in cols_used}], columns=cols_used)
                
        qX = pre_use.transform(q_row)
    else:
        pre_use, cols_used = _mini_preprocessor_for(use_cols, train_idx)
        X_cand = pre_use.transform(X_df.iloc[cand_ids][cols_used])
        q_row = pd.DataFrame([{col: user_input.get(col, np.nan) for col in cols_used}], columns=cols_used)
        qX = pre_use.transform(q_row)

    # kNN over candidate subset
    k_eff = min(k, len(cand_ids))
    nn_local = NearestNeighbors(metric="cosine", n_neighbors=k_eff, algorithm="brute").fit(X_cand)
    dists, inds = nn_local.kneighbors(qX, n_neighbors=k_eff)
    sims = 1.0 - dists[0]
    top_ids = cand_ids[inds[0]]

    # Prepare display
    default_show = display_cols if display_cols else df_raw.columns.tolist()
    show_cols = show_cols or default_show
    out = df_raw.iloc[top_ids][[c for c in show_cols if c in df_raw.columns]].copy()
    out.insert(0, "rank_in_list", np.arange(1, len(top_ids)+1))
    out.insert(1, "sim_cosine", np.round(sims, 4))
    return out

if __name__ == "__main__":
    user = {
        "Industry_Group": "Artificial Intelligence",
        "country": "Taiwan",
        "current_employees": 54,
        "total_funding": 25_000_000,
        "founded": 2014,
        "current_objectives": ["hit $1.5M ARR in 12 months", "land first enterprise logo"],
        "strengths": ["excellent multilingual support", "lightweight API"],
        "weaknesses": ["limited admin features", "no SOC2", "weak sales motion"]
    }
    top10 = recommend_topk_for_input(
        user_input=user, k=10,
        # constraints={"Industry": "Artificial Intelligence", "country": "United States"},
        use_cols=None,  # or specify: ["Industry","country","current_employees","employee_growth","total_funding","founded"]
        show_cols=display_cols
    )

    def ordered(row):
        return f"- Company {row['rank_in_list']}\n" + row["description"]

    top10["description"] = top10.apply(ordered, axis=1)

    print(top10.head(10))