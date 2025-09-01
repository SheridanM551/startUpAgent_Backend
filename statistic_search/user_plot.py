# Update: numeric columns use BOX PLOTS (with marker) and we emit JSON-friendly stats
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import json
from startUpAgent_Backend.statistic_search.topk_company import *
from typing import Dict, List, Optional, Tuple
from startUpAgent_Backend import config
# ---------- Helpers ----------
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _safe_percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float((s <= value).mean() * 100.0)

def _pick_group(df: pd.DataFrame, user_input: Dict, group_key: Optional[str]) -> pd.DataFrame:
    if group_key and group_key in df.columns and group_key in user_input:
        return df[df[group_key] == user_input[group_key]]
    return df

def _describe_numeric_boxplot(colname: str, stats: Dict, minmax=False) -> str:
    if stats is None or stats.get("count", 0) == 0:
        return f"No valid numeric data for {colname}."
    yv = stats.get("your_value")
    pct = stats.get("percentile")
    q1 = stats.get("q1"); med = stats.get("median"); q3 = stats.get("q3")
    lw = stats.get("lower_whisker"); uw = stats.get("upper_whisker")
    count = stats.get("count")
    outlier_flag = None
    if yv is not None and lw is not None and uw is not None:
        if yv < lw:
            outlier_flag = "below the lower whisker (low outlier)"
        elif yv > uw:
            outlier_flag = "above the upper whisker (high outlier)"
    parts = []
    parts.append(f"For **{colname}** , n={count}.")
    if minmax:
        min_ = stats.get("min")
        max_ = stats.get("max")
        parts.append(f"The IQR spans [{q1:,.3g}, {q3:,.3g}] with median {med:,.3g}, minimum vlaue {min_:,.3g} and , maximum vlaue {max_:,.3g}.")
    else:
        parts.append(f"The IQR spans [{q1:,.3g}, {q3:,.3g}] with median {med:,.3g}.")
    if yv is not None and pct is not None:
        parts.append(f"User's input value is {yv:,.3g}, around the {pct:.1f}th percentile.")
    if outlier_flag:
        parts.append(f"This lies {outlier_flag}.")
    else:
        parts.append("This lies within the whisker range." if yv is not None else "")

    if minmax:
        min_ = stats.get("min")
        max_ = stats.get("max")
    return " ".join(p for p in parts if p)

def _describe_categorical(colname: str, payload: Dict) -> str:
    if not payload or not payload.get("counts"):
        return f"No valid categorical data for {colname}."
    total = sum(payload["counts"].values())
    uv = payload.get("your_value")
    rank = payload.get("rank")
    rate = payload.get("rate")
    top_cat = max(payload["counts"], key=lambda k: payload["counts"][k]) if total > 0 else None
    lines = [f"For **{colname}** , n={total}. The most common category is '{top_cat}' ({payload['counts'][top_cat]})."]
    if uv is not None:
        if rate is not None:
            lines.append(f"User's category '{uv}' has rank {rank} and accounts for {rate:.2f}% of entries.")
        else:
            lines.append(f"User's category '{uv}' does not appear in the data.")
    return " ".join(lines)

# ---------- New numeric plotter: BOX PLOT + stats ----------
def _boxplot_with_marker_and_stats(series: pd.Series, value: float, title: str, minmax:bool=False):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or pd.isna(value):
        print(f"[Skip] {title}: no numeric data or user value is NaN.")
        return None
    
    # Compute Tukey boxplot stats
    q1 = float(s.quantile(0.25))
    q2 = float(s.quantile(0.50))  # median
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower_whisker = float(max(s.min(), q1 - 1.5 * iqr))
    upper_whisker = float(min(s.max(), q3 + 1.5 * iqr))
    outliers = [float(v) for v in s[(s < lower_whisker) | (s > upper_whisker)].tolist()]
    percentile = _safe_percentile_rank(s, value)
    
    # Plot (horizontal box plot) + marker for user's value
    # plt.figure()
    # plt.boxplot(s, vert=False, widths=0.6)
    # plt.scatter([value], [1], marker="X", s=100)  # highlight the user point
    # plt.title(title + f"\nYour value={value:.3g}, Percentile≈{percentile:.1f}%")
    # plt.xlabel(series.name)
    # plt.show()
    
    # Return JSON-friendly stats
    payload = {
        "q1": q1,
        "median": q2,
        "q3": q3,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers": outliers,
        "your_value": float(value) if value is not None and not pd.isna(value) else None,
        "percentile": float(percentile) if not pd.isna(percentile) else None,
        "count": int(len(s)),
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
    }

    payload["description"] = _describe_numeric_boxplot(series.name , payload, minmax=minmax)
    return payload

def _plot_categorical_location(series: pd.Series, user_value, title: str, top_k: int = 20):
    # kept for completeness; not a box plot
    counts = series.value_counts(dropna=True)
    if len(counts) == 0:
        print(f"[Skip] {title}: no categorical data.")
        return {"counts": {}, "your_value": user_value, "rank": None, "rate": None, "top_k": top_k}
    top_counts = counts
    # Plot bar just for local inspection (can be removed if not needed)
    # plt.figure()
    top_counts.plot(kind="bar")
    # plt.title(title + (f"\nUser's input category = {user_value}" if user_value is not None else ""))
    # plt.xlabel(series.name)
    # plt.ylabel("Frequency")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.show()
    rank = int(counts.rank(ascending=False, method="dense").get(user_value, np.nan)) if user_value in counts.index else None
    rate = float(counts[user_value] / counts.sum() * 100.0) if user_value in counts.index else None
    payload = {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "your_value": user_value,
        "rank": rank,
        "rate": rate
    }

    payload["description"] = _describe_categorical(series.name , payload)
    return payload

def _plot_founded_year(series: pd.Series, user_value, title: str = "Distribution of Founded Year"):
    # Parse to year safely
    years = series
    if years.empty:
        print("[Skip] Founded year: no valid year data.")
        return {"counts": {}, "your_value": None, "rank": None, "rate_pct": None}
    
    # Count by year, then fill gaps so the axis is continuous
    counts = years.value_counts().sort_index()
    full_index = pd.RangeIndex(start=int(years.min()), stop=int(years.max()) + 1)
    counts = counts.reindex(full_index, fill_value=0)

    # Normalize user_value
    try:
        uv = int(user_value) if user_value is not None and not pd.isna(user_value) else None
    except Exception:
        uv = None

    # Rank and share
    by_freq = counts.sort_values(ascending=False)
    rank = int(by_freq.rank(ascending=False, method="dense").get(uv, np.nan)) if uv in counts.index else None
    rate = float((counts.get(uv, 0) / counts.sum()) * 100.0) if uv in counts.index else None

    # Plot (bar timeline) + marker for the user's year

    # plt.figure()
    # plt.plot(counts.index.astype(int), counts.values)          # no color set (frontend can style)
    # if uv in counts.index:
        # plt.axvline(uv, linestyle="--")                       # mark user's founded year
    # plt.xlabel("Founded Year")
    # plt.ylabel("Number of Companies")
    # plt.title(title)
    # Show at most ~10 tick labels to avoid clutter
    # step = max(len(counts) // 10, 1)
    # plt.xticks(counts.index[::step], rotation=45)
    # plt.tight_layout()
    # plt.show()

    # JSON-friendly payload for your website
    payload = {
        "counts": {str(int(k)): int(v) for k, v in zip(counts.index, counts.values)},
        "your_value": uv,
        "rank": rank,
        "rate": rate
    }
    
    return payload


def _scatter_two_numeric(x: pd.Series, y: pd.Series, xv: float, yv: float, title: str):
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    m = xs.notna() & ys.notna()
    if m.sum() == 0 or pd.isna(xv) or pd.isna(yv):
        print(f"[Skip] {title}: not enough numeric pairs or user values are NaN.")
        return None
    # plt.figure()
    # plt.scatter(xs[m], ys[m], alpha=0.5)
    # plt.scatter([xv], [yv], s=120, marker="X")
    # plt.title(title + f"\nYour point = ({x.name}={xv:.3g}, {y.name}={yv:.3g})")
    # plt.xlabel(x.name)
    # plt.ylabel(y.name)
    # plt.show()
    # Return a compact payload for web plotting (only summary, not all points)
    return {
        "x_col": x.name,
        "y_col": y.name,
        "your_point": {"x": float(xv), "y": float(yv)},
        "n_points": int(m.sum()),
        "x_min": float(xs[m].min()),
        "x_max": float(xs[m].max()),
        "y_min": float(ys[m].min()),
        "y_max": float(ys[m].max()),
    }

# ---------- Main pipeline (updated) ----------
def show_company_location(
    user_input: Dict,
    df: pd.DataFrame,
    group_key: Optional[str] = "Industry_Group",
    numeric_pairs: Optional[List[Tuple[str, str]]] = None,
    id_column: Optional[str] = None,
    top_k = None,
    save_payload_path: Optional[str] = None,
    minmax: bool=False
):
    """
    Payload
    - founded_year -> founded year of given Industry_Group
    - Industry_Group -> calculate the value among all dataset
    - Numeric columns -> BOX PLOT with marker + returns stats for web rendering.
    - Categorical columns -> returns frequency table (+ optional local bar for inspection).
    - Returns a JSON-friendly payload; optionally saves it to disk.
    """
    # Determine cohort
    comp_df = _pick_group(df, user_input, group_key)
    cohort_desc = f"within {group_key}='{user_input.get(group_key)}'" if (group_key and group_key in user_input) else "overall industry"
    print(f"Comparing {cohort_desc}. Cohort size = {len(comp_df)}")
    
    # Optional backfill from df by id
    if id_column and id_column in df.columns and id_column in user_input:
        row = df.loc[df[id_column] == user_input[id_column]]
        if len(row) == 1:
            for k in df.columns:
                user_input.setdefault(k, row.iloc[0].get(k))
    
    payload = {
        "cohort": {"desc": cohort_desc, "size": int(len(comp_df))},
        "numeric": {},
        "categorical": {},
        "founded_year": {},
        "scatters": {}
    }
    
    # Summaries + plots
    for col, val in user_input.items():
        if col not in comp_df.columns:
            continue
        series = comp_df[col]
        if top_k is None and col == "founded":
            year_payload = _plot_founded_year(series, val)
            payload["founded_year"][col] = year_payload

        elif top_k is None and col == "Industry_Group":
            # emit counts for web; keep local bar for quick check
            cat_payload = _plot_categorical_location(df[col], val, f"{col} — {cohort_desc}")
            payload["categorical"][col] = cat_payload

        elif _is_numeric(series) and col != "founded":
            stats = _boxplot_with_marker_and_stats(series, val, f"{col} — {cohort_desc}", minmax=minmax)
            if stats is not None:
                payload["numeric"][col] = stats
        
    
    # Optional 2D numeric scatters (summary only)
    if numeric_pairs:
        for x_col, y_col in numeric_pairs:
            if x_col in comp_df.columns and y_col in comp_df.columns and (x_col in user_input) and (y_col in user_input):
                scat = _scatter_two_numeric(comp_df[x_col], comp_df[y_col], user_input[x_col], user_input[y_col],
                                            title=f"{x_col} vs {y_col} — {cohort_desc}")
                if scat:
                    payload["scatters"][f"{x_col}|{y_col}"] = scat
            else:
                print(f"[Skip] Scatter {x_col} vs {y_col}: missing columns or user values.")
    
    # Optional save
    if save_payload_path:
        with open(save_payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved payload to: {save_payload_path}")
    
    return payload

def topk_generator(user_input:dict, greedy_record=1, debug=0)-> pd.DataFrame:
    """
    A pipeline integrate
    - Find top 10 similar companies
    - Generate differences, analysis, suggestions according to information of 10 similar companies    
    """
    top10 = recommend_topk_for_input(
        user_input=user_input, k=10,
        constraints={"Industry_Group": user_input["Industry_Group"]},
        use_cols=None,  # or specify: ["Industry","country","current_employees","employee_growth","total_funding","founded"]
        show_cols=display_cols
    )

    # def ordered(row):
    #     return f"- Company {row['rank_in_list']}\n" + row["description"]

    # top10["description"] = top10.apply(ordered, axis=1)
    if debug:
        print(top10)
    return top10


def user_plot_pipeline(user_input, df=None):
    config.SERVER_STATUS = config.RagStatus.RETRIEVING_STATISTIC_DATA.value
    if df is None:
        df = pd.read_csv("startUpAgent_Backend/statistic_search/data/growjo_desc.csv")
    payload = show_company_location(
        user_input=user_input,
        df=df,
        group_key="Industry_Group",
        save_payload_path=None
    )
    
    topk = topk_generator(user_input, greedy_record=1)

    topk_payload = show_company_location(
        user_input=user_input,
        df=topk,
        group_key="Industry_Group",                          # or None for overall
        top_k=10,                                  # for categorical summaries
        save_payload_path=None,
        minmax=True
    )
    topk.drop(columns=['rank_in_list', 'sim_cosine'], inplace=True)
    desc = f'''
<top 10 company description>
{topk.to_json(orient="records", lines=True)}
</top 10 company description>

<compare to top 10>
{topk_payload['numeric'].get('current_employees', {}).get('description', {})}
{topk_payload['numeric'].get('total_funding', {}).get('description', {})}
</compare to top 10>

<all industry info>
{payload['categorical'].get('Industry_Group', {}).get('description', {})}
</all industry info>
'''
    plot_ = dict()
    plot_["Industry_Group"] = payload['categorical'].get('Industry_Group', {})
    plot_['current_employees'] = topk_payload['numeric'].get('current_employees', {})
    plot_['total_funding'] = topk_payload['numeric'].get('total_funding', {})
    return desc, plot_

if __name__ == "__main__":
    df = pd.read_csv("startUpAgent_Backend/statistic_search/data/growjo_desc.csv")
    test_input = {
        "Industry_Group": "Artificial Intelligence",
        "country": "US",
        "current_employees": 54,
        "total_funding": 25000000,
        "founded": 2014,
        "current_objectives": "hit $1.5M ARR in 12 months; land first enterprise logo",
        "strengths": "excellent multilingual support; lightweight API",
        "weaknesses": "limited admin features; no SOC2; weak sales motion",
        "desc": "The company develops an AI-powered customer support automation platform that specializes in multilingual markets across Asia-Pacific. Its lightweight API integrates seamlessly into existing CRMs and messaging apps, allowing SMEs to quickly deploy chatbots and virtual assistants in over 12 languages. Having raised $25M across Seed, Series A, and a recent Series B round, the company is now under pressure from investors to prove enterprise adoption. While it has seen strong traction with SMBs and mid-market clients in Taiwan and Southeast Asia, it has struggled to close larger enterprise contracts due to limited admin dashboards, lack of SOC2 compliance, and an inexperienced outbound sales team. The next 12 months are critical as the company plans to professionalize its go-to-market motion, hire enterprise sales talent, and pursue certifications to win the trust of Fortune 500 and regional conglomerates."
    }
    i = 0
    desc, plot_ = user_plot_pipeline(test_input, df)

    # write payload into json in ./advisor_output
    os.makedirs("./advisor_output", exist_ok=True)
    with open(f"./advisor_output/user_plot_ret.json", "w") as f:
        print(desc)
        json.dump(plot_, f, indent=2)
