import os
import pandas as pd
import numpy as np


def load_and_concat(paths):
    """
    Load multiple CSV files and concatenate them into a single DataFrame.
    Adds a column 'source_file' indicating the file name.

    Args:
        paths (list): List of file paths.

    Returns:
        pandas.DataFrame: Combined dataframe.
    """
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

        df = pd.read_csv(p)
        df['source_file'] = os.path.basename(p)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True, sort=False)


def find_first(cols, keywords):
    """
    Find the first column name that contains any of the given keywords.

    Args:
        cols (list): List of column names.
        keywords (list): List of keywords to search.

    Returns:
        str or None: Matching column name or None.
    """
    for k in keywords:
        for c in cols:
            if k in c.lower():
                return c
    return None


def build_composite_and_label(
    df,
    safety_keywords=["safety", "crime", "security", "police", "24x7"],
    infra_keywords=["infra", "infrastructure", "road", "area", "connect", "transport"],
    env_keywords=["env", "pollution", "air", "green", "water", "rain", "rainwater", "waste"],
    weights=(0.4, 0.35, 0.25)
):
    """
    Build composite score and recommendation label.

    Returns:
        df (DataFrame): Modified dataframe with scores and labels
        used_columns (tuple): (safety_col, infra_col, env_col)
    """

    colnames = [c for c in df.columns if isinstance(c, str)]

    safety_col = find_first(colnames, safety_keywords)
    infra_col = find_first(colnames, infra_keywords)
    env_col = find_first(colnames, env_keywords)

    # Fallback to first numeric columns if matching keywords are not found
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_candidates = [
        c for c in numeric_cols
        if c.lower() not in ("id", "index", "serial", "sr", "sno")
    ]

    if safety_col is None and len(numeric_candidates) >= 1:
        safety_col = numeric_candidates[0]

    if infra_col is None and len(numeric_candidates) >= 2:
        infra_col = numeric_candidates[1]

    if env_col is None and len(numeric_candidates) >= 3:
        env_col = numeric_candidates[2]

    # Convert selected columns to numeric
    for c in [safety_col, infra_col, env_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Min-max normalization
    def minmax(series):
        if series.isnull().all():
            return series.fillna(0)

        mn = series.min()
        mx = series.max()

        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return series.fillna(0)

        return (series - mn) / (mx - mn)

    df['_safety_n'] = minmax(df[safety_col]) if safety_col in df.columns else 0
    df['_infra_n'] = minmax(df[infra_col]) if infra_col in df.columns else 0
    df['_env_n'] = minmax(df[env_col]) if env_col in df.columns else 0

    
    env_name = (env_col or "").lower()
    if any(k in env_name for k in ["pollut", "air", "pm2", "pm10", "noise", "contamin"]):
        df['_env_n'] = 1 - df['_env_n']

    w_s, w_i, w_e = weights

    df['_composite_score'] = (
        w_s * df['_safety_n'].fillna(0) +
        w_i * df['_infra_n'].fillna(0) +
        w_e * df['_env_n'].fillna(0)
    )

    df['recommendation_label'] = pd.cut(
        df['_composite_score'],
        bins=[-0.01, 0.33, 0.67, 1.01],
        labels=[0, 1, 2]
    ).astype(int)

    return df, (safety_col, infra_col, env_col)
