"""Phase 1 retail sales analysis.

This script cleans a retail dataset, produces exploratory visualizations, and
writes a structured markdown report with actionable business insights.

Usage:
    python phase1_retail_insights.py --input path/to/data.csv --output-dir outputs
    python phase1_retail_insights.py --output-dir outputs --demo-rows 1200
"""

from __future__ import annotations

import argparse
import calendar
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")
sns.set_palette(["#0f766e", "#2563eb", "#ca8a04", "#16a34a", "#dc2626"])


ROLE_ORDER = [
    "sales",
    "quantity",
    "price",
    "discount",
    "date",
    "category",
    "product",
    "region",
    "customer_id",
    "order_id",
]

ROLE_ALIASES: dict[str, list[str]] = {
    "sales": [
        "revenue",
        "amount",
        "total_sales",
        "sales_amount",
        "gross_sales",
        "net_sales",
        "order_value",
        "total_revenue",
    ],
    "quantity": [
        "qty",
        "units",
        "unit_sold",
        "units_sold",
        "items",
        "count",
        "items_sold",
    ],
    "price": [
        "unit_price",
        "price_per_unit",
        "selling_price",
        "sale_price",
        "item_price",
        "cost_per_unit",
    ],
    "discount": [
        "discount_rate",
        "discount_percent",
        "markdown",
        "promo_discount",
        "promotion_discount",
    ],
    "date": [
        "order_date",
        "transaction_date",
        "invoice_date",
        "sales_date",
        "purchase_date",
        "timestamp",
        "datetime",
        "date_time",
    ],
    "category": [
        "product_category",
        "category_name",
        "department",
        "segment",
        "group",
        "product_type",
    ],
    "product": [
        "product_name",
        "item",
        "item_name",
        "sku",
        "product_code",
        "product_id",
        "product_title",
    ],
    "region": [
        "area",
        "state",
        "city",
        "market",
        "territory",
        "zone",
        "store_region",
        "location",
    ],
    "customer_id": [
        "customer",
        "customer_name",
        "client",
        "client_id",
        "buyer_id",
        "user_id",
        "cust_id",
    ],
    "order_id": [
        "order",
        "order_number",
        "transaction_id",
        "invoice_id",
        "receipt_id",
        "sale_id",
    ],
}

ROLE_LABELS: dict[str, str] = {
    "sales": "Sales",
    "quantity": "Quantity",
    "price": "Price",
    "discount": "Discount",
    "date": "Date",
    "category": "Category",
    "product": "Product",
    "region": "Region",
    "customer_id": "Customer ID",
    "order_id": "Order ID",
}

PRIMARY_COLOR = "#0f766e"
SECONDARY_COLOR = "#2563eb"
ACCENT_COLOR = "#ca8a04"
PLOT_DPI = 300


@dataclass
class AnalysisArtifacts:
    cleaned_csv: Path
    report_md: Path
    plot_paths: dict[str, Path]
    summary: dict[str, Any]


def emit_progress(progress_callback: Any | None, stage: str, progress: float, message: str, detail: str | None = None) -> None:
    if progress_callback is None:
        return
    payload: dict[str, Any] = {
        "stage": stage,
        "progress": float(progress),
        "message": message,
    }
    if detail:
        payload["detail"] = detail
    progress_callback(payload)


def project_root() -> Path:
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "column"


def make_unique_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in labels:
        base = label or "column"
        count = seen.get(base, 0)
        unique_labels.append(base if count == 0 else f"{base}_{count + 1}")
        seen[base] = count + 1
    return unique_labels


def human_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " ").title())


def escape_markdown(text: Any) -> str:
    value = format_value(text)
    return value.replace("|", "\\|").replace("\n", " ")


def format_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return "N/A"
    if isinstance(value, (pd.Timestamp, pd.Period)):
        if pd.isna(value):
            return "N/A"
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("\n", " ")
        return cleaned if cleaned else "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "N/A"
        if abs(value - round(float(value))) < 1e-9:
            return f"{int(round(float(value))):,}"
        return f"{float(value):,.2f}"
    return str(value)


def format_ratio(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows available._"

    preview = df.head(max_rows).copy()
    headers = [escape_markdown(col) for col in preview.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in preview.itertuples(index=False):
        cells = [escape_markdown(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string")
    text = text.str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
    text = text.str.replace(r"[^0-9.\-]", "", regex=True)
    text = text.replace({"": pd.NA, "-": pd.NA, ".": pd.NA, "-.": pd.NA})
    return pd.to_numeric(text, errors="coerce")


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError(
                "Excel input was provided, but pandas could not read it because the Excel engine is missing."
            ) from exc

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to load dataset from {path}")


def generate_synthetic_retail_data(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    catalog: dict[str, list[tuple[str, float, float]]] = {
        "Electronics": [
            ("Laptop", 800, 2200),
            ("Smartphone", 350, 1200),
            ("Headphones", 40, 350),
            ("Monitor", 120, 600),
        ],
        "Furniture": [
            ("Desk", 120, 750),
            ("Office Chair", 70, 420),
            ("Bookshelf", 60, 260),
            ("Lamp", 25, 140),
        ],
        "Grocery": [
            ("Coffee", 4, 18),
            ("Cereal", 3, 15),
            ("Snack Box", 5, 22),
            ("Tea", 3, 16),
        ],
        "Clothing": [
            ("T-Shirt", 10, 45),
            ("Jeans", 20, 110),
            ("Jacket", 35, 220),
            ("Sneakers", 45, 180),
        ],
        "Beauty": [
            ("Shampoo", 6, 28),
            ("Skin Cream", 12, 80),
            ("Perfume", 18, 160),
            ("Face Wash", 5, 24),
        ],
    }

    categories = list(catalog.keys())
    regions = ["North", "South", "East", "West", "Central"]
    customer_pool = [f"C{index:04d}" for index in range(1, 601)]

    product_name: list[str] = []
    category_name: list[str] = []
    base_price: list[float] = []
    price_ceiling: list[float] = []
    for category, products in catalog.items():
        for product, low, high in products:
            product_name.append(product)
            category_name.append(category)
            base_price.append(low)
            price_ceiling.append(high)

    product_catalog = pd.DataFrame(
        {
            "product": product_name,
            "category": category_name,
            "base_price": base_price,
            "price_ceiling": price_ceiling,
        }
    )

    chosen_products = product_catalog.sample(n=rows, replace=True, random_state=seed).reset_index(drop=True)
    date_start = pd.Timestamp("2024-01-01")
    date_end = pd.Timestamp("2025-12-31")
    total_days = (date_end - date_start).days
    date_offsets = rng.integers(0, total_days + 1, size=rows)
    order_dates = date_start + pd.to_timedelta(date_offsets, unit="D")

    quantity = rng.poisson(lam=2.5, size=rows) + 1
    quantity = np.clip(quantity, 1, 12)

    unit_price = np.array([
        rng.uniform(low, high) for low, high in zip(chosen_products["base_price"], chosen_products["price_ceiling"])
    ])
    discount_rate = rng.uniform(0.0, 0.30, size=rows)
    regional_bias = np.select(
        [chosen_products["category"].eq("Electronics"), chosen_products["category"].eq("Furniture")],
        [1.10, 1.03],
        default=1.0,
    )
    sales = quantity * unit_price * (1 - discount_rate) * regional_bias
    sales = np.maximum(sales, 1.0)

    region = rng.choice(regions, size=rows, p=[0.22, 0.21, 0.2, 0.19, 0.18])
    customer_id = rng.choice(customer_pool, size=rows, replace=True)
    order_id = [f"O{index:06d}" for index in range(1, rows + 1)]

    df = pd.DataFrame(
        {
            "order_id": order_id,
            "customer_id": customer_id,
            "date": order_dates,
            "product": chosen_products["product"].to_numpy(),
            "category": chosen_products["category"].to_numpy(),
            "region": region,
            "quantity": quantity,
            "price": unit_price,
            "discount": discount_rate,
            "sales": sales,
        }
    )

    df["price"] = df["price"].astype(object)
    df["sales"] = df["sales"].astype(object)

    missing_quantity = rng.choice(df.index, size=max(1, rows // 30), replace=False)
    missing_price = rng.choice(df.index, size=max(1, rows // 35), replace=False)
    missing_category = rng.choice(df.index, size=max(1, rows // 40), replace=False)
    missing_region = rng.choice(df.index, size=max(1, rows // 45), replace=False)
    missing_date = rng.choice(df.index, size=max(1, rows // 50), replace=False)
    missing_sales = rng.choice(df.index, size=max(1, rows // 55), replace=False)

    df.loc[missing_quantity, "quantity"] = np.nan
    df.loc[missing_price, "price"] = np.nan
    df.loc[missing_category, "category"] = np.nan
    df.loc[missing_region, "region"] = np.nan
    df.loc[missing_date, "date"] = pd.NaT
    df.loc[missing_sales, "sales"] = np.nan

    noisy_currency = rng.choice(df.index, size=max(1, rows // 25), replace=False)
    df.loc[noisy_currency, "price"] = df.loc[noisy_currency, "price"].map(
        lambda value: f"${value:,.2f}" if pd.notna(value) else value
    )

    noisy_sales = rng.choice(df.index, size=max(1, rows // 25), replace=False)
    df.loc[noisy_sales, "sales"] = df.loc[noisy_sales, "sales"].map(
        lambda value: f"${value:,.2f}" if pd.notna(value) else value
    )

    messy_region = rng.choice(df.index, size=max(1, rows // 20), replace=False)
    df.loc[messy_region, "region"] = df.loc[messy_region, "region"].map(
        lambda value: f"  {value.upper()}  " if pd.notna(value) else value
    )

    duplicate_rows = df.sample(frac=0.02, random_state=seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


def best_match(columns: list[str], candidates: list[str]) -> str | None:
    normalized_candidates = [normalize_name(candidate) for candidate in candidates]
    normalized_columns = {column: normalize_name(column) for column in columns}

    for candidate in normalized_candidates:
        for column, normalized_column in normalized_columns.items():
            if normalized_column == candidate:
                return column

    best_column: str | None = None
    best_score = 0.0
    for column, normalized_column in normalized_columns.items():
        candidate_scores: list[float] = []
        for candidate in normalized_candidates:
            if candidate in normalized_column or normalized_column in candidate:
                candidate_scores.append(0.97)
            else:
                candidate_scores.append(SequenceMatcher(None, normalized_column, candidate).ratio())
        score = max(candidate_scores) if candidate_scores else 0.0
        if score > best_score:
            best_score = score
            best_column = column
    return best_column if best_score >= 0.72 else None


def infer_date_column(df: pd.DataFrame, excluded: set[str]) -> str | None:
    best_column: str | None = None
    best_score = 0.0
    for column in df.columns:
        if column in excluded:
            continue
        series = df[column].dropna()
        if series.empty or pd.api.types.is_numeric_dtype(series):
            continue
        sample = series.astype("string").head(200)
        parsed = pd.to_datetime(sample, errors="coerce")
        score = float(parsed.notna().mean())
        normalized_column = normalize_name(column)
        if any(token in normalized_column for token in ("date", "time", "timestamp", "datetime")):
            score += 0.15
        if score > best_score:
            best_score = score
            best_column = column
    return best_column if best_score >= 0.55 else None


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    working = df.copy()
    working.columns = make_unique_labels([normalize_name(column) for column in working.columns])

    selected_roles: dict[str, str | None] = {}
    available_columns = list(working.columns)
    rename_map: dict[str, str] = {}

    for role in ROLE_ORDER:
        match = best_match(available_columns, [role, *ROLE_ALIASES.get(role, [])])
        selected_roles[role] = match
        if match is None:
            continue
        if match != role and role not in available_columns:
            rename_map[match] = role
            available_columns.remove(match)
        elif match == role:
            available_columns.remove(match)

    if rename_map:
        working = working.rename(columns=rename_map)
        for role, source in list(selected_roles.items()):
            if source in rename_map:
                selected_roles[role] = rename_map[source]

    excluded = {column for column in selected_roles.values() if column is not None}
    if selected_roles.get("date") is None:
        inferred_date = infer_date_column(working, excluded=excluded)
        if inferred_date is not None:
            selected_roles["date"] = inferred_date

    return working, selected_roles


def prepare_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_rows = int(raw_df.shape[0])
    before_columns = int(raw_df.shape[1])
    before_missing = int(raw_df.isna().sum().sum())
    before_duplicates = int(raw_df.duplicated().sum())

    df, selected_roles = canonicalize_columns(raw_df)

    string_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for column in string_columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    for column in [col for col in ["category", "product", "region"] if col in df.columns]:
        df[column] = df[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.title()

    numeric_roles = ["sales", "quantity", "price", "discount"]
    for role in numeric_roles:
        if role in df.columns:
            df[role] = coerce_numeric_series(df[role])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "sales" not in df.columns and {"quantity", "price"}.issubset(df.columns):
        df["sales"] = df["quantity"] * df["price"]
        selected_roles["sales"] = "sales"
    elif "sales" in df.columns and {"quantity", "price"}.issubset(df.columns):
        sales_mask = df["sales"].isna() & df["quantity"].notna() & df["price"].notna()
        df.loc[sales_mask, "sales"] = df.loc[sales_mask, "quantity"] * df.loc[sales_mask, "price"]

    if "price" not in df.columns and {"sales", "quantity"}.issubset(df.columns):
        valid_mask = df["price"].isna() if "price" in df.columns else pd.Series(False, index=df.index)
        if "price" not in df.columns:
            df["price"] = pd.NA
        valid_mask = df["sales"].notna() & df["quantity"].notna() & (df["quantity"] != 0)
        df.loc[valid_mask, "price"] = df.loc[valid_mask, "sales"] / df.loc[valid_mask, "quantity"]
        selected_roles["price"] = selected_roles.get("price") or "price"

    if "quantity" not in df.columns and {"sales", "price"}.issubset(df.columns):
        if "quantity" not in df.columns:
            df["quantity"] = pd.NA
        valid_mask = df["sales"].notna() & df["price"].notna() & (df["price"] != 0)
        df.loc[valid_mask, "quantity"] = df.loc[valid_mask, "sales"] / df.loc[valid_mask, "price"]
        selected_roles["quantity"] = selected_roles.get("quantity") or "quantity"

    for column in ["quantity", "price", "discount"]:
        if column in df.columns:
            df.loc[df[column] < 0, column] = pd.NA

    df = df.dropna(how="all")
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns and column not in datetime_columns]

    numeric_imputations: dict[str, float] = {}
    datetime_imputations: dict[str, str] = {}
    categorical_imputations: dict[str, str] = {}

    for column in numeric_columns:
        missing_count = int(df[column].isna().sum())
        if missing_count > 0:
            fill_value = df[column].median()
            if pd.isna(fill_value):
                fill_value = 0.0
            df[column] = df[column].fillna(fill_value)
            numeric_imputations[column] = float(fill_value)

    for column in datetime_columns:
        missing_count = int(df[column].isna().sum())
        if missing_count > 0:
            fill_value = df[column].dropna().median()
            if pd.isna(fill_value):
                fill_value = pd.Timestamp("today").normalize()
            df[column] = df[column].fillna(fill_value)
            datetime_imputations[column] = str(fill_value)

    for column in categorical_columns:
        missing_count = int(df[column].isna().sum())
        if missing_count > 0:
            mode_values = df[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)
            categorical_imputations[column] = str(fill_value)

    after_rows = int(df.shape[0])
    after_columns = int(df.shape[1])
    after_missing = int(df.isna().sum().sum())

    notes: list[str] = ["Standardized column names to snake_case.", "Trimmed whitespace from text fields.", "Removed duplicate rows."]
    if "sales" in df.columns:
        notes.append("Ensured a canonical sales field was available for revenue analysis.")
    if selected_roles.get("date") is not None:
        notes.append(f"Detected date column: {selected_roles['date']}.")
    else:
        notes.append("No reliable date column was detected; time-based analysis will be skipped if a date field is unavailable.")
    if "customer_id" not in df.columns:
        notes.append("No explicit customer identifier was detected; customer behavior will be summarized at the transaction level.")

    summary = {
        "rows_before": before_rows,
        "columns_before": before_columns,
        "missing_before": before_missing,
        "duplicates_before": before_duplicates,
        "rows_after": after_rows,
        "columns_after": after_columns,
        "missing_after": after_missing,
        "duplicates_removed": duplicates_removed,
        "selected_roles": selected_roles,
        "numeric_imputations": numeric_imputations,
        "datetime_imputations": datetime_imputations,
        "categorical_imputations": categorical_imputations,
        "notes": notes,
    }
    return df, summary


def top_missing_columns(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(limit)
    if missing.empty:
        return pd.DataFrame(columns=["column", "missing_values"])
    return missing.rename_axis("column").reset_index(name="missing_values")


def select_numeric_metrics(df: pd.DataFrame) -> list[str]:
    metrics = [column for column in ["sales", "quantity", "price", "discount"] if column in df.columns]
    if not metrics:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        metrics = [column for column in numeric_columns if not column.endswith("_id")]
    return metrics


def plot_numeric_distributions(df: pd.DataFrame, output_path: Path) -> Path | None:
    numeric_metrics = [column for column in ["sales", "quantity", "price"] if column in df.columns]
    if not numeric_metrics:
        return None

    fig, axes = plt.subplots(1, len(numeric_metrics), figsize=(6 * len(numeric_metrics), 5))
    if len(numeric_metrics) == 1:
        axes = [axes]

    for axis, column in zip(axes, numeric_metrics):
        values = df[column].dropna()
        bins = min(30, max(10, int(np.sqrt(len(values))) if len(values) > 0 else 10))
        sns.histplot(
            values,
            bins=bins,
            kde=values.nunique() > 10,
            color=PRIMARY_COLOR,
            ax=axis,
        )
        axis.set_title(f"Distribution of {human_label(column)}")
        axis.set_xlabel(human_label(column))
        axis.set_ylabel("Count")

    fig.suptitle("Distributions of Key Numerical Variables", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_category_vs_sales(df: pd.DataFrame, output_path: Path) -> Path | None:
    group_column = df.columns.intersection(["category", "product"])
    if group_column.empty or "sales" not in df.columns:
        return None

    if "category" in df.columns:
        chosen_group = "category"
        title = "Product Category vs Sales"
        subtitle = "Top categories by total sales"
    else:
        chosen_group = "product"
        title = "Product Performance vs Sales"
        subtitle = "Top products by total sales"

    aggregated = (
        df.groupby(chosen_group, dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index()
    )
    if aggregated.empty:
        return None

    aggregated[chosen_group] = aggregated[chosen_group].astype(str)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.55 * len(aggregated) + 2)))
    sns.barplot(data=aggregated, y=chosen_group, x="sales", color=PRIMARY_COLOR, ax=ax)
    ax.set_xlabel("Sales")
    ax.set_ylabel(human_label(chosen_group))
    ax.set_title(f"{title}\n{subtitle}", fontsize=18, fontweight="bold")
    for container in ax.containers:
        ax.bar_label(container, labels=[format_currency(value) for value in aggregated["sales"]], padding=3, fontsize=10)
        break
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_region_vs_revenue(df: pd.DataFrame, output_path: Path) -> Path | None:
    if "region" not in df.columns or "sales" not in df.columns:
        return None

    aggregated = df.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index()
    if aggregated.empty:
        return None

    aggregated["region"] = aggregated["region"].astype(str)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.55 * len(aggregated) + 2)))
    sns.barplot(data=aggregated, y="region", x="sales", color=SECONDARY_COLOR, ax=ax)
    ax.set_title("Region vs Revenue", fontsize=18, fontweight="bold")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Region")
    for container in ax.containers:
        ax.bar_label(container, labels=[format_currency(value) for value in aggregated["sales"]], padding=3, fontsize=10)
        break
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def choose_time_frequency(date_series: pd.Series) -> str:
    valid_dates = date_series.dropna()
    if valid_dates.empty:
        return "M"
    span_days = int((valid_dates.max() - valid_dates.min()).days)
    unique_dates = int(valid_dates.dt.normalize().nunique())
    if span_days > 730 or unique_dates > 365:
        return "M"
    if span_days > 120 or unique_dates > 90:
        return "W"
    return "D"


def plot_time_trend(df: pd.DataFrame, output_path: Path) -> tuple[Path | None, dict[str, Any]]:
    if "date" not in df.columns or "sales" not in df.columns:
        return None, {}

    time_series = df[["date", "sales"]].dropna().sort_values("date")
    if time_series.empty:
        return None, {}

    frequency = choose_time_frequency(time_series["date"])
    trend = time_series.set_index("date")["sales"].resample(frequency).sum()
    if trend.empty:
        return None, {}

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(trend.index, trend.values, color=PRIMARY_COLOR, marker="o", linewidth=2)
    ax.set_title(f"Sales Trend Over Time ({frequency})", fontsize=18, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    month_groups = time_series.assign(month=time_series["date"].dt.month).groupby("month")["sales"].sum().sort_index()
    peak_period = trend.idxmax()
    low_period = trend.idxmin()
    peak_month = month_groups.idxmax() if not month_groups.empty else None
    low_month = month_groups.idxmin() if not month_groups.empty else None
    analysis = {
        "frequency": frequency,
        "trend": trend,
        "peak_period": peak_period,
        "low_period": low_period,
        "peak_month": peak_month,
        "low_month": low_month,
        "month_groups": month_groups,
    }
    return output_path, analysis


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> Path | None:
    candidate_columns = [column for column in ["sales", "quantity", "price", "discount"] if column in df.columns]
    if len(candidate_columns) < 2:
        return None

    corr = df[candidate_columns].corr(numeric_only=True)
    if corr.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="crest", center=0, linewidths=0.5, ax=ax, fmt=".2f")
    ax.set_title("Correlation Heatmap", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def summarize_top_entities(df: pd.DataFrame, entity_column: str, sales_column: str = "sales", quantity_column: str | None = "quantity", order_id_column: str | None = "order_id") -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {sales_column: (sales_column, "sum")}
    if quantity_column and quantity_column in df.columns:
        aggregations[quantity_column] = (quantity_column, "sum")
    if order_id_column and order_id_column in df.columns:
        aggregations["orders"] = (order_id_column, "nunique")
    else:
        aggregations["orders"] = (sales_column, "size")

    grouped = df.groupby(entity_column, dropna=False).agg(**aggregations).reset_index()
    grouped = grouped.sort_values(by=sales_column, ascending=False)
    return grouped.head(10)


def build_customer_behavior(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if "customer_id" in df.columns:
        customer_summary = (
            df.groupby("customer_id", dropna=False)
            .agg(
                orders=("sales", "size"),
                revenue=("sales", "sum"),
                average_order_value=("sales", "mean"),
            )
            .reset_index()
            .sort_values(by="revenue", ascending=False)
        )
        repeat_rate = float((customer_summary["orders"] > 1).mean()) if not customer_summary.empty else np.nan
        avg_orders_per_customer = float(customer_summary["orders"].mean()) if not customer_summary.empty else np.nan
        avg_spend_per_customer = float(customer_summary["revenue"].mean()) if not customer_summary.empty else np.nan
        top_customer_share = float(customer_summary.head(5)["revenue"].sum() / df["sales"].sum()) if df["sales"].sum() else np.nan
        metrics = {
            "active_customers": int(customer_summary.shape[0]),
            "repeat_rate": repeat_rate,
            "avg_orders_per_customer": avg_orders_per_customer,
            "avg_spend_per_customer": avg_spend_per_customer,
            "top_customer_share": top_customer_share,
            "analysis_type": "customer_level",
        }
        return metrics, customer_summary.head(10)

    transaction_summary = pd.DataFrame(
        [
            {
                "metric": "Average transaction value",
                "value": float(df["sales"].mean()) if "sales" in df.columns else np.nan,
            },
            {
                "metric": "Median transaction value",
                "value": float(df["sales"].median()) if "sales" in df.columns else np.nan,
            },
            {
                "metric": "Average quantity per transaction",
                "value": float(df["quantity"].mean()) if "quantity" in df.columns else np.nan,
            },
            {
                "metric": "Transactions above median value",
                "value": float((df["sales"] > df["sales"].median()).mean()) if "sales" in df.columns else np.nan,
            },
        ]
    )
    metrics = {
        "active_customers": np.nan,
        "repeat_rate": np.nan,
        "avg_orders_per_customer": np.nan,
        "avg_spend_per_customer": np.nan,
        "top_customer_share": np.nan,
        "analysis_type": "transaction_level",
    }
    return metrics, transaction_summary


def build_revenue_driver_metrics(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    correlation_rows: list[dict[str, Any]] = []
    for column in ["quantity", "price", "discount"]:
        if column in df.columns and column != "sales":
            pair = df[["sales", column]].dropna()
            if len(pair) > 1 and pair[column].nunique() > 1 and pair["sales"].nunique() > 1:
                correlation_rows.append(
                    {
                        "factor": human_label(column),
                        "correlation_with_sales": float(pair["sales"].corr(pair[column])),
                    }
                )

    correlation_table = pd.DataFrame(correlation_rows)
    if not correlation_table.empty:
        correlation_table = correlation_table.sort_values(by="correlation_with_sales", key=lambda values: values.abs(), ascending=False)

    category_table = pd.DataFrame()
    if "category" in df.columns:
        category_table = (
            df.groupby("category", dropna=False)["sales"].sum().sort_values(ascending=False).head(5).reset_index()
        )
        category_table["share_of_sales"] = category_table["sales"] / df["sales"].sum() if df["sales"].sum() else np.nan

    region_table = pd.DataFrame()
    if "region" in df.columns:
        region_table = df.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False).head(5).reset_index()
        region_table["share_of_sales"] = region_table["sales"] / df["sales"].sum() if df["sales"].sum() else np.nan

    metrics = {
        "category_top5_share": float(category_table["share_of_sales"].sum()) if not category_table.empty else np.nan,
        "region_top5_share": float(region_table["share_of_sales"].sum()) if not region_table.empty else np.nan,
        "correlation_table": correlation_table,
    }
    return metrics, category_table, region_table


def build_seasonality_metrics(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if "date" not in df.columns:
        return {"available": False}, pd.DataFrame()

    monthly_sales = (
        df.dropna(subset=["date"])
        .assign(month_period=lambda frame: frame["date"].dt.to_period("M"))
        .groupby("month_period")["sales"]
        .sum()
        .sort_index()
        .reset_index()
    )
    if monthly_sales.empty:
        return {"available": False}, monthly_sales

    month_pattern = (
        df.dropna(subset=["date"])
        .assign(month=lambda frame: frame["date"].dt.month)
        .groupby("month")["sales"]
        .sum()
        .sort_index()
        .reset_index()
    )
    peak_month_index = int(month_pattern.loc[month_pattern["sales"].idxmax(), "month"]) if not month_pattern.empty else None
    low_month_index = int(month_pattern.loc[month_pattern["sales"].idxmin(), "month"]) if not month_pattern.empty else None

    metrics = {
        "available": True,
        "monthly_sales": monthly_sales,
        "month_pattern": month_pattern,
        "peak_month": peak_month_index,
        "low_month": low_month_index,
        "first_period": str(monthly_sales.iloc[0]["month_period"]),
        "last_period": str(monthly_sales.iloc[-1]["month_period"]),
        "first_period_sales": float(monthly_sales.iloc[0]["sales"]),
        "last_period_sales": float(monthly_sales.iloc[-1]["sales"]),
    }
    return metrics, monthly_sales


def generate_insight_summary(
    df: pd.DataFrame,
    summary: dict[str, Any],
    product_table: pd.DataFrame,
    region_table: pd.DataFrame,
    seasonality: dict[str, Any],
    customer_metrics: dict[str, Any],
    customer_table: pd.DataFrame,
    revenue_metrics: dict[str, Any],
    selected_roles: dict[str, str | None],
) -> dict[str, Any]:
    top_product_row = product_table.iloc[0] if not product_table.empty else None
    top_region_row = region_table.iloc[0] if not region_table.empty else None

    insights: dict[str, Any] = {
        "top_product_row": top_product_row,
        "top_region_row": top_region_row,
        "product_table": product_table,
        "region_table": region_table,
        "customer_table": customer_table,
        "revenue_metrics": revenue_metrics,
        "seasonality": seasonality,
        "customer_metrics": customer_metrics,
        "selected_roles": selected_roles,
    }

    total_sales = float(df["sales"].sum()) if "sales" in df.columns else np.nan
    insights["total_sales"] = total_sales
    insights["top_category_share"] = revenue_metrics.get("category_top5_share")
    insights["top_region_share"] = revenue_metrics.get("region_top5_share")
    insights["duplicate_rows_removed"] = summary.get("duplicates_removed", 0)
    return insights


def build_report(
    source_label: str,
    output_dir: Path,
    cleaned_path: Path,
    plot_paths: dict[str, Path | None],
    df: pd.DataFrame,
    summary: dict[str, Any],
    insights: dict[str, Any],
    missing_table: pd.DataFrame,
) -> str:
    def relative_to_output(path: Path) -> str:
        try:
            return path.relative_to(output_dir).as_posix()
        except ValueError:
            return path.as_posix()

    selected_roles = insights["selected_roles"]
    role_rows = []
    for role in ROLE_ORDER:
        role_rows.append({"role": human_label(role), "detected_column": selected_roles.get(role) or "Not detected"})
    role_table = pd.DataFrame(role_rows)

    data_quality_table = pd.DataFrame(
        [
            {"metric": "Source", "value": source_label},
            {"metric": "Rows before cleaning", "value": summary["rows_before"]},
            {"metric": "Rows after cleaning", "value": summary["rows_after"]},
            {"metric": "Columns before cleaning", "value": summary["columns_before"]},
            {"metric": "Columns after cleaning", "value": summary["columns_after"]},
            {"metric": "Missing values before", "value": summary["missing_before"]},
            {"metric": "Missing values after", "value": summary["missing_after"]},
            {"metric": "Duplicate rows removed", "value": summary["duplicates_removed"]},
            {"metric": "Cleaned CSV", "value": relative_to_output(cleaned_path)},
        ]
    )

    sections: list[str] = []
    sections.append("# Retail Store Sales Insights and Prediction Model - Phase 1 Report")
    sections.append("")
    sections.append("## Dataset Overview")
    sections.append("This report summarizes the cleaned dataset, exploratory analysis results, and the main business patterns detected in the data.")
    sections.append("")
    sections.append(dataframe_to_markdown(data_quality_table, max_rows=20))
    sections.append("")
    sections.append("### Detected Fields")
    sections.append(dataframe_to_markdown(role_table, max_rows=20))
    sections.append("")
    sections.append("### Data Cleaning Actions")
    for note in summary.get("notes", []):
        sections.append(f"- {note}")
    if summary.get("numeric_imputations"):
        sections.append(f"- Numeric values were imputed for {len(summary['numeric_imputations'])} field(s) using median values.")
    if summary.get("categorical_imputations"):
        sections.append(f"- Categorical values were imputed for {len(summary['categorical_imputations'])} field(s) using mode values.")
    if summary.get("datetime_imputations"):
        sections.append(f"- Date values were imputed for {len(summary['datetime_imputations'])} field(s) using median dates.")
    sections.append("")

    if not missing_table.empty:
        sections.append("### Missing Values Before Cleaning")
        sections.append(dataframe_to_markdown(missing_table, max_rows=10))
        sections.append("")

    sections.append("## Exploratory Data Analysis")
    if plot_paths.get("numeric_distributions") is not None:
        sections.append("### Distributions of Sales, Quantity, and Price")
        sections.append(f"![Numeric distributions]({relative_to_output(plot_paths['numeric_distributions'])})")
        sections.append("")
    if plot_paths.get("category_vs_sales") is not None:
        sections.append("### Category vs Sales")
        sections.append(f"![Category vs sales]({relative_to_output(plot_paths['category_vs_sales'])})")
        sections.append("")
    if plot_paths.get("region_vs_revenue") is not None:
        sections.append("### Region vs Revenue")
        sections.append(f"![Region vs revenue]({relative_to_output(plot_paths['region_vs_revenue'])})")
        sections.append("")
    if plot_paths.get("time_trend") is not None:
        sections.append("### Time vs Sales Trend")
        sections.append(f"![Time trend]({relative_to_output(plot_paths['time_trend'])})")
        sections.append("")
    if plot_paths.get("correlation_heatmap") is not None:
        sections.append("### Numeric Correlation Heatmap")
        sections.append(f"![Correlation heatmap]({relative_to_output(plot_paths['correlation_heatmap'])})")
        sections.append("")

    sections.append("## Business Insights")
    total_sales = insights.get("total_sales", np.nan)

    product_title = "Top-Performing Products"
    if "product" not in df.columns and "category" in df.columns:
        product_title = "Top-Performing Categories Used as a Product Proxy"
    elif "product" not in df.columns and "category" not in df.columns:
        product_title = "Top-Performing Sales Segments"

    sections.append(f"### {product_title}")
    if not insights["product_table"].empty:
        sections.append(dataframe_to_markdown(insights["product_table"], max_rows=10))
        best_product = insights["top_product_row"]
        if best_product is not None:
            best_product_label = best_product.iloc[0]
            sections.append(
                f"- Highest revenue contribution came from **{best_product_label}** with {format_currency(float(best_product['sales']))} in sales."
            )
            if len(insights["product_table"]) >= 5 and not pd.isna(total_sales) and total_sales != 0:
                top_share = float(insights["product_table"].head(5)["sales"].sum() / total_sales)
                sections.append(f"- The top 5 entries in this table account for {format_ratio(top_share)} of total sales.")
    else:
        sections.append("_No product or category field was detected, so product-level insights could not be generated._")
    sections.append("")

    sections.append("### Best-Performing Regions")
    if not insights["region_table"].empty:
        sections.append(dataframe_to_markdown(insights["region_table"], max_rows=10))
        top_region = insights["top_region_row"]
        if top_region is not None:
            top_region_label = top_region.iloc[0]
            sections.append(
                f"- The strongest region was **{top_region_label}** with {format_currency(float(top_region['sales']))} in sales."
            )
            if len(insights["region_table"]) >= 5 and not pd.isna(total_sales) and total_sales != 0:
                region_share = float(insights["region_table"].head(5)["sales"].sum() / total_sales)
                sections.append(f"- The top 5 regions contribute {format_ratio(region_share)} of total sales.")
    else:
        sections.append("_No region field was detected._")
    sections.append("")

    sections.append("### Seasonal Trends")
    seasonality_metrics = insights["seasonality"]
    if seasonality_metrics.get("available"):
        sections.append(
            f"- Sales peaked in **{calendar.month_name[seasonality_metrics['peak_month']]}** and were lowest in **{calendar.month_name[seasonality_metrics['low_month']]}** when looking at the month-of-year pattern."
        )
        sections.append(
            f"- The time series runs from **{seasonality_metrics['first_period']}** to **{seasonality_metrics['last_period']}**, showing how sales evolved across the available period."
        )
        sections.append(
            f"- Sales moved from {format_currency(seasonality_metrics['first_period_sales'])} in the first observed period to {format_currency(seasonality_metrics['last_period_sales'])} in the last observed period."
        )
    else:
        sections.append("_A reliable date field was not available, so seasonality could not be measured directly._")
    sections.append("")

    sections.append("### Customer Purchasing Behavior")
    customer_metrics = insights["customer_metrics"]
    if customer_metrics.get("analysis_type") == "customer_level":
        sections.append(
            f"- The dataset contains {format_value(customer_metrics['active_customers'])} active customers.")
        sections.append(
            f"- Repeat customer rate is {format_ratio(customer_metrics['repeat_rate'])}, which is a useful indicator of loyalty and retention strength."
        )
        sections.append(
            f"- Average orders per customer are {format_value(customer_metrics['avg_orders_per_customer'])}, and average spend per customer is {format_currency(customer_metrics['avg_spend_per_customer'])}."
        )
        sections.append(
            f"- The top 5 customers account for {format_ratio(customer_metrics['top_customer_share'])} of total sales."
        )
        if not insights["customer_table"].empty:
            sections.append(dataframe_to_markdown(insights["customer_table"], max_rows=10))
    else:
        sections.append("- No explicit customer identifier was available, so customer-level analysis was reduced to transaction-level behavior.")
        if not insights["customer_table"].empty:
            sections.append(dataframe_to_markdown(insights["customer_table"], max_rows=10))
    sections.append("")

    sections.append("### Primary Factors Influencing Revenue")
    revenue_metrics = insights["revenue_metrics"]
    if not revenue_metrics["correlation_table"].empty:
        sections.append(dataframe_to_markdown(revenue_metrics["correlation_table"], max_rows=10))
        best_factor = revenue_metrics["correlation_table"].iloc[0]
        sections.append(
            f"- The strongest measured relationship with sales was **{best_factor['factor']}** at {best_factor['correlation_with_sales']:.2f}."
        )
    else:
        sections.append("_Not enough numeric drivers were available to estimate correlations with sales._")
    if not pd.isna(revenue_metrics.get("category_top5_share")):
        sections.append(
            f"- The top 5 categories contribute {format_ratio(revenue_metrics['category_top5_share'])} of total sales, indicating category concentration."
        )
    if not pd.isna(revenue_metrics.get("region_top5_share")):
        sections.append(
            f"- The top 5 regions contribute {format_ratio(revenue_metrics['region_top5_share'])} of total sales, indicating geographic concentration."
        )
    sections.append("")

    sections.append("## Suggested Actions")
    suggestion_bullets = [
        "Use the top-performing product and category groups to guide inventory planning and replenishment.",
        "Prioritize marketing spend in the strongest regions while testing localized promotions in underperforming areas.",
        "Align promotions and staffing with the peak seasonal periods identified in the time trend analysis.",
        "If repeat customer rate is low, consider loyalty incentives, targeted offers, or post-purchase follow-ups.",
    ]
    for bullet in suggestion_bullets:
        sections.append(f"- {bullet}")
    sections.append("")
    sections.append("## Deliverables")
    sections.append(f"- Cleaned dataset: {relative_to_output(cleaned_path)}")
    sections.append(f"- Plots directory: {relative_to_output(output_dir.joinpath('plots'))}")
    sections.append("- This markdown report can be exported to PDF or Word using your preferred Markdown conversion tool.")

    return "\n".join(sections)


def run_phase1_analysis(
    input_path: Path | None,
    output_dir: Path,
    demo_rows: int,
    seed: int,
    progress_callback: Any | None = None,
) -> AnalysisArtifacts:
    emit_progress(progress_callback, "setup", 5, "Preparing workspace folders and analysis outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if input_path is not None and input_path.exists():
        emit_progress(progress_callback, "load", 12, f"Loading dataset from {input_path.name}...")
        source_df = load_dataset(input_path)
        source_label = str(input_path)
        demo_path: Path | None = None
    else:
        emit_progress(progress_callback, "generate", 12, "Generating synthetic retail demo data...")
        source_df = generate_synthetic_retail_data(demo_rows, seed=seed)
        source_label = f"Synthetic demo data ({demo_rows} rows, seed={seed})"
        demo_path = output_dir / "demo_retail_data.csv"
        source_df.to_csv(demo_path, index=False)

    emit_progress(progress_callback, "cleaning", 25, "Standardizing columns, cleaning values, and resolving missing data...")
    cleaned_df, summary = prepare_dataframe(source_df)
    missing_table = top_missing_columns(source_df, limit=10)

    cleaned_path = output_dir / "cleaned_retail_data.csv"
    if "date" in cleaned_df.columns:
        cleaned_df.to_csv(cleaned_path, index=False, date_format="%Y-%m-%d")
    else:
        cleaned_df.to_csv(cleaned_path, index=False)

    emit_progress(progress_callback, "saved_clean_data", 40, "Cleaned dataset saved. Generating EDA charts...")

    plot_paths: dict[str, Path | None] = {
        "numeric_distributions": plot_numeric_distributions(cleaned_df, plots_dir / "01_numeric_distributions.png"),
        "category_vs_sales": plot_category_vs_sales(cleaned_df, plots_dir / "02_category_vs_sales.png"),
        "region_vs_revenue": plot_region_vs_revenue(cleaned_df, plots_dir / "03_region_vs_revenue.png"),
    }
    emit_progress(progress_callback, "eda_numeric", 50, "Distribution charts created.")
    time_trend_path, seasonality_plot = plot_time_trend(cleaned_df, plots_dir / "04_time_sales_trend.png")
    plot_paths["time_trend"] = time_trend_path
    emit_progress(progress_callback, "eda_time", 62, "Time trend chart created.")
    plot_paths["correlation_heatmap"] = plot_correlation_heatmap(cleaned_df, plots_dir / "05_correlation_heatmap.png")
    emit_progress(progress_callback, "eda_heatmap", 70, "Correlation heatmap created.")

    product_group_column = "product" if "product" in cleaned_df.columns else "category" if "category" in cleaned_df.columns else None
    if product_group_column is not None:
        product_table = summarize_top_entities(cleaned_df, product_group_column, quantity_column="quantity" if "quantity" in cleaned_df.columns else None)
    else:
        product_table = pd.DataFrame()

    if "region" in cleaned_df.columns:
        region_table = summarize_top_entities(cleaned_df, "region", quantity_column="quantity" if "quantity" in cleaned_df.columns else None)
    else:
        region_table = pd.DataFrame()

    customer_metrics, customer_table = build_customer_behavior(cleaned_df)
    revenue_metrics, category_table, region_revenue_table = build_revenue_driver_metrics(cleaned_df)
    seasonality_metrics, monthly_sales_table = build_seasonality_metrics(cleaned_df)

    if product_group_column is not None and product_table.empty and not category_table.empty:
        product_table = category_table.rename(columns={"category": product_group_column})

    emit_progress(progress_callback, "summaries", 82, "Building business insight summaries and markdown report...")

    insights = generate_insight_summary(
        cleaned_df,
        summary,
        product_table,
        region_table,
        seasonality_metrics,
        customer_metrics,
        customer_table,
        revenue_metrics,
        summary["selected_roles"],
    )

    report_md = build_report(
        source_label=source_label,
        output_dir=output_dir,
        cleaned_path=cleaned_path,
        plot_paths=plot_paths,
        df=cleaned_df,
        summary=summary,
        insights=insights,
        missing_table=missing_table,
    )
    report_path = output_dir / "phase1_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    emit_progress(progress_callback, "complete", 100, "Phase 1 analysis complete.")

    return AnalysisArtifacts(
        cleaned_csv=cleaned_path,
        report_md=report_path,
        plot_paths={key: value for key, value in plot_paths.items() if value is not None},
        summary={
            **summary,
            "source_label": source_label,
            "demo_path": str(demo_path) if input_path is None else None,
            "seasonality_plot": seasonality_plot,
            "monthly_sales_table": monthly_sales_table,
            "category_table": category_table,
            "region_revenue_table": region_revenue_table,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean retail sales data, perform EDA, and generate a markdown insights report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None, help="Path to the input CSV or Excel file.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for cleaned data, plots, and report.")
    parser.add_argument("--demo-rows", type=int, default=1200, help="Number of rows to generate when no input file is provided.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for synthetic data generation.")
    return parser.parse_args()


def print_summary(artifacts: AnalysisArtifacts) -> None:
    summary = artifacts.summary
    print("Analysis complete.")
    print(f"Cleaned CSV: {artifacts.cleaned_csv}")
    print(f"Report: {artifacts.report_md}")
    print(f"Plots directory: {artifacts.report_md.parent / 'plots'}")
    if summary.get("demo_path"):
        print(f"Synthetic dataset: {summary['demo_path']}")
    print(f"Rows before cleaning: {summary['rows_before']}")
    print(f"Rows after cleaning: {summary['rows_after']}")
    print(f"Columns after cleaning: {summary['columns_after']}")
    print(f"Duplicate rows removed: {summary['duplicates_removed']}")
    print(f"Missing values before cleaning: {summary['missing_before']}")
    print(f"Missing values after cleaning: {summary['missing_after']}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser() if args.input else None
    output_dir = project_root() / args.output_dir
    artifacts = run_phase1_analysis(input_path=input_path, output_dir=output_dir, demo_rows=args.demo_rows, seed=args.seed)
    print_summary(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
