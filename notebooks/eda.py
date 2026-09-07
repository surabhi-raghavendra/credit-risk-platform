"""Exploratory data analysis for the credit risk master dataset.

Run with ``python notebooks/eda.py``. Every chart is written to
``documents/screenshots/`` as a PNG so the figures can be dropped straight into
a presentation without manual screenshotting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import build_master_dataset  # noqa: E402
from src.data.preprocessor import (  # noqa: E402
    RETIREE_PLACEHOLDER,
    Preprocessor,
    categorize_features,
    data_quality_report,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("eda")

SCREENSHOT_DIR = PROJECT_ROOT / "documents" / "screenshots"
CHART_WIDTH = 1000
CHART_HEIGHT = 600
CHART_SCALE = 2
DEFAULT_COLOURS = {"Repaid": "#2E86C1", "Defaulted": "#C0392B"}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_chart(fig, filename: str) -> Path:
    """Write a figure as a PNG for slides and as JSON for the Streamlit UI."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / filename
    fig.write_image(
        path, width=CHART_WIDTH, height=CHART_HEIGHT, scale=CHART_SCALE
    )
    # The UI reloads these to render interactive charts without re-reading the CSVs.
    fig.write_json(path.with_suffix(".json"))
    logger.info("Saved %s (+ .json)", path.relative_to(PROJECT_ROOT))
    return path


def default_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Applicant count, default count, and default rate for each group."""
    grouped = (
        df.groupby(column, observed=True)["TARGET"]
        .agg(applicants="size", defaults="sum", rate="mean")
        .reset_index()
    )
    grouped["default_rate_pct"] = (grouped["rate"] * 100).round(2)
    return grouped.drop(columns="rate")


def report_insight(number: int, title: str, table: pd.DataFrame, implication: str) -> None:
    print(f"\n--- Insight {number}: {title} ---")
    print(table.to_string(index=False))
    print(f"\nBusiness implication: {implication}")


# ---------------------------------------------------------------- section 1-3


def describe_dataset(df: pd.DataFrame) -> None:
    banner("1. SHAPE, DTYPES, TARGET DISTRIBUTION")

    print(f"Rows: {df.shape[0]:,}    Columns: {df.shape[1]:,}")

    dtype_counts = df.dtypes.astype(str).value_counts()
    print("\nColumn dtypes:")
    print(dtype_counts.to_string())

    counts = df["TARGET"].value_counts().sort_index()
    default_rate = df["TARGET"].mean() * 100
    print("\nTarget distribution:")
    print(f"  0 (repaid)    {counts.get(0, 0):>8,}  ({100 - default_rate:.2f}%)")
    print(f"  1 (defaulted) {counts.get(1, 0):>8,}  ({default_rate:.2f}%)")
    print(f"\nOverall default rate: {default_rate:.2f}%")
    print(f"Class imbalance ratio: {counts.get(0, 0) / max(counts.get(1, 1), 1):.1f} : 1")


def report_data_quality(df: pd.DataFrame) -> None:
    banner("2. DATA QUALITY")

    report = data_quality_report(df)
    top_missing = (
        report["columns"]
        .sort_values("missing_percentage", ascending=False)
        .head(15)
        .reset_index()
    )
    print("Top 15 columns by missing percentage:")
    print(top_missing.to_string(index=False))

    threshold_pct = Preprocessor().missing_threshold * 100
    dropped = report["columns"].index[
        report["columns"]["missing_percentage"] > threshold_pct
    ].tolist()
    print(
        f"\n{len(dropped)} columns exceed the {threshold_pct:.0f}% missing threshold "
        "and are dropped by the Preprocessor:"
    )
    for column in dropped:
        print(f"  {column}")

    banner("2b. DAYS_EMPLOYED ANOMALY")
    placeholder = df["DAYS_EMPLOYED"].eq(RETIREE_PLACEHOLDER)
    share = placeholder.mean() * 100
    cleaned = df["DAYS_EMPLOYED"].mask(placeholder)

    print(
        f"DAYS_EMPLOYED == {RETIREE_PLACEHOLDER} appears {placeholder.sum():,} times "
        f"({share:.2f}% of rows)."
    )
    print(
        "That is a placeholder for applicants with no current employment record "
        "(largely pensioners), not a real tenure of ~1000 years."
    )
    print(
        f"\n  Raw   -> min {df['DAYS_EMPLOYED'].min():>10,.0f}   "
        f"max {df['DAYS_EMPLOYED'].max():>10,.0f}   "
        f"mean {df['DAYS_EMPLOYED'].mean():>10,.0f}"
    )
    print(
        f"  Fixed -> min {cleaned.min():>10,.0f}   "
        f"max {cleaned.max():>10,.0f}   "
        f"mean {cleaned.mean():>10,.0f}"
    )
    print(
        "\nThe Preprocessor replaces the placeholder with NaN before median "
        "imputation and keeps the signal in the binary 'is_retiree_placeholder' "
        "flag, so the group stays visible to the model without corrupting the scale."
    )

    flagged_default = df.loc[placeholder, "TARGET"].mean() * 100
    other_default = df.loc[~placeholder, "TARGET"].mean() * 100
    print(
        f"\nDefault rate among flagged rows: {flagged_default:.2f}% "
        f"vs {other_default:.2f}% elsewhere - the flag carries real signal."
    )


def report_feature_groups(df: pd.DataFrame) -> None:
    banner("3. FEATURE CATEGORIES (categorize_features)")

    groups = categorize_features(df)
    for name, columns in groups.items():
        print(f"\n{name.upper()}  ({len(columns)} columns)")
        preview = ", ".join(columns[:12])
        suffix = f", ... (+{len(columns) - 12} more)" if len(columns) > 12 else ""
        print(f"  {preview}{suffix}" if columns else "  (none)")

    total = sum(len(columns) for columns in groups.values())
    print(f"\nTotal categorised: {total} of {df.shape[1]} columns "
          "(SK_ID_* and TARGET are excluded by design).")


# -------------------------------------------------------------------- insights


def insight_education(df: pd.DataFrame, overall: float) -> None:
    table = default_rate_by(df, "NAME_EDUCATION_TYPE").sort_values(
        "default_rate_pct", ascending=False
    )

    fig = px.bar(
        table,
        x="NAME_EDUCATION_TYPE",
        y="default_rate_pct",
        text="default_rate_pct",
        color="default_rate_pct",
        color_continuous_scale="Reds",
        labels={
            "NAME_EDUCATION_TYPE": "Education level",
            "default_rate_pct": "Default rate (%)",
        },
        title="Default rate by education level",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"Portfolio average {overall:.2f}%",
        annotation_position="top left",
        annotation_bgcolor="rgba(255,255,255,0.85)",
        annotation_font_color="#444444",
    )
    fig.update_layout(coloraxis_showscale=False, template="plotly_white")
    save_chart(fig, "01_default_rate_by_education.png")

    top = table.iloc[0]
    bottom = table.iloc[-1]
    implication = (
        f"{top['NAME_EDUCATION_TYPE']} applicants default at "
        f"{top['default_rate_pct']:.2f}% versus {bottom['default_rate_pct']:.2f}% for "
        f"{bottom['NAME_EDUCATION_TYPE']}, a "
        f"{top['default_rate_pct'] / bottom['default_rate_pct']:.1f}x gap, so education "
        "level alone justifies differentiated pricing or tighter limits at the lower end."
    )
    report_insight(1, "Default rate by education type", table, implication)


def insight_income_quartile(df: pd.DataFrame, overall: float) -> None:
    frame = df[["AMT_INCOME_TOTAL", "TARGET"]].dropna().copy()
    frame["income_quartile"] = pd.qcut(
        frame["AMT_INCOME_TOTAL"],
        4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    )

    bounds = frame.groupby("income_quartile", observed=True)["AMT_INCOME_TOTAL"].agg(
        min_income="min", max_income="max"
    )
    table = default_rate_by(frame, "income_quartile").merge(
        bounds.reset_index(), on="income_quartile"
    )

    fig = px.bar(
        table,
        x="income_quartile",
        y="default_rate_pct",
        text="default_rate_pct",
        color="default_rate_pct",
        color_continuous_scale="Reds",
        labels={
            "income_quartile": "Income quartile",
            "default_rate_pct": "Default rate (%)",
        },
        title="Default rate by income quartile",
        custom_data=["min_income", "max_income", "applicants"],
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate=(
            "%{x}<br>Default rate: %{y:.2f}%"
            "<br>Income range: %{customdata[0]:,.0f} - %{customdata[1]:,.0f}"
            "<br>Applicants: %{customdata[2]:,}<extra></extra>"
        ),
    )
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"Portfolio average {overall:.2f}%",
        annotation_position="top left",
        annotation_bgcolor="rgba(255,255,255,0.85)",
        annotation_font_color="#444444",
    )
    fig.update_layout(coloraxis_showscale=False, template="plotly_white")
    save_chart(fig, "02_default_rate_by_income_quartile.png")

    lowest = table.iloc[0]
    highest = table.iloc[-1]
    implication = (
        f"Default risk falls from {lowest['default_rate_pct']:.2f}% in the lowest income "
        f"quartile to {highest['default_rate_pct']:.2f}% in the highest, but the "
        f"{lowest['default_rate_pct'] - highest['default_rate_pct']:.2f} point spread is "
        "modest, so income on its own is a weak screen and should not carry much weight "
        "in the decision rule."
    )
    report_insight(2, "Default rate by income quartile", table, implication)


def insight_age_bracket(df: pd.DataFrame, overall: float) -> None:
    frame = df[["DAYS_BIRTH", "TARGET"]].dropna().copy()
    frame["age_years"] = -frame["DAYS_BIRTH"] / 365.25
    frame["age_bracket"] = pd.cut(
        frame["age_years"],
        bins=[20, 30, 40, 50, 60, 120],
        labels=["20-29", "30-39", "40-49", "50-59", "60+"],
        right=False,
    )

    table = default_rate_by(frame.dropna(subset=["age_bracket"]), "age_bracket")

    fig = px.bar(
        table,
        x="age_bracket",
        y="default_rate_pct",
        text="default_rate_pct",
        color="default_rate_pct",
        color_continuous_scale="Reds",
        labels={"age_bracket": "Age bracket", "default_rate_pct": "Default rate (%)"},
        title="Default rate by age bracket (derived from DAYS_BIRTH)",
        custom_data=["applicants"],
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate=(
            "Age %{x}<br>Default rate: %{y:.2f}%"
            "<br>Applicants: %{customdata[0]:,}<extra></extra>"
        ),
    )
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"Portfolio average {overall:.2f}%",
        annotation_position="top left",
        annotation_bgcolor="rgba(255,255,255,0.85)",
        annotation_font_color="#444444",
    )
    fig.update_layout(coloraxis_showscale=False, template="plotly_white")
    save_chart(fig, "03_default_rate_by_age_bracket.png")

    youngest = table.iloc[0]
    oldest = table.iloc[-1]
    implication = (
        f"Applicants aged {youngest['age_bracket']} default at "
        f"{youngest['default_rate_pct']:.2f}% while the {oldest['age_bracket']} group sits "
        f"at {oldest['default_rate_pct']:.2f}%, a monotonic decline with age that makes "
        "age one of the strongest single demographic predictors and an obvious candidate "
        "for risk-based pricing tiers."
    )
    report_insight(3, "Default rate by age bracket", table, implication)


def insight_credit_to_income(df: pd.DataFrame) -> None:
    frame = df[["AMT_CREDIT", "AMT_INCOME_TOTAL", "TARGET"]].dropna()
    frame = frame[frame["AMT_INCOME_TOTAL"] > 0].copy()
    frame["credit_to_income"] = frame["AMT_CREDIT"] / frame["AMT_INCOME_TOTAL"]
    frame["status"] = frame["TARGET"].map({0: "Repaid", 1: "Defaulted"})

    upper = frame["credit_to_income"].quantile(0.99)
    plot_frame = frame[frame["credit_to_income"] <= upper]

    summary = (
        frame.groupby("status")["credit_to_income"]
        .agg(
            applicants="size",
            median="median",
            mean="mean",
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
        )
        .round(2)
        .reset_index()
    )

    fig = px.histogram(
        plot_frame,
        x="credit_to_income",
        color="status",
        barmode="overlay",
        nbins=60,
        histnorm="percent",
        opacity=0.6,
        color_discrete_map=DEFAULT_COLOURS,
        labels={
            "credit_to_income": "Credit amount / annual income",
            "status": "Loan outcome",
        },
        title="Credit-to-income ratio by repayment outcome (99th percentile trimmed)",
    )
    fig.update_layout(
        template="plotly_white", yaxis_title="Share of group (%)", bargap=0.02
    )
    # The two medians nearly coincide, so stagger the labels to keep them legible.
    positions = {"Repaid": ("top left", -10), "Defaulted": ("top right", -30)}
    for status, colour in DEFAULT_COLOURS.items():
        median = summary.loc[summary["status"] == status, "median"].iloc[0]
        position, yshift = positions[status]
        fig.add_vline(
            x=median,
            line_dash="dash",
            line_color=colour,
            annotation_text=f"{status} median {median:.2f}x",
            annotation_position=position,
            annotation_yshift=yshift,
            annotation_font_color=colour,
        )
    save_chart(fig, "04_credit_to_income_by_outcome.png")

    defaulted_median = summary.loc[summary["status"] == "Defaulted", "median"].iloc[0]
    repaid_median = summary.loc[summary["status"] == "Repaid", "median"].iloc[0]

    deciles = frame.assign(
        ratio_decile=pd.qcut(
            frame["credit_to_income"], 10, labels=[f"D{i}" for i in range(1, 11)]
        )
    )
    decile_table = default_rate_by(deciles, "ratio_decile").merge(
        deciles.groupby("ratio_decile", observed=True)["credit_to_income"]
        .agg(min_ratio="min", max_ratio="max")
        .round(2)
        .reset_index(),
        on="ratio_decile",
    )

    lowest_decile = decile_table.iloc[0]
    highest_decile = decile_table.iloc[-1]
    spread = highest_decile["default_rate_pct"] - lowest_decile["default_rate_pct"]

    if abs(defaulted_median - repaid_median) < 0.25 and abs(spread) < 2:
        implication = (
            f"The two distributions sit almost on top of each other, with defaulters at a "
            f"{defaulted_median:.2f}x median credit-to-income multiple against "
            f"{repaid_median:.2f}x for repayers and the most leveraged decile defaulting at "
            f"{highest_decile['default_rate_pct']:.2f}% versus "
            f"{lowest_decile['default_rate_pct']:.2f}% for the least leveraged, so raw "
            "leverage is not a usable standalone screen here and the underwriting signal "
            "has to come from repayment behaviour instead, where prior bureau delinquency "
            "separates far more sharply."
        )
    else:
        direction = "more" if spread > 0 else "less"
        implication = (
            f"Defaulters carry a {defaulted_median:.2f}x median credit-to-income multiple "
            f"against {repaid_median:.2f}x for repayers, and the most leveraged decile "
            f"defaults {direction} often "
            f"({highest_decile['default_rate_pct']:.2f}% versus "
            f"{lowest_decile['default_rate_pct']:.2f}%), so the credit-to-income multiple "
            "carries enough signal to support an affordability cap."
        )

    report_insight(4, "Credit-to-income ratio by default status", summary, implication)
    print("\nDefault rate across credit-to-income deciles:")
    print(decile_table.to_string(index=False))


def insight_prior_overdue(df: pd.DataFrame, overall: float) -> None:
    frame = df[["bureau_overdue_count", "bureau_loan_count", "TARGET"]].copy()
    frame["overdue_bucket"] = pd.cut(
        frame["bureau_overdue_count"],
        bins=[-0.5, 0.5, 1.5, 2.5, float("inf")],
        labels=["0", "1", "2", "3+"],
    )

    table = default_rate_by(frame, "overdue_bucket")

    fig = px.bar(
        table,
        x="overdue_bucket",
        y="default_rate_pct",
        text="default_rate_pct",
        color="default_rate_pct",
        color_continuous_scale="Reds",
        labels={
            "overdue_bucket": "Prior credit-bureau loans currently overdue",
            "default_rate_pct": "Default rate (%)",
        },
        title="Default rate by number of prior overdue loans (bureau data)",
        custom_data=["applicants"],
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate=(
            "Prior overdue loans: %{x}<br>Default rate: %{y:.2f}%"
            "<br>Applicants: %{customdata[0]:,}<extra></extra>"
        ),
    )
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"Portfolio average {overall:.2f}%",
        annotation_position="top left",
        annotation_bgcolor="rgba(255,255,255,0.85)",
        annotation_font_color="#444444",
    )
    fig.update_layout(coloraxis_showscale=False, template="plotly_white")
    save_chart(fig, "05_default_rate_by_prior_overdue.png")

    clean = table.iloc[0]
    worst = table.iloc[-1]
    any_overdue = frame[frame["bureau_overdue_count"] > 0]
    implication = (
        f"Applicants with no overdue bureau loans default at "
        f"{clean['default_rate_pct']:.2f}% while those with {worst['overdue_bucket']} default at "
        f"{worst['default_rate_pct']:.2f}%, and although only {len(any_overdue):,} applicants "
        f"({len(any_overdue) / len(frame) * 100:.1f}%) carry any overdue record, the lift is "
        "large enough that this aggregate belongs in the model and in manual review triggers."
    )
    report_insight(5, "Default rate by prior overdue loan count", table, implication)


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    master = build_master_dataset()
    overall = master["TARGET"].mean() * 100

    describe_dataset(master)
    report_data_quality(master)
    report_feature_groups(master)

    banner("4. BUSINESS INSIGHTS")
    insight_education(master, overall)
    insight_income_quartile(master, overall)
    insight_age_bracket(master, overall)
    insight_credit_to_income(master)
    insight_prior_overdue(master, overall)

    banner("SAVED CHARTS")
    for path in sorted(SCREENSHOT_DIR.glob("*.png")):
        print(f"  {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
