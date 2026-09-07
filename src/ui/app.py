"""Credit Risk Intelligence Platform - Streamlit front end.

Run with ``streamlit run src/ui/app.py``. Every page reads real artifacts
produced by the earlier phases; nothing here is mocked.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (  # noqa: E402
    DB_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
)

ACCENT = "#0F6E6B"
ACCENT_DARK = "#0B534F"
BAND_COLOURS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}
# Amber needs dark text to stay readable inside the pill.
BAND_TEXT = {"Low": "#FFFFFF", "Medium": "#4A3200", "High": "#FFFFFF"}
INK = "#1B2A32"
MUTED = "#5B6B77"

SCREENSHOT_DIR = PROJECT_ROOT / "documents" / "screenshots"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_PATH = MODELS_DIR / "eval_results.json"
RULES_PATH = MODELS_DIR / "business_rules.json"
SCHEMA_PATH = MODELS_DIR / "feature_names.json"

CHART_FILES = [
    ("Default rate by education", "01_default_rate_by_education"),
    ("Default rate by income quartile", "02_default_rate_by_income_quartile"),
    ("Default rate by age bracket", "03_default_rate_by_age_bracket"),
    ("Credit-to-income by outcome", "04_credit_to_income_by_outcome"),
    ("Default rate by prior overdue loans", "05_default_rate_by_prior_overdue"),
]

st.set_page_config(
    layout="wide",
    page_title="Credit Risk Intelligence Platform",
    page_icon="💳",
)


# --------------------------------------------------------------------- styling


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: #F4F6F8; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1500px; }}
        #MainMenu, footer {{ visibility: hidden; }}

        .hero-title {{
            font-size: 1.9rem; font-weight: 700; color: {INK}; margin-bottom: .15rem;
        }}
        .hero-tag {{ font-size: 1rem; color: {MUTED}; margin-bottom: 1.4rem; }}
        .section-title {{
            font-size: 1.05rem; font-weight: 700; color: {INK};
            margin: .2rem 0 .7rem 0; padding-left: .6rem;
            border-left: 4px solid {ACCENT};
        }}

        .card {{
            background: #FFFFFF; border: 1px solid #E7EBEF; border-radius: 10px;
            padding: 1rem 1.15rem; margin-bottom: .9rem;
            box-shadow: 0 1px 3px rgba(16,32,40,.07), 0 1px 2px rgba(16,32,40,.04);
        }}
        .card h4 {{ margin: 0 0 .45rem 0; font-size: .98rem; color: {INK}; }}
        .card p  {{ margin: .2rem 0; font-size: .88rem; color: {MUTED}; }}
        .card ul {{ margin: .35rem 0 .5rem 1.1rem; padding: 0; }}
        .card li {{ font-size: .88rem; color: {INK}; margin-bottom: .18rem; }}

        .pill {{
            display: inline-block; padding: .2rem .78rem; border-radius: 999px;
            color: #FFFFFF; font-weight: 600; font-size: .8rem; letter-spacing: .2px;
        }}
        .pill-ghost {{
            display: inline-block; padding: .18rem .7rem; border-radius: 999px;
            font-weight: 600; font-size: .76rem; border: 1px solid #D8DEE4;
            color: {MUTED}; background: #FFFFFF;
        }}
        .pill-warn {{
            display: inline-block; padding: .2rem .78rem; border-radius: 999px;
            background: #FFF4DA; color: #8A5A00; border: 1px solid #F2D6A0;
            font-weight: 600; font-size: .8rem;
        }}

        div[data-testid="stMetric"] {{
            background: #FFFFFF; border: 1px solid #E7EBEF; border-radius: 10px;
            padding: .9rem 1rem; border-left: 4px solid {ACCENT};
            box-shadow: 0 1px 3px rgba(16,32,40,.07);
        }}
        div[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .82rem; }}
        div[data-testid="stMetricValue"] {{ color: {INK}; font-size: 1.65rem; }}

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button {{
            background: {ACCENT}; color: #FFFFFF; border: 0; border-radius: 8px;
            font-weight: 600; padding: .45rem 1.1rem;
        }}
        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {{
            background: {ACCENT_DARK}; color: #FFFFFF;
        }}

        div[data-testid="stExpander"] {{
            background: #FFFFFF; border: 1px solid #E7EBEF; border-radius: 10px;
        }}
        /* Bordered containers are used to card-wrap every chart block. */
        div[data-testid="stVerticalBlockBorderWrapper"][style*="border"],
        div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: #FFFFFF; border-color: #E7EBEF; border-radius: 10px;
            box-shadow: 0 1px 3px rgba(16,32,40,.07);
        }}
        .stChatMessage {{
            background: #FFFFFF; border: 1px solid #E7EBEF; border-radius: 10px;
        }}
        .muted {{ color: {MUTED}; font-size: .85rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def band_pill(band: str) -> str:
    colour = BAND_COLOURS.get(band, MUTED)
    text = BAND_TEXT.get(band, "#FFFFFF")
    return (
        f'<span class="pill" style="background:{colour};color:{text}">{band} risk</span>'
    )


def card(html: str) -> None:
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------- data loading


def artifact_missing(path: Path, command: str) -> bool:
    if path.is_file():
        return False
    st.error(f"`{path.relative_to(PROJECT_ROOT)}` not found. Run `{command}` first.")
    return True


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_chart(stem: str):
    """Interactive figure if the JSON export exists, else the PNG path."""
    json_path = SCREENSHOT_DIR / f"{stem}.json"
    if json_path.is_file():
        return pio.from_json(json_path.read_text(encoding="utf-8"))
    return None


def db_available() -> bool:
    return Path(DB_PATH).is_file()


def connect():
    return sqlite3.connect(f"file:{Path(DB_PATH).as_posix()}?mode=ro", uri=True)


@st.cache_data(show_spinner=False)
def load_snapshot() -> dict:
    with connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*), AVG(TARGET) * 100 FROM applicants"
        ).fetchone()
        bands = dict(
            connection.execute(
                "SELECT risk_band, COUNT(*) FROM applicants GROUP BY risk_band"
            ).fetchall()
        )
    return {"applicants": row[0], "default_rate_pct": row[1], "bands": bands}


@st.cache_data(show_spinner=False)
def load_missingness(top_n: int = 15) -> pd.DataFrame:
    with connect() as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(applicants)")
        ]
        total = connection.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]
        expression = ", ".join(
            f'SUM(CASE WHEN "{name}" IS NULL THEN 1 ELSE 0 END)' for name in columns
        )
        counts = connection.execute(f"SELECT {expression} FROM applicants").fetchone()

    frame = pd.DataFrame({"column": columns, "missing": counts})
    frame["missing_pct"] = (frame["missing"] / total * 100).round(2)
    return frame.sort_values("missing_pct", ascending=False).head(top_n)


NUMERIC_FORM_FIELDS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "age_years",
    "years_employed",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]
CATEGORICAL_FORM_FIELDS = [
    "NAME_EDUCATION_TYPE",
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
]


@st.cache_data(show_spinner=False)
def form_defaults() -> tuple[dict[str, float], dict[str, list[str]]]:
    """Population medians and category options that seed the scoring form."""
    columns = ", ".join(f'"{name}"' for name in NUMERIC_FORM_FIELDS + CATEGORICAL_FORM_FIELDS)
    with connect() as connection:
        frame = pd.read_sql_query(f"SELECT {columns} FROM applicants", connection)

    medians = {name: float(frame[name].median()) for name in NUMERIC_FORM_FIELDS}
    options = {}
    for name in CATEGORICAL_FORM_FIELDS:
        counts = frame[name].value_counts()
        # Most common category first, so the form opens on a typical applicant.
        options[name] = counts.index.tolist()
    return medians, options


@st.cache_data(show_spinner=False)
def sample_applicants(modulus: int = 200) -> pd.DataFrame:
    """Deterministic spread-out sample, cheap enough to SHAP on demand."""
    with connect() as connection:
        return pd.read_sql_query(
            f"SELECT * FROM applicants WHERE SK_ID_CURR % {modulus} = 0", connection
        )


@st.cache_resource(show_spinner=False)
def get_artifacts():
    from src.ml.predict import load_model_artifacts

    return load_model_artifacts()


@st.cache_data(show_spinner=False)
def global_shap_summary() -> pd.DataFrame:
    from src.ml.explain import get_global_shap_summary

    return get_global_shap_summary(sample_applicants())


# ----------------------------------------------------------------------- pages


def page_overview() -> None:
    st.markdown('<div class="hero-title">Credit Risk Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-tag">Default prediction, explanation, and plain-language '
        "querying over 307k consumer loan applications.</div>",
        unsafe_allow_html=True,
    )

    if artifact_missing(EVAL_PATH, "python -m src.ml.evaluate"):
        return
    if not db_available():
        st.error(
            f"Database not found at `{DB_PATH}`. Run "
            "`python -m src.talk_to_data.query_runner --rebuild` first."
        )
        return

    evaluation = load_json(str(EVAL_PATH))
    schema = load_json(str(SCHEMA_PATH)) if SCHEMA_PATH.is_file() else {}
    snapshot = load_snapshot()

    section("Live snapshot")
    columns = st.columns(4)
    columns[0].metric("Total applicants", f"{snapshot['applicants']:,}")
    columns[1].metric("Portfolio default rate", f"{snapshot['default_rate_pct']:.2f}%")
    columns[2].metric("Model ROC-AUC (holdout)", f"{evaluation['roc_auc']:.4f}")
    columns[3].metric("Model features", f"{schema.get('n_model_features', '-')}")

    left, right = st.columns([1.15, 1])
    with left:
        section("Risk banding")
        bands = snapshot["bands"]
        total = max(sum(bands.values()), 1)
        rows = "".join(
            f'<p>{band_pill(band)} &nbsp; <strong>{bands.get(band, 0):,}</strong> '
            f"applicants &nbsp;<span class='muted'>"
            f"({bands.get(band, 0) / total * 100:.1f}%)</span></p>"
            for band in ("Low", "Medium", "High")
        )
        card(
            f"<h4>Scored population</h4>{rows}"
            f"<p class='muted'>Thresholds: Low &lt; {RISK_THRESHOLD_LOW}, "
            f"High &ge; {RISK_THRESHOLD_HIGH}.</p>"
        )

    with right:
        section("Model performance")
        best = evaluation["at_best_f1_threshold"]
        card(
            "<h4>Held-out test set</h4>"
            f"<p>ROC-AUC <strong>{evaluation['roc_auc']:.4f}</strong> &nbsp;·&nbsp; "
            f"PR-AUC <strong>{evaluation['pr_auc']:.4f}</strong></p>"
            f"<p>At the F1-optimal cut-off ({best['threshold']:.2f}): precision "
            f"<strong>{best['precision']:.3f}</strong>, recall "
            f"<strong>{best['recall']:.3f}</strong></p>"
            f"<p class='muted'>{evaluation['test_rows']:,} rows, "
            f"{evaluation['test_positive_rate_pct']:.2f}% positive.</p>"
        )


def page_eda() -> None:
    section("Exploratory findings")
    figures = [(title, load_chart(stem)) for title, stem in CHART_FILES]
    if all(figure is None for _, figure in figures):
        st.error("No saved charts found. Run `python notebooks/eda.py` first.")
        return

    for index in range(0, len(figures), 2):
        columns = st.columns(2)
        for column, (title, figure) in zip(columns, figures[index : index + 2]):
            with column:
                if figure is not None:
                    figure.update_layout(height=380)
                    with st.container(border=True):
                        st.plotly_chart(figure, width="stretch")
                else:
                    png = SCREENSHOT_DIR / f"{title}.png"
                    if png.is_file():
                        st.image(str(png), caption=title, width="stretch")

    with st.expander("Data quality report"):
        if not db_available():
            st.info("Build the database to see the missingness report.")
            return
        schema = load_json(str(SCHEMA_PATH)) if SCHEMA_PATH.is_file() else {}
        dropped = schema.get("dropped_columns", [])
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("**Top 15 columns by missing values**")
            st.dataframe(
                load_missingness()[["column", "missing_pct"]],
                hide_index=True,
                width="stretch",
                height=300,
            )
        with right:
            st.markdown("**Handling**")
            st.markdown(
                f"- `DAYS_EMPLOYED = 365243` is a retiree placeholder, not a real "
                f"tenure. It is nulled before imputation and preserved as the binary "
                f"`is_retiree_placeholder` feature.\n"
                f"- Columns above 65% missing are dropped: "
                f"**{len(dropped)}** columns removed.\n"
                f"- Numeric features are median-imputed and standardised; "
                f"low-cardinality categoricals are one-hot encoded."
            )
            if dropped:
                st.caption(", ".join(dropped))


def risk_gauge(probability: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED},
                "bar": {"color": ACCENT, "thickness": 0.72},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, RISK_THRESHOLD_LOW * 100], "color": "#E8F3E9"},
                    {
                        "range": [RISK_THRESHOLD_LOW * 100, RISK_THRESHOLD_HIGH * 100],
                        "color": "#FDF3DA",
                    },
                    {"range": [RISK_THRESHOLD_HIGH * 100, 100], "color": "#FAE4E4"},
                ],
                "threshold": {
                    "line": {"color": BAND_COLOURS["High"], "width": 3},
                    "thickness": 0.8,
                    "value": RISK_THRESHOLD_HIGH * 100,
                },
            },
        )
    )
    figure.update_layout(
        height=290,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def page_prediction() -> None:
    section("Score an applicant")
    if artifact_missing(SCHEMA_PATH, "python -m src.ml.train"):
        return
    if not db_available():
        st.error("Build the database first so the form can seed realistic defaults.")
        return

    from src.ml.predict import predict_risk

    medians, options = form_defaults()
    left, right = st.columns([1.05, 1])

    with left:
        with st.form("applicant_form"):
            st.markdown("**Top features by model importance** — everything else is imputed")
            row1 = st.columns(3)
            ext1 = row1[0].slider("External score 1", 0.0, 1.0, round(medians["EXT_SOURCE_1"], 2), 0.01)
            ext2 = row1[1].slider("External score 2", 0.0, 1.0, round(medians["EXT_SOURCE_2"], 2), 0.01)
            ext3 = row1[2].slider("External score 3", 0.0, 1.0, round(medians["EXT_SOURCE_3"], 2), 0.01)

            row2 = st.columns(2)
            age = row2[0].slider("Age (years)", 21, 70, int(medians["age_years"]))
            employed = row2[1].slider(
                "Years in current job", 0.0, 45.0, round(medians["years_employed"], 1), 0.5
            )

            row3 = st.columns(2)
            income = row3[0].number_input(
                "Annual income", 25_000.0, 5_000_000.0, medians["AMT_INCOME_TOTAL"], 5_000.0
            )
            credit = row3[1].number_input(
                "Loan amount", 45_000.0, 5_000_000.0, medians["AMT_CREDIT"], 5_000.0
            )

            row4 = st.columns(2)
            annuity = row4[0].number_input(
                "Annual repayment", 1_500.0, 300_000.0, medians["AMT_ANNUITY"], 500.0
            )
            goods = row4[1].number_input(
                "Goods price", 40_000.0, 5_000_000.0, medians["AMT_GOODS_PRICE"], 5_000.0
            )

            row5 = st.columns(2)
            education = row5[0].selectbox("Education", options["NAME_EDUCATION_TYPE"])
            gender = row5[1].selectbox("Gender", options["CODE_GENDER"])

            row6 = st.columns(2)
            contract = row6[0].selectbox("Contract type", options["NAME_CONTRACT_TYPE"])
            overdue = row6[1].number_input("Prior overdue bureau loans", 0, 20, 0, 1)

            submitted = st.form_submit_button("Score applicant")

    if submitted:
        applicant = {
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "DAYS_BIRTH": -int(age * 365.25),
            "DAYS_EMPLOYED": -int(employed * 365.25),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "NAME_EDUCATION_TYPE": education,
            "CODE_GENDER": gender,
            "NAME_CONTRACT_TYPE": contract,
            "bureau_overdue_count": overdue,
        }
        st.session_state["last_applicant"] = applicant
        st.session_state["last_prediction"] = predict_risk(applicant)
        st.session_state.pop("last_explanation", None)

    with right:
        prediction = st.session_state.get("last_prediction")
        if not prediction:
            card(
                "<h4>No applicant scored yet</h4>"
                "<p>Fill in the form and press <strong>Score applicant</strong>. "
                "Fields you leave at their defaults are treated as the population "
                "median; any feature not shown here is imputed by the preprocessor.</p>"
            )
            return

        with st.container(border=True):
            st.plotly_chart(
                risk_gauge(prediction["probability"]), width="stretch"
            )

        borderline = ""
        if prediction["borderline_review"]:
            borderline = (
                f'<p><span class="pill-warn">⚑ Borderline - route to review</span></p>'
                f"<p class='muted'>Within "
                f"{prediction['distance_to_threshold']:.3f} of the "
                f"{prediction['nearest_threshold']} threshold "
                f"(margin {prediction['thresholds']['borderline_margin']}).</p>"
            )
        card(
            f"<h4>Decision</h4><p>{band_pill(prediction['risk_band'])}</p>"
            f"<p>Probability of default: <strong>"
            f"{prediction['probability'] * 100:.2f}%</strong></p>"
            f"{borderline}"
            f"<p class='muted'>{prediction['fields_supplied']} of "
            f"{prediction['fields_expected']} raw fields supplied; the rest imputed.</p>"
        )


def waterfall_figure(explanation: dict, probability: float) -> go.Figure:
    base = explanation["base_value_log_odds"]
    contributors = explanation["top_contributors"]
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    logit = math.log(clipped / (1 - clipped))
    remainder = logit - base - sum(item["shap_value"] for item in contributors)

    def shorten(label: str) -> str:
        return label if len(label) <= 24 else f"{label[:23]}…"

    labels = ["Base rate"] + [shorten(item["label"]) for item in contributors] + [
        "All other features",
        "Final score",
    ]
    values = [base] + [item["shap_value"] for item in contributors] + [remainder, 0]
    measures = ["absolute"] + ["relative"] * len(contributors) + ["relative", "total"]

    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[f"{value:+.2f}" if measure != "total" else f"{logit:.2f}"
                  for value, measure in zip(values, measures)],
            textposition="outside",
            connector={"line": {"color": "#C9D2D9"}},
            increasing={"marker": {"color": BAND_COLOURS["High"]}},
            decreasing={"marker": {"color": BAND_COLOURS["Low"]}},
            totals={"marker": {"color": ACCENT}},
        )
    )
    figure.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        yaxis_title="Log-odds contribution",
        showlegend=False,
    )
    return figure


def page_explainability() -> None:
    if artifact_missing(SCHEMA_PATH, "python -m src.ml.train"):
        return

    applicant = st.session_state.get("last_applicant")
    prediction = st.session_state.get("last_prediction")

    section("Why this applicant scored the way they did")
    if not applicant:
        card(
            "<h4>No prediction yet</h4><p>Score an applicant on the "
            "<strong>Risk Prediction</strong> page and the per-decision explanation "
            "will appear here.</p>"
        )
    else:
        if "last_explanation" not in st.session_state:
            from src.ml.explain import explain_prediction

            with st.spinner("Computing SHAP contributions..."):
                st.session_state["last_explanation"] = explain_prediction(applicant)
        explanation = st.session_state["last_explanation"]

        left, right = st.columns([1, 1.25])
        with left:
            bullets = "".join(
                f"<li>{item['explanation']} "
                f"<span class='pill-ghost'>{item['shap_value']:+.3f}</span></li>"
                for item in explanation["top_contributors"]
            )
            card(
                f"<h4>Top 5 drivers &nbsp; {band_pill(prediction['risk_band'])}</h4>"
                f"<ul>{bullets}</ul>"
                f"<p class='muted'>Values are SHAP contributions in log-odds. "
                f"Positive pushes risk up.</p>"
            )
        with right, st.container(border=True):
            st.plotly_chart(
                waterfall_figure(explanation, prediction["probability"]),
                width="stretch",
            )

    section("What drives the model overall")
    if not db_available():
        st.info("Build the database to compute the global SHAP summary.")
        return

    with st.spinner("Computing global SHAP values..."):
        summary = global_shap_summary()

    figure = go.Figure(
        go.Bar(
            x=summary["mean_abs_shap"][::-1],
            y=summary["label"][::-1],
            orientation="h",
            marker_color=ACCENT,
        )
    )
    figure.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=20, b=10),
        template="plotly_white",
        xaxis_title="Mean |SHAP| (log-odds)",
    )
    with st.container(border=True):
        st.plotly_chart(figure, width="stretch")


def page_rules() -> None:
    if artifact_missing(RULES_PATH, "python -m src.ml.rules"):
        return
    rules_data = load_json(str(RULES_PATH))
    variants = rules_data.get("variants", [])

    section("Surrogate rule sets")
    comparison = rules_data.get("comparison", {})
    columns = st.columns(len(variants))
    for column, name in zip(columns, variants):
        block = comparison.get(name, {})
        by_band = block.get("holdout_agreement_by_band_pct", {})
        with column:
            card(
                f"<h4>{block.get('label', name)}</h4>"
                f"<p>Holdout fidelity <strong>"
                f"{block.get('holdout_agreement_pct', 0):.2f}%</strong> · "
                f"{block.get('n_leaves', 0)} rules</p>"
                f"<p class='muted'>By band — High {by_band.get('High', 0):.1f}% · "
                f"Medium {by_band.get('Medium', 0):.1f}% · "
                f"Low {by_band.get('Low', 0):.1f}%</p>"
            )

    labels = {name: rules_data[name]["label"] for name in variants}
    chosen = st.radio(
        "Rule set",
        variants,
        format_func=lambda name: labels[name],
        horizontal=True,
    )
    variant = rules_data[chosen]
    st.caption(variant["description"])

    rules = variant["rules"]
    for index in range(0, len(rules), 2):
        row = st.columns(2)
        for column, rule in zip(row, rules[index : index + 2]):
            with column:
                conditions = "".join(f"<li>{clause}</li>" for clause in rule["conditions"])
                card(
                    f"<h4>Rule {rule['rule_id']} &nbsp; "
                    f"{band_pill(rule['predicted_band'])}</h4>"
                    f"<ul>{conditions}</ul>"
                    f"<p>Covers <strong>{rule['applicants_covered']:,}</strong> "
                    f"applicants ({rule['coverage_pct']:.2f}%) &nbsp;·&nbsp; "
                    f"actual default rate <strong>"
                    f"{rule['actual_default_rate_pct']:.2f}%</strong></p>"
                )


def render_turn(turn: dict) -> None:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar="💳"):
        st.write(turn["answer"])
        if turn.get("sql"):
            with st.expander("Generated SQL"):
                st.code(turn["sql"], language="sql")
        if turn.get("rows"):
            with st.expander(f"Result rows ({len(turn['rows'])})"):
                st.dataframe(
                    pd.DataFrame(turn["rows"]), hide_index=True, width="stretch"
                )
        if turn.get("error"):
            st.warning(turn["error"])


def page_chat() -> None:
    section("Ask a question about the portfolio")
    if not db_available():
        st.error(
            f"Database not found at `{DB_PATH}`. Run "
            "`python -m src.talk_to_data.query_runner --rebuild` first."
        )
        return

    from src.talk_to_data.query_runner import answer_question

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if not st.session_state["chat"]:
        st.caption(
            "Try: “Which age group has the highest default rate?” · "
            "“Average income for defaulters vs non-defaulters” · "
            "“Default rate by education type”"
        )

    for turn in st.session_state["chat"]:
        render_turn(turn)

    question = st.chat_input("Ask about defaults, income, education, risk bands...")
    if not question:
        return

    with st.spinner("Generating SQL and querying..."):
        try:
            outcome = answer_question(question)
            rows = (
                outcome.result.dataframe.to_dict("records")
                if outcome.result and outcome.result.ok
                else []
            )
            turn = {
                "question": question,
                "answer": outcome.answer,
                "sql": outcome.sql,
                "rows": rows,
                "error": "; ".join(outcome.errors) if outcome.errors else "",
            }
        except Exception as error:
            turn = {
                "question": question,
                "answer": "The assistant could not complete this request.",
                "sql": "",
                "rows": [],
                "error": str(error),
            }

    st.session_state["chat"].append(turn)
    render_turn(turn)


# ------------------------------------------------------------------------ main

PAGES = {
    "🏠 Overview": page_overview,
    "📊 EDA": page_eda,
    "🎯 Risk Prediction": page_prediction,
    "🔍 Explainability": page_explainability,
    "📋 Business Rules": page_rules,
    "💬 Talk to Data": page_chat,
}


def page_slug(name: str) -> str:
    return name.split(" ", 1)[1].lower().replace(" ", "-")


def main() -> None:
    inject_css()

    from streamlit_option_menu import option_menu

    slugs = [page_slug(name) for name in PAGES]
    requested = st.query_params.get("page", "")
    start_index = slugs.index(requested) if requested in slugs else 0

    with st.sidebar:
        st.markdown(
            f"<div style='font-weight:700;font-size:1.05rem;color:{ACCENT};"
            f"padding:.2rem 0 .6rem 0'>💳 Credit Risk</div>",
            unsafe_allow_html=True,
        )
        choice = option_menu(
            menu_title=None,
            options=list(PAGES),
            icons=[""] * len(PAGES),
            default_index=start_index,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                # Labels already carry emoji, so suppress the bootstrap icon slot.
                "icon": {"display": "none"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "3px 0",
                    "border-radius": "8px",
                    "color": INK,
                },
                "nav-link-selected": {"background-color": ACCENT, "color": "#FFFFFF"},
            },
        )
        st.markdown(
            f"<p class='muted' style='margin-top:1rem'>Thresholds<br>"
            f"Low &lt; {RISK_THRESHOLD_LOW} · High &ge; {RISK_THRESHOLD_HIGH}</p>",
            unsafe_allow_html=True,
        )

    st.query_params["page"] = page_slug(choice)
    PAGES[choice]()


if __name__ == "__main__":
    main()
