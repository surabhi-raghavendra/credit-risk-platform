"""Prompts for the natural-language-to-SQL feature.

The schema description here is the model's only view of the database, so it is
kept deliberately explicit about encodings that are easy to get wrong (negative
day counts, the retiree placeholder, TARGET polarity).
"""

from __future__ import annotations

TABLE_NAME = "applicants"

# The columns worth documenting in detail. The table mirrors the full master
# dataset, but these are the ones questions actually reach for.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "SK_ID_CURR": "INTEGER. Unique applicant id (primary key).",
    "TARGET": "INTEGER. 1 = the applicant defaulted, 0 = repaid. Overall default rate is about 8%.",
    "default_probability": "REAL. Probability of default predicted by the LightGBM model, 0 to 1.",
    "risk_band": "TEXT. Model risk band: 'Low' (<0.3), 'Medium' (0.3-0.6), 'High' (>=0.6).",
    "age_years": "REAL. Applicant age in years (already converted from DAYS_BIRTH).",
    "age_bracket": "TEXT. Age band: '20-29', '30-39', '40-49', '50-59', '60+'.",
    "years_employed": "REAL. Years in current job. NULL for applicants with no employment record.",
    "is_retiree_placeholder": "INTEGER. 1 when DAYS_EMPLOYED held the 365243 placeholder (typically retired).",
    "credit_to_income": "REAL. AMT_CREDIT divided by AMT_INCOME_TOTAL.",
    "annuity_to_income": "REAL. AMT_ANNUITY divided by AMT_INCOME_TOTAL.",
    "income_quartile": "TEXT. Income quartile: 'Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'.",
    "AMT_INCOME_TOTAL": "REAL. Declared annual income.",
    "AMT_CREDIT": "REAL. Credit amount of the loan applied for.",
    "AMT_ANNUITY": "REAL. Loan annuity (annual repayment amount).",
    "AMT_GOODS_PRICE": "REAL. Price of the goods the loan is financing.",
    "NAME_CONTRACT_TYPE": "TEXT. 'Cash loans' or 'Revolving loans'.",
    "CODE_GENDER": "TEXT. 'M', 'F', or 'XNA'.",
    "NAME_EDUCATION_TYPE": "TEXT. One of 'Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree'.",
    "NAME_FAMILY_STATUS": "TEXT. e.g. 'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'.",
    "NAME_HOUSING_TYPE": "TEXT. e.g. 'House / apartment', 'With parents', 'Rented apartment'.",
    "NAME_INCOME_TYPE": "TEXT. e.g. 'Working', 'Commercial associate', 'Pensioner', 'State servant'.",
    "OCCUPATION_TYPE": "TEXT. Occupation group, e.g. 'Laborers', 'Core staff', 'Managers'. Often NULL.",
    "ORGANIZATION_TYPE": "TEXT. Employer industry.",
    "CNT_CHILDREN": "INTEGER. Number of children.",
    "CNT_FAM_MEMBERS": "REAL. Household size.",
    "FLAG_OWN_CAR": "TEXT. 'Y' or 'N'.",
    "FLAG_OWN_REALTY": "TEXT. 'Y' or 'N'.",
    "EXT_SOURCE_1": "REAL. External credit bureau score 1, 0 to 1. Higher is safer. Often NULL.",
    "EXT_SOURCE_2": "REAL. External credit bureau score 2, 0 to 1. Higher is safer.",
    "EXT_SOURCE_3": "REAL. External credit bureau score 3, 0 to 1. Higher is safer.",
    "DAYS_BIRTH": "INTEGER. Age in days, stored NEGATIVE (-9461 means 25.9 years old). Prefer age_years.",
    "DAYS_EMPLOYED": "INTEGER. Employment length in days, NEGATIVE. The value 365243 is a placeholder, not real. Prefer years_employed.",
    "DAYS_REGISTRATION": "REAL. Days since registration was changed, NEGATIVE.",
    "DAYS_ID_PUBLISH": "INTEGER. Days since the identity document was issued, NEGATIVE.",
    "DAYS_LAST_PHONE_CHANGE": "REAL. Days since the phone number changed, NEGATIVE.",
    "REGION_RATING_CLIENT": "INTEGER. Internal region risk rating, 1 to 3 (3 is worst).",
    "REGION_POPULATION_RELATIVE": "REAL. Normalised population density of the applicant's region.",
    "bureau_loan_count": "INTEGER. Number of prior loans on file at the credit bureau. 0 if none.",
    "bureau_avg_credit_sum": "REAL. Average credit amount of those bureau loans. NULL if no bureau history.",
    "bureau_overdue_count": "INTEGER. Number of prior bureau loans currently overdue. 0 if none.",
    "previous_application_count": "INTEGER. Number of previous applications to this lender. 0 if none.",
    "previous_avg_amount": "REAL. Average amount of those previous applications. NULL if none.",
    "previous_approval_rate": "REAL. Share of previous applications approved, 0 to 1. NULL if none.",
}

SCHEMA_NOTES = """
Additional columns exist on the table and mirror the raw Home Credit dataset:
FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21 (0/1 document indicators),
AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT/YEAR (bureau enquiry counts),
REG_/LIVE_ region and city mismatch flags, WEEKDAY_APPR_PROCESS_START,
HOUR_APPR_PROCESS_START, and a family of building statistics such as
APARTMENTS_AVG, BASEMENTAREA_AVG, LANDAREA_AVG and their _MODE/_MEDI variants
(these are mostly NULL and rarely useful).
""".strip()

FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "How many applicants are in each risk band?",
        "SELECT risk_band,\n"
        "       COUNT(*) AS applicants\n"
        "FROM applicants\n"
        "GROUP BY risk_band\n"
        "ORDER BY applicants DESC;",
    ),
    (
        "What is the average loan amount for applicants who defaulted?",
        "SELECT ROUND(AVG(AMT_CREDIT), 2) AS avg_loan_amount,\n"
        "       COUNT(*) AS defaulters\n"
        "FROM applicants\n"
        "WHERE TARGET = 1;",
    ),
    (
        "Show the default rate by education type.",
        "SELECT NAME_EDUCATION_TYPE,\n"
        "       COUNT(*) AS applicants,\n"
        "       SUM(TARGET) AS defaults,\n"
        "       ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct\n"
        "FROM applicants\n"
        "GROUP BY NAME_EDUCATION_TYPE\n"
        "ORDER BY default_rate_pct DESC;",
    ),
    (
        "Which 5 occupations have the highest default rate among groups with at least 1000 applicants?",
        "SELECT OCCUPATION_TYPE,\n"
        "       COUNT(*) AS applicants,\n"
        "       ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct\n"
        "FROM applicants\n"
        "WHERE OCCUPATION_TYPE IS NOT NULL\n"
        "GROUP BY OCCUPATION_TYPE\n"
        "HAVING COUNT(*) >= 1000\n"
        "ORDER BY default_rate_pct DESC\n"
        "LIMIT 5;",
    ),
    (
        "Compare the average age of defaulters and non-defaulters.",
        "SELECT CASE WHEN TARGET = 1 THEN 'Defaulted' ELSE 'Repaid' END AS outcome,\n"
        "       COUNT(*) AS applicants,\n"
        "       ROUND(AVG(age_years), 2) AS avg_age_years\n"
        "FROM applicants\n"
        "GROUP BY outcome\n"
        "ORDER BY outcome;",
    ),
]


def _render_columns() -> str:
    return "\n".join(
        f"  {name} -- {description}" for name, description in COLUMN_DESCRIPTIONS.items()
    )


def _render_examples() -> str:
    blocks = []
    for question, sql in FEW_SHOT_EXAMPLES:
        blocks.append(f"Question: {question}\nSQL:\n{sql}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = f"""You translate questions about a consumer credit portfolio into a single SQLite SELECT query.

There is exactly one table, `{TABLE_NAME}`, with one row per loan applicant.

KEY COLUMNS
{_render_columns()}

{SCHEMA_NOTES}

RULES
1. Return ONE SQLite SELECT statement and nothing else. No markdown fences, no
   commentary, no trailing prose.
2. Never emit INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, or PRAGMA.
   Never emit more than one statement.
3. Query only the `{TABLE_NAME}` table.
4. TARGET = 1 means the applicant defaulted. Express a default rate as
   AVG(TARGET) * 100 and round it, e.g. ROUND(AVG(TARGET) * 100, 2).
5. Always alias aggregates with a readable name.
6. Prefer the pre-derived helper columns (age_years, age_bracket,
   years_employed, credit_to_income, annuity_to_income, income_quartile) over
   recomputing them from the negative DAYS_* columns.
7. Include COUNT(*) alongside grouped rates so the reader can judge sample size.
8. If the question asks for a ranking or a list of rows, add an explicit LIMIT
   (100 or fewer). Pure aggregate queries do not need a LIMIT.
9. SQLite has no CORR or STDDEV function. Build correlations arithmetically with
   SUM, COUNT and the sqrt() available via POWER/`*`, or answer with grouped
   averages instead, whichever is clearer.
10. Exclude NULL grouping keys with an IS NOT NULL filter unless the question
    specifically asks about missing values.

EXAMPLES
{_render_examples()}
"""


SUMMARY_SYSTEM_PROMPT = """You summarise the result of a database query for a credit risk analyst.

Write 1-2 plain sentences answering the question directly. Follow these rules
without exception:

1. Use ONLY numbers that appear in the result rows given to you. Never estimate,
   extrapolate, or recall figures from memory.
2. If the result set is empty, say plainly that the query returned no rows and
   that you therefore cannot answer. Do not guess what the answer might be.
3. If you are told the query failed, say plainly that it failed and do not
   attempt to answer the question.
4. Do not describe the SQL or the schema. Answer the business question.
5. Quote figures with the same precision they are given in. Do not round further.
6. If the rows do not actually answer the question that was asked, say so.
"""


def build_sql_user_prompt(question: str) -> str:
    """The user turn for SQL generation."""
    return f"Question: {question}\nSQL:"


def build_summary_user_prompt(
    question: str, results_csv: str, row_count: int, error: str | None = None
) -> str:
    """The user turn for result summarisation."""
    if error:
        return (
            f"Question: {question}\n\n"
            f"The query FAILED with this error: {error}\n\n"
            "Tell the user the query failed. Do not answer the question."
        )
    if row_count == 0:
        return (
            f"Question: {question}\n\n"
            "The query ran successfully but returned NO ROWS.\n\n"
            "Tell the user there were no matching results. Do not invent an answer."
        )
    return (
        f"Question: {question}\n\n"
        f"The query returned {row_count} row(s). Results as CSV:\n"
        f"{results_csv}\n\n"
        "Answer the question using only these numbers."
    )
