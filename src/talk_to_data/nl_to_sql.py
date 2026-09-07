"""Turn natural-language questions into validated SQLite SELECT statements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import sqlglot
from sqlglot import expressions as exp

from src.talk_to_data.prompt_templates import (
    SYSTEM_PROMPT,
    TABLE_NAME,
    build_sql_user_prompt,
)
from src.utils.config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_TABLES = {TABLE_NAME}
SQL_DIALECT = "sqlite"
TEMPERATURE = 0.0
MAX_TOKENS = 700

# Statement types that must never reach the database, even nested in a subquery.
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Command,  # PRAGMA, ATTACH, VACUUM and friends parse to this.
)

_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)


class LLMConfigurationError(RuntimeError):
    """Raised when the Groq client cannot be constructed."""


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of the static SQL safety check."""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


OPENAI_COMPAT_SUFFIX = "/openai/v1"


def groq_sdk_base_url(url: str) -> str:
    """Strip the OpenAI-compatibility suffix the Groq SDK appends itself.

    GROQ_BASE_URL is configured as the full OpenAI-compatible endpoint
    (https://api.groq.com/openai/v1), which is what a raw HTTP client or the
    openai package would need. The Groq SDK adds that path segment internally,
    so passing it through unchanged produces /openai/v1/openai/v1/... and a 404.
    """
    trimmed = url.rstrip("/")
    if trimmed.endswith(OPENAI_COMPAT_SUFFIX):
        trimmed = trimmed[: -len(OPENAI_COMPAT_SUFFIX)]
    return trimmed or url


@lru_cache(maxsize=1)
def get_client():
    """Cached Groq client pointed at the OpenAI-compatible endpoint."""
    if not GROQ_API_KEY:
        raise LLMConfigurationError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    from groq import Groq

    logger.info("Creating Groq client for model %s", LLM_MODEL)
    return Groq(api_key=GROQ_API_KEY, base_url=groq_sdk_base_url(GROQ_BASE_URL))


def complete(system_prompt: str, user_prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """Single-turn chat completion. No conversation history is retained."""
    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def strip_sql_fences(text: str) -> str:
    """Remove markdown fences and stray prose the model sometimes adds."""
    cleaned = _FENCE.sub("", text.strip()).strip()
    if "```" in cleaned:
        # A fenced block survived; keep only its contents.
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.removeprefix("sql").strip()
            if candidate.lower().startswith(("select", "with")):
                cleaned = candidate
                break
    return cleaned.rstrip().rstrip(";").strip()


def check_sql_safety(sql: str) -> SafetyResult:
    """Static validation: exactly one read-only SELECT against the known table."""
    if not sql or not sql.strip():
        return SafetyResult(False, "The generated SQL was empty.")

    try:
        statements = [
            statement
            for statement in sqlglot.parse(sql, read=SQL_DIALECT)
            if statement is not None
        ]
    except Exception as error:  # sqlglot raises several parse error types
        return SafetyResult(False, f"The SQL could not be parsed: {error}")

    if len(statements) == 0:
        return SafetyResult(False, "No SQL statement was found.")
    if len(statements) > 1:
        return SafetyResult(
            False, f"Only one statement is allowed; found {len(statements)}."
        )

    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union)):
        return SafetyResult(
            False, f"Only SELECT statements are allowed; got {type(statement).__name__.upper()}."
        )

    for node in statement.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return SafetyResult(
                False,
                f"Statement contains a forbidden operation ({type(node).__name__.upper()}).",
            )
        if isinstance(node, exp.Into):
            return SafetyResult(False, "SELECT ... INTO is not allowed.")

    referenced = {
        table.name.lower() for table in statement.find_all(exp.Table) if table.name
    }
    # Common table expressions define names that are not real tables.
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    unknown = referenced - ALLOWED_TABLES - cte_names
    if unknown:
        return SafetyResult(
            False, f"Query references unknown table(s): {', '.join(sorted(unknown))}."
        )

    return SafetyResult(True)


def is_safe_sql(sql: str) -> bool:
    """True when the SQL is a single read-only SELECT on the applicants table."""
    return bool(check_sql_safety(sql))


def nl_to_sql(question: str) -> str:
    """Generate a SQLite SELECT statement for a natural-language question.

    Deliberately stateless: no conversation history is carried between calls.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    logger.info("Generating SQL for: %s", question)
    raw = complete(SYSTEM_PROMPT, build_sql_user_prompt(question))
    sql = strip_sql_fences(raw)
    logger.info("Generated SQL: %s", sql.replace("\n", " ")[:160])
    return sql
