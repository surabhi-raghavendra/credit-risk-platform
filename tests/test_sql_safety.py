"""Tests for the SQL safety gate that guards every LLM-generated statement.

``check_sql_safety`` only parses text, so these tests are pure and offline: no
database, no source CSVs, and no Groq key.
"""

import unittest

from src.talk_to_data.nl_to_sql import check_sql_safety, is_safe_sql


class RejectsUnsafeStatements(unittest.TestCase):
    def assertRejected(self, sql: str) -> str:
        """Assert the statement is refused, and return the stated reason."""
        result = check_sql_safety(sql)
        self.assertFalse(result.ok, f"expected a rejection for: {sql}")
        self.assertTrue(result.reason.strip(), "a rejection must explain itself")
        self.assertFalse(is_safe_sql(sql))
        return result.reason

    def test_rejects_drop(self) -> None:
        reason = self.assertRejected("DROP TABLE applicants")
        self.assertIn("Only SELECT statements are allowed", reason)

    def test_rejects_multiple_statements(self) -> None:
        reason = self.assertRejected(
            "SELECT COUNT(*) FROM applicants; DROP TABLE applicants"
        )
        self.assertIn("Only one statement is allowed", reason)

    def test_rejects_pragma(self) -> None:
        # PRAGMA, ATTACH and VACUUM all parse to a Command node, not a Select.
        reason = self.assertRejected("PRAGMA table_info(applicants)")
        self.assertIn("Only SELECT statements are allowed", reason)
        self.assertRejected("ATTACH DATABASE 'other.db' AS other")

    def test_rejects_reading_sqlite_master(self) -> None:
        # Parses as a valid SELECT, so only the table whitelist stops it.
        reason = self.assertRejected("SELECT name FROM sqlite_master")
        self.assertIn("sqlite_master", reason)

    def test_rejects_insert_nested_in_subquery(self) -> None:
        # The guard walks the whole AST, so a write hidden inside an otherwise
        # ordinary SELECT is caught rather than just the top-level statement.
        for sql in (
            "WITH injected AS (INSERT INTO applicants (TARGET) VALUES (1)) "
            "SELECT * FROM injected",
            "SELECT * FROM (INSERT INTO applicants (TARGET) VALUES (1))",
        ):
            with self.subTest(sql=sql):
                self.assertRejected(sql)

    def test_rejects_select_into(self) -> None:
        self.assertRejected("SELECT * INTO backup FROM applicants")

    def test_rejects_delete(self) -> None:
        self.assertRejected("DELETE FROM applicants WHERE TARGET = 1")

    def test_rejects_empty_or_unparseable_sql(self) -> None:
        for sql in ("", "   ", "\n", "not sql at all !!!"):
            with self.subTest(sql=sql):
                self.assertRejected(sql)


class AcceptsReadOnlySelects(unittest.TestCase):
    def assertAccepted(self, sql: str) -> None:
        result = check_sql_safety(sql)
        self.assertTrue(result.ok, f"unexpectedly rejected because: {result.reason}")
        self.assertTrue(is_safe_sql(sql))

    def test_accepts_plain_select(self) -> None:
        self.assertAccepted(
            "SELECT risk_band, COUNT(*) AS applicants "
            "FROM applicants GROUP BY risk_band ORDER BY applicants DESC"
        )

    def test_accepts_cte(self) -> None:
        # A CTE name is not a real table and must not trip the whitelist.
        self.assertAccepted(
            "WITH banded AS (SELECT risk_band, TARGET FROM applicants) "
            "SELECT risk_band, ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct "
            "FROM banded GROUP BY risk_band"
        )

    def test_accepts_union_of_two_selects(self) -> None:
        for operator in ("UNION", "UNION ALL"):
            with self.subTest(operator=operator):
                self.assertAccepted(
                    "SELECT 'High' AS band, COUNT(*) AS n FROM applicants "
                    "WHERE risk_band = 'High' "
                    f"{operator} "
                    "SELECT 'Low' AS band, COUNT(*) AS n FROM applicants "
                    "WHERE risk_band = 'Low'"
                )


if __name__ == "__main__":
    unittest.main()
