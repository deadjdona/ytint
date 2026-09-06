"""Unit tests for Zero-Copy Analytical SQL Engine (ytint-sql)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.sql_engine import (
    PRESET_QUERIES,
    QueryResult,
    SQLEngine,
    main,
)


def test_sql_engine_initialization() -> None:
    """Verifies that the engine initializes and registers Parquet tables."""
    engine = SQLEngine()
    assert engine.is_available() is True
    assert len(engine.registered_tables) > 10
    assert "comments" in engine.registered_tables or "authors" in engine.registered_tables


def test_canonical_aliases_registered() -> None:
    """Verifies that canonical user-friendly aliases are registered in DuckDB."""
    engine = SQLEngine()
    aliases = ["authors", "videos", "comments", "creator_uplift", "topic_metadata"]
    for alias in aliases:
        assert alias in engine.registered_tables


def test_query_execution_and_telemetry() -> None:
    """Tests basic query execution and telemetry capture."""
    engine = SQLEngine()
    res = engine.execute_query("SELECT 1 AS num, 'test' AS label;")
    assert res.error is None
    assert res.row_count == 1
    assert res.column_count == 2
    assert res.columns == ["num", "label"]
    assert res.execution_time_ms > 0.0
    assert isinstance(res.df, pd.DataFrame)
    assert res.df.iloc[0]["num"] == 1

    d = res.to_dict()
    assert d["success"] is True
    assert d["row_count"] == 1

    table_str = res.to_table_string()
    assert "test" in table_str


def test_query_limit_application() -> None:
    """Verifies that the limit parameter is respected."""
    engine = SQLEngine()
    res = engine.execute_query("SELECT * FROM authors", limit=5)
    assert res.error is None
    assert res.row_count == 5


def test_aggregation_query() -> None:
    """Tests an aggregation query grouping over authors."""
    engine = SQLEngine()
    res = engine.execute_query("SELECT rfm_cohort, count(*) AS cnt FROM authors GROUP BY 1 ORDER BY 2 DESC;")
    assert res.error is None
    assert res.row_count >= 1
    assert "rfm_cohort" in res.columns
    assert "cnt" in res.columns


def test_safety_restricted_keywords() -> None:
    """Verifies that destructive SQL operations are strictly blocked."""
    engine = SQLEngine()
    destructive_queries = [
        "DROP TABLE authors;",
        "DELETE FROM comments WHERE like_count < 5;",
        "UPDATE authors SET frequency = 0;",
        "TRUNCATE TABLE videos;",
        "ALTER TABLE comments ADD COLUMN hack INT;"
    ]
    for bad_sql in destructive_queries:
        res = engine.execute_query(bad_sql)
        assert res.error is not None
        assert "prohibited" in res.error.lower() or "rejected" in res.error.lower()
        assert res.row_count == 0


def test_table_catalog_and_describe() -> None:
    """Verifies catalog retrieval and schema descriptions."""
    engine = SQLEngine()
    catalog = engine.get_table_catalog()
    assert isinstance(catalog, list)
    assert len(catalog) > 10
    first = catalog[0]
    assert "table_name" in first
    assert "file_name" in first
    assert "size_mb" in first

    schema_df = engine.get_table_schema("authors")
    assert isinstance(schema_df, pd.DataFrame)
    assert not schema_df.empty
    assert "column_name" in schema_df.columns
    assert "column_type" in schema_df.columns

    with pytest.raises(ValueError):
        engine.get_table_schema("non_existent_table_xyz")


def test_explain_query() -> None:
    """Verifies that DuckDB physical query explanation works."""
    engine = SQLEngine()
    plan = engine.explain_query("SELECT rfm_cohort, count(*) FROM authors GROUP BY 1;")
    assert "EXPLAIN" in plan or "SCAN" in plan or "HASH_GROUP_BY" in plan or "authors" in plan


def test_preset_queries_execution() -> None:
    """Verifies that all curated analytical SQL presets execute without error."""
    engine = SQLEngine()
    assert len(PRESET_QUERIES) >= 6
    for p_id, p_info in PRESET_QUERIES.items():
        res = engine.execute_query(p_info["sql"])
        assert res.error is None, f"Preset '{p_id}' failed: {res.error}"
        assert res.execution_time_ms > 0.0


def test_export_csv_and_parquet(tmp_path: Path) -> None:
    """Verifies exporting query results directly to CSV and Parquet files."""
    engine = SQLEngine()
    csv_file = tmp_path / "test_out.csv"
    ok, msg = engine.export_query_to_csv("SELECT rfm_cohort, count(*) AS cnt FROM authors GROUP BY 1", csv_file)
    assert ok is True
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    parquet_file = tmp_path / "test_out.parquet"
    ok_p, msg_p = engine.export_query_to_parquet("SELECT rfm_cohort, count(*) AS cnt FROM authors GROUP BY 1", parquet_file)
    assert ok_p is True
    assert parquet_file.exists()
    assert parquet_file.stat().st_size > 0


def test_cli_execution(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verifies that the CLI entrypoint handles commands and options correctly."""
    # 1. List tables
    code = main(["--list-tables"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Registered Parquet Views" in captured.out

    # 2. Describe table
    code = main(["--describe", "authors"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Schema for table 'authors'" in captured.out

    # 3. List presets
    code = main(["--list-presets"])
    assert code == 0
    captured = capsys.readouterr()
    assert "champions_rfm" in captured.out

    # 4. Run preset
    code = main(["--preset", "champions_rfm"])
    assert code == 0
    captured = capsys.readouterr()
    assert "rfm_cohort" in captured.out

    # 5. Ad-hoc query with JSON format
    code = main(["SELECT 42 as num", "--format", "json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert data["data"][0]["num"] == 42

    # 6. Export via CLI
    cli_out = tmp_path / "cli_export.csv"
    code = main(["SELECT 1 as id, 'hello' as greeting", "--export", str(cli_out)])
    assert code == 0
    assert cli_out.exists()
