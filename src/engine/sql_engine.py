"""ytint // Zero-Copy Analytical SQL Engine & DuckDB Studio (src/engine/sql_engine.py)

High-performance, out-of-core analytical query engine powered by DuckDB.
Enables sub-10ms ANSI SQL queries, multi-table joins, and window functions
directly over all 80+ Apache Parquet tables across data/interim and data/output
without in-memory database loading or ETL copies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Safe standard output configuration for Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("ytint.sql_engine")

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore
    _DUCKDB_AVAILABLE = False


# ==============================================================================
# 1. DATA MODELS & TELEMETRY
# ==============================================================================

@dataclass
class QueryResult:
    """Encapsulates the execution result and performance telemetry of an analytical SQL query."""
    df: pd.DataFrame
    execution_time_ms: float
    row_count: int
    column_count: int
    columns: List[str]
    sql: str
    error: Optional[str] = None
    table_scanned: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes query metadata and top results to a structured dictionary."""
        return {
            "success": self.error is None,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "sql": self.sql,
            "error": self.error,
            "data": self.df.to_dict(orient="records") if not self.df.empty else []
        }

    def to_table_string(self, max_rows: int = 50, max_col_width: int = 40) -> str:
        """Formats the DataFrame as an aligned ASCII terminal table."""
        if self.error:
            return f"[ERROR] {self.error}"
        if self.df.empty:
            return f"(Empty result set — 0 rows returned in {self.execution_time_ms:.2f}ms)"
        
        # Display sample
        display_df = self.df.head(max_rows).copy()
        # Truncate string columns for terminal readability
        for col in display_df.select_dtypes(include=["object", "string"]).columns:
            display_df[col] = display_df[col].astype(str).apply(
                lambda x: (x[:max_col_width-3] + "...") if len(x) > max_col_width else x
            )
        table_str = display_df.to_string(index=False)
        telemetry = f"\n\n[INFO] {self.row_count:,} rows returned ({self.column_count} columns) in {self.execution_time_ms:.2f}ms"
        if len(self.df) > max_rows:
            telemetry += f" (showing first {max_rows} rows)"
        return table_str + telemetry


# ==============================================================================
# 2. CURATED ANALYTICAL SQL PRESETS
# ==============================================================================

PRESET_QUERIES: Dict[str, Dict[str, str]] = {
    "champions_rfm": {
        "title": "👑 VIP Champions & RFM Loyalty Distribution",
        "description": "Aggregates community authors by RFM loyalty cohort, computing comment volume, like influence, and mean sentiment.",
        "sql": """SELECT 
    rfm_cohort,
    count(*) AS author_count,
    round(avg(frequency), 1) AS avg_comments,
    round(avg(monetary), 1) AS avg_likes,
    round(avg(avg_sentiment), 3) AS avg_sentiment
FROM authors
GROUP BY rfm_cohort
ORDER BY avg_likes DESC;"""
    },
    "toxicity_outbreaks": {
        "title": "🚨 Extreme Toxicity Outbreaks & Flame-War Triggers",
        "description": "Filters comments with extreme toxicity (>0.70) and hostile negative sentiment, identifying root controversy sparks.",
        "sql": """SELECT 
    c.comment_id,
    c.video_id,
    c.author_display_name,
    c.like_count,
    round(c.vader_compound, 3) AS sentiment,
    round(c.toxicity, 3) AS toxicity,
    substring(c.text, 1, 90) AS comment_snippet
FROM comments c
WHERE c.toxicity > 0.70
ORDER BY c.toxicity DESC, c.like_count DESC
LIMIT 20;"""
    },
    "creator_causal_lift": {
        "title": "🎯 Creator Difference-in-Differences (DiD) Causal Lift",
        "description": "Summarizes causal uplift across reply counts, like accumulation, and sentiment elevation following creator interventions.",
        "sql": """SELECT 
    dimension,
    round(treated_mean, 2) AS creator_engaged_mean,
    round(control_mean, 2) AS organic_control_mean,
    round(absolute_lift, 2) AS absolute_lift,
    round(relative_lift_pct, 1) AS relative_lift_pct,
    interpretation
FROM creator_uplift
ORDER BY relative_lift_pct DESC;"""
    },
    "cib_astroturfing_rings": {
        "title": "🤖 Coordinated Inauthentic Behavior (CIB) Rings",
        "description": "Inspects detected synchronized commenting clusters, displaying ring density, author count, and synchronization latency.",
        "sql": """SELECT 
    ring_id,
    ring_size,
    author_count,
    total_synchronized_events,
    round(avg_interval_seconds, 2) AS mean_sync_seconds
FROM cib_rings
ORDER BY total_synchronized_events DESC
LIMIT 15;"""
    },
    "topical_valence_drift": {
        "title": "🧠 Topical Prevalence & Sentiment Landscape",
        "description": "Ranks conversation topics by total discussion volume, measuring aggregate valence and mean toxicity.",
        "sql": """SELECT 
    Topic AS topic_id,
    Name AS topic_name,
    Count AS comment_count,
    round(pct_positive * 100, 1) AS pct_positive,
    round(pct_negative * 100, 1) AS pct_negative,
    round(pct_neutral * 100, 1) AS pct_neutral
FROM topic_metadata
WHERE Topic >= 0
ORDER BY comment_count DESC
LIMIT 15;"""
    },
    "video_performance_benchmark": {
        "title": "🎬 Video Performance & Engagement Rate Benchmark",
        "description": "Ranks channel uploads by total views, likes, comments, and normalized engagement rates per 1,000 views.",
        "sql": """SELECT 
    video_id,
    title,
    total_views,
    video_likes,
    total_comments,
    round(comment_rate, 2) AS comments_per_1k_views
FROM videos
ORDER BY total_views DESC
LIMIT 15;"""
    },
    "bot_suspects_discrepancy": {
        "title": "🛡️ Bot Suspects & Repetition Discrepancy",
        "description": "Cross-references high-confidence bot classifications with author community stats and repetition ratios.",
        "sql": """SELECT 
    b.author_channel_id,
    a.author_display_name,
    b.classification AS bot_tier,
    b.unique_comments,
    b.total_comments,
    round(b.unique_ratio, 3) AS unique_ratio,
    a.rfm_cohort,
    a.monetary AS total_likes
FROM bot_classifications b
LEFT JOIN authors a ON b.author_channel_id = a.author_channel_id
WHERE b.is_bot_suspect = true
ORDER BY b.total_comments DESC
LIMIT 20;"""
    },
    "polarity_vs_engagement": {
        "title": "📊 Sentiment Polarity vs Upvote Attention Economy",
        "description": "Compares engagement volume, mean likes, and p90 attention distribution across sentiment classifications.",
        "sql": """SELECT 
    sentiment_label,
    n_comments,
    round(mean_likes, 2) AS avg_likes,
    round(median_likes, 2) AS median_likes,
    total_likes,
    p90_likes
FROM polarity_engagement
ORDER BY n_comments DESC;"""
    }
}


# ==============================================================================
# 3. CORE SQL ENGINE CLASS
# ==============================================================================

class SQLEngine:
    """Zero-copy analytical query engine powered by DuckDB over Parquet tables."""

    # Canonical aliases mapping user-friendly names to specific Parquet files
    CANONICAL_ALIASES: Dict[str, str] = {
        "comments": "comments_clean.parquet",
        "videos": "videos_clean.parquet",
        "authors": "authors_final.parquet",
        "bot_classifications": "bot_classifications.parquet",
        "cib_rings": "cib_rings.parquet",
        "cib_coordinated_comments": "cib_coordinated_comments.parquet",
        "creator_uplift": "creator_causal_uplift.parquet",
        "causal_impact": "causal_impact_summary.parquet",
        "topic_metadata": "topic_metadata.parquet",
        "audience_intent": "audience_intent_summary.parquet",
        "arrival_speed": "arrival_speed.parquet",
        "cohort_retention": "cohort_retention.parquet",
        "shelf_life": "shelf_life_likes.parquet",
        "polarity_engagement": "polarity_engagement.parquet",
        "emoji_signatures": "emoji_signatures.parquet",
        "sentiment_anomalies": "sentiment_anomalies.parquet",
        "corpus_quality": "corpus_quality.parquet",
        "cross_modal_reactions": "cross_modal_scene_reactions.parquet",
        "video_reaction_map": "video_reaction_map.parquet",
        "impersonation_alerts": "impersonation_alerts.parquet",
        "like_inflation": "like_inflation_authors.parquet",
        "top_k_concentration": "top_k_concentration.parquet",
        "network_bowtie": "network_bowtie_structure.parquet",
    }

    # Safety regex to reject destructive or unauthorized SQL operations
    RESTRICTED_KEYWORDS = re.compile(
        r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|ATTACH|DETACH|INSTALL|LOAD|PRAGMA)\b",
        re.IGNORECASE
    )

    def __init__(
        self,
        interim_dir: Optional[pathlib.Path | str] = None,
        output_dir: Optional[pathlib.Path | str] = None,
        database_path: str = ":memory:"
    ) -> None:
        """Initializes the DuckDB connection and registers all Parquet tables as views."""
        self.interim_dir = pathlib.Path(interim_dir or "data/interim")
        self.output_dir = pathlib.Path(output_dir or "data/output")
        self.database_path = database_path
        
        self.con: Any = None
        self.registered_tables: Dict[str, pathlib.Path] = {}
        
        if not _DUCKDB_AVAILABLE:
            logger.warning("DuckDB is not installed in the active environment. SQL capabilities will be unavailable.")
            return

        try:
            self.con = duckdb.connect(database_path)
            self._register_parquet_views()
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB connection: {e}")
            self.con = None

    def _register_parquet_views(self) -> None:
        """Scans interim and output directories, creating zero-copy views for all Parquet files."""
        if not self.con:
            return

        # 1. Discover all Parquet files across interim and output
        file_map: Dict[str, pathlib.Path] = {}

        if self.interim_dir.exists():
            for p in self.interim_dir.glob("*.parquet"):
                file_map[p.name] = p

        if self.output_dir.exists():
            for p in self.output_dir.glob("*.parquet"):
                file_map[p.name] = p

        # 2. Register Canonical Aliases first
        for alias, filename in self.CANONICAL_ALIASES.items():
            if filename in file_map:
                filepath = file_map[filename]
                posix_path = filepath.resolve().as_posix()
                try:
                    self.con.execute(f"CREATE OR REPLACE VIEW {alias} AS SELECT * FROM '{posix_path}';")
                    self.registered_tables[alias] = filepath
                except Exception as e:
                    logger.debug(f"Could not register canonical view '{alias}': {e}")

        # 3. Register all other Parquet files under their sanitized stem
        for filename, filepath in file_map.items():
            stem = filepath.stem
            # Sanitize identifier (replace non-alphanumerics with underscore)
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
            if safe_name not in self.registered_tables:
                posix_path = filepath.resolve().as_posix()
                try:
                    self.con.execute(f"CREATE OR REPLACE VIEW {safe_name} AS SELECT * FROM '{posix_path}';")
                    self.registered_tables[safe_name] = filepath
                except Exception as e:
                    logger.debug(f"Could not register view '{safe_name}': {e}")

        logger.info(f"Registered {len(self.registered_tables)} Parquet views in DuckDB.")

    def is_available(self) -> bool:
        """Returns True if DuckDB is installed and the connection is active."""
        return _DUCKDB_AVAILABLE and self.con is not None

    def execute_query(self, sql: str, limit: Optional[int] = None) -> QueryResult:
        """Executes an ANSI SQL query with performance timing and safety validation."""
        start_time = time.perf_counter()

        if not self.is_available():
            return QueryResult(
                df=pd.DataFrame(),
                execution_time_ms=0.0,
                row_count=0,
                column_count=0,
                columns=[],
                sql=sql,
                error="DuckDB is not installed or engine connection failed."
            )

        # Safety validation
        clean_sql = sql.strip()
        if not clean_sql:
            return QueryResult(
                df=pd.DataFrame(),
                execution_time_ms=0.0,
                row_count=0,
                column_count=0,
                columns=[],
                sql=sql,
                error="Query string is empty."
            )

        match = self.RESTRICTED_KEYWORDS.search(clean_sql)
        if match:
            forbidden = match.group(0).upper()
            return QueryResult(
                df=pd.DataFrame(),
                execution_time_ms=0.0,
                row_count=0,
                column_count=0,
                columns=[],
                sql=sql,
                error=f"Operation rejected: Destructive keyword '{forbidden}' is prohibited in read-only SQL Studio."
            )

        # Apply optional limit if not already specified in query
        executed_sql = clean_sql
        if limit is not None and limit > 0:
            if not re.search(r"\bLIMIT\s+\d+\b", clean_sql, re.IGNORECASE):
                executed_sql = f"{clean_sql.rstrip(';')} LIMIT {limit};"

        try:
            rel = self.con.execute(executed_sql)
            df = rel.df()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return QueryResult(
                df=df,
                execution_time_ms=elapsed_ms,
                row_count=len(df),
                column_count=len(df.columns),
                columns=list(df.columns),
                sql=executed_sql,
                error=None
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = str(e)
            logger.error(f"SQL Execution Error: {error_msg}")
            return QueryResult(
                df=pd.DataFrame(),
                execution_time_ms=elapsed_ms,
                row_count=0,
                column_count=0,
                columns=[],
                sql=executed_sql,
                error=error_msg
            )

    def get_table_catalog(self) -> List[Dict[str, Any]]:
        """Returns a list of all registered tables with row counts and file metadata."""
        catalog: List[Dict[str, Any]] = []
        if not self.is_available():
            return catalog

        for table_name, filepath in sorted(self.registered_tables.items()):
            size_mb = 0.0
            if filepath.exists():
                size_mb = round(filepath.stat().st_size / (1024 * 1024), 2)
            
            # Fast row count estimate
            row_count: Optional[int] = None
            try:
                res = self.con.execute(f"SELECT count(*) FROM {table_name};").fetchone()
                if res:
                    row_count = res[0]
            except Exception:
                row_count = None

            catalog.append({
                "table_name": table_name,
                "file_name": filepath.name,
                "path": str(filepath),
                "size_mb": size_mb,
                "row_count": row_count
            })
        return catalog

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Returns a DataFrame describing the schema and types of a specific table."""
        if not self.is_available():
            return pd.DataFrame()

        if table_name not in self.registered_tables:
            raise ValueError(f"Table '{table_name}' is not a registered view.")

        try:
            return self.con.execute(f"DESCRIBE SELECT * FROM {table_name};").df()
        except Exception as e:
            logger.error(f"Error describing table '{table_name}': {e}")
            return pd.DataFrame()

    def explain_query(self, sql: str) -> str:
        """Returns the DuckDB physical query execution plan."""
        if not self.is_available():
            return "DuckDB is not available."
        try:
            res = self.con.execute(f"EXPLAIN {sql};").df()
            return str(res.to_string())
        except Exception as e:
            return f"Error explaining query: {e}"

    def export_query_to_csv(self, sql: str, output_path: pathlib.Path | str) -> Tuple[bool, str]:
        """Executes the query and writes the full result directly to a CSV file."""
        res = self.execute_query(sql)
        if res.error:
            return False, res.error
        try:
            out_p = pathlib.Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            res.df.to_csv(out_p, index=False)
            return True, f"Successfully exported {res.row_count:,} rows to '{out_p}' in {res.execution_time_ms:.2f}ms."
        except Exception as e:
            return False, f"Failed to export CSV: {e}"

    def export_query_to_parquet(self, sql: str, output_path: pathlib.Path | str) -> Tuple[bool, str]:
        """Executes the query and writes the full result directly to an Apache Parquet file."""
        res = self.execute_query(sql)
        if res.error:
            return False, res.error
        try:
            out_p = pathlib.Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            res.df.to_parquet(out_p, index=False, compression="snappy")
            return True, f"Successfully exported {res.row_count:,} rows to '{out_p}' in {res.execution_time_ms:.2f}ms."
        except Exception as e:
            return False, f"Failed to export Parquet: {e}"


# ==============================================================================
# 4. INTERACTIVE CLI REPL & RUNNER
# ==============================================================================

def run_interactive_repl(engine: SQLEngine) -> None:
    """Launches an interactive ANSI SQL REPL with session commands."""
    print("=" * 72)
    print(" ⚡ ytint // Interactive Analytical SQL Studio (DuckDB Engine)")
    print(f" Registered Views: {len(engine.registered_tables)} Parquet tables")
    print(" Commands: \\dt (list tables) | \\d <table> (describe) | \\presets | \\q (quit)")
    print("=" * 72 + "\n")

    while True:
        try:
            query = input("ytint-sql> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive SQL studio.")
            break

        if not query:
            continue

        # Session slash-commands
        lower_q = query.lower()
        if lower_q in ("\\q", "exit", "quit", ":q"):
            print("Goodbye!")
            break
        elif lower_q in ("\\dt", "\\tables", "show tables"):
            catalog = engine.get_table_catalog()
            print("\nRegistered Parquet Views:")
            print(f"{'Table Name':<28} {'Rows':>10} {'Size':>10} {'Source Parquet'}")
            print("-" * 72)
            for item in catalog:
                rc = f"{item['row_count']:,}" if item['row_count'] is not None else "N/A"
                sz = f"{item['size_mb']} MB"
                print(f"{item['table_name']:<28} {rc:>10} {sz:>10} {item['file_name']}")
            print()
            continue
        elif lower_q.startswith("\\d ") or lower_q.startswith("describe "):
            parts = query.split(maxsplit=1)
            if len(parts) > 1:
                t_name = parts[1].strip().rstrip(";")
                try:
                    df_schema = engine.get_table_schema(t_name)
                    print(f"\nSchema for '{t_name}':")
                    print(df_schema.to_string(index=False))
                    print()
                except Exception as ex:
                    print(f"[ERROR] {ex}\n")
            continue
        elif lower_q in ("\\presets", "\\list-presets"):
            print("\nCurated Analytical SQL Presets:")
            for p_id, p_info in PRESET_QUERIES.items():
                print(f"  • {p_id:<22} : {p_info['title']}")
                print(f"    {p_info['description']}")
            print("\nRun with: \\run <preset_id>\n")
            continue
        elif lower_q.startswith("\\run "):
            p_id = query.split(maxsplit=1)[1].strip()
            if p_id in PRESET_QUERIES:
                print(f"\nRunning preset '{p_id}'...")
                query = PRESET_QUERIES[p_id]["sql"]
            else:
                print(f"[ERROR] Preset '{p_id}' not found. Type \\presets to view available options.\n")
                continue

        # Execute standard SQL query
        res = engine.execute_query(query)
        print(res.to_table_string(max_rows=40))
        print()


def main(args: Optional[List[str]] = None) -> int:
    """CLI Entrypoint for ytint-sql."""
    parser = argparse.ArgumentParser(
        prog="ytint-sql",
        description="ytint // High-Speed Zero-Copy Analytical SQL Engine (DuckDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  ytint-sql --list-tables
  ytint-sql "SELECT rfm_cohort, count(*) FROM authors GROUP BY 1"
  ytint-sql "SELECT * FROM comments WHERE toxicity > 0.8 LIMIT 10" --format json
  ytint-sql "SELECT * FROM authors WHERE total_likes > 100" --export top_authors.csv
  ytint-sql --preset champions_rfm
  ytint-sql --interactive
"""
    )

    parser.add_argument("query", nargs="?", default=None, help="ANSI SQL query string to execute")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive SQL REPL studio")
    parser.add_argument("--list-tables", "-l", action="store_true", help="List all registered Parquet views and record counts")
    parser.add_argument("--describe", "-d", type=str, default=None, help="Describe the schema and column types of a table")
    parser.add_argument("--preset", "-p", type=str, choices=list(PRESET_QUERIES.keys()), default=None, help="Execute a curated analytical preset query")
    parser.add_argument("--list-presets", action="store_true", help="List all available analytical preset queries")
    parser.add_argument("--format", "-f", choices=["table", "json", "csv"], default="table", help="Output format (default: table)")
    parser.add_argument("--limit", type=int, default=None, help="Max records to return")
    parser.add_argument("--export", "-e", type=str, default=None, help="Export query result to CSV or Parquet file path")
    parser.add_argument("--explain", action="store_true", help="Display DuckDB execution plan for the query")
    parser.add_argument("--interim-dir", type=str, default="data/interim", help="Path to interim Parquet directory")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Path to output Parquet directory")

    parsed = parser.parse_args(args)

    engine = SQLEngine(
        interim_dir=parsed.interim_dir,
        output_dir=parsed.output_dir
    )

    if not engine.is_available():
        print("[ERROR] DuckDB is required for ytint-sql but is not available in the current environment.", file=sys.stderr)
        print("Please install DuckDB: pip install duckdb>=1.0.0", file=sys.stderr)
        return 1

    # Mode 1: List tables
    if parsed.list_tables:
        catalog = engine.get_table_catalog()
        print(f"\nRegistered Parquet Views ({len(catalog)} tables):")
        print(f"{'Table Name':<28} {'Rows':>10} {'Size':>10} {'Source File'}")
        print("=" * 72)
        for item in catalog:
            rc = f"{item['row_count']:,}" if item['row_count'] is not None else "N/A"
            sz = f"{item['size_mb']} MB"
            print(f"{item['table_name']:<28} {rc:>10} {sz:>10} {item['file_name']}")
        print()
        return 0

    # Mode 2: Describe table schema
    if parsed.describe:
        try:
            df_schema = engine.get_table_schema(parsed.describe)
            print(f"\nSchema for table '{parsed.describe}':")
            print(df_schema.to_string(index=False))
            print()
            return 0
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1

    # Mode 3: List presets
    if parsed.list_presets:
        print("\nCurated Analytical SQL Presets:")
        for p_id, p_info in PRESET_QUERIES.items():
            print(f"  • {p_id:<22} : {p_info['title']}")
            print(f"    {p_info['description']}\n")
        return 0

    # Mode 4: Interactive REPL
    if parsed.interactive or (parsed.query is None and parsed.preset is None):
        run_interactive_repl(engine)
        return 0

    # Mode 5: Execute preset or ad-hoc query
    sql_to_run = parsed.query
    if parsed.preset:
        sql_to_run = PRESET_QUERIES[parsed.preset]["sql"]

    if parsed.explain:
        plan = engine.explain_query(sql_to_run)
        print("\nExecution Plan:")
        print(plan)
        return 0

    if parsed.export:
        if parsed.export.endswith(".parquet"):
            ok, msg = engine.export_query_to_parquet(sql_to_run, parsed.export)
        else:
            ok, msg = engine.export_query_to_csv(sql_to_run, parsed.export)
        if ok:
            print(f"[SUCCESS] {msg}")
            return 0
        else:
            print(f"[ERROR] {msg}", file=sys.stderr)
            return 1

    # Standard execution
    res = engine.execute_query(sql_to_run, limit=parsed.limit)

    if res.error:
        print(f"[ERROR] {res.error}", file=sys.stderr)
        return 1

    if parsed.format == "json":
        print(json.dumps(res.to_dict(), indent=2))
    elif parsed.format == "csv":
        print(res.df.to_csv(index=False))
    else:
        print(res.to_table_string(max_rows=parsed.limit or 50))

    return 0


if __name__ == "__main__":
    sys.exit(main())
