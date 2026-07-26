# ytint

`ytint` turns YouTube comment exports into sentiment, topic, and timeline analytics.

## Architecture

1. **s00 — ingest:** reads `data/raw/commentsuite.sqlite3` and writes cleaned comment/video Parquet layers.
2. **s01 — enrich:** adds transformer-based sentiment and time-since-upload features.
3. **s02 — topics:** clusters top-level comments with BERTopic while retaining reply records in the canonical comments artifact.
4. **s03 — narrative:** creates daily volume timelines and optional z-score anomaly events.
5. **s04 — synthesis:** joins engagement and sentiment metrics into topic metadata.
6. **dashboard:** `src/app.py` provides the canonical Streamlit interface.

See [install.md](install.md) for environment setup and [run.txt](run.txt) for pipeline, validation, and dashboard commands.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

An empty `viral_events.parquet` is valid when no date exceeds the configured anomaly threshold; its schema is tested regardless of row count.
