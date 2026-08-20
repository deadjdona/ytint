import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
import pandas as pd
import tempfile
from pipeline.s35_audience_intent import classify_intent, run_audience_intent

def test_classify_intent_patterns():
    # Content Ideas
    assert classify_intent("Сделай видео про ядерную энергетику пожалуйста") == "CONTENT_IDEA"
    assert classify_intent("Can you please make a video about quantum computing?") == "CONTENT_IDEA"
    assert classify_intent("Жду вторую часть выпуска") == "CONTENT_IDEA"
    
    # Critiques
    assert classify_intent("Очень тихий звук, сделай микрофон погромче") == "CRITIQUE_FEEDBACK"
    assert classify_intent("Audio is too quiet, fix the volume") == "CRITIQUE_FEEDBACK"
    
    # Questions
    assert classify_intent("А почему в 04:12 он так сказал?") == "QUESTION_CONFUSION"
    assert classify_intent("Why did they decide to do this?") == "QUESTION_CONFUSION"
    
    # Appreciation
    assert classify_intent("Спасибо за отличный ролик, лучший контент!") == "APPRECIATION"
    assert classify_intent("Thank you for this amazing video, you are a legend!") == "APPRECIATION"
    
    # General Opinion / Debate
    assert classify_intent("Это обычная геополитическая ситуация.") == "DEBATE_OPINION"

def test_run_audience_intent_execution(monkeypatch, tmp_path):
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir()
    output_dir.mkdir()

    # Create dummy comments parquet
    dummy_comments = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4", "c5"],
        "video_id": ["v1", "v1", "v2", "v2", "v2"],
        "author_channel_id": ["a1", "a2", "a3", "a4", "a5"],
        "text_original": [
            "Сделай видео про ИИ",
            "Звук тихий на 02:30",
            "Почему автор так думает?",
            "Спасибо, крутой видос",
            "Просто коммент ни о чем"
        ],
        "like_count": [10, 5, 2, 50, 1],
        "vader_compound": [0.2, -0.4, 0.0, 0.9, 0.1]
    })
    dummy_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir)
        }
    }
    monkeypatch.setattr("pipeline.s35_audience_intent.load_config", lambda: fake_config)

    run_audience_intent()

    summary_file = output_dir / "audience_intent_summary.parquet"
    requests_file = output_dir / "audience_content_requests.parquet"

    assert summary_file.exists()
    assert requests_file.exists()

    df_summary = pd.read_parquet(summary_file)
    assert len(df_summary) == 5
    assert set(df_summary["audience_intent"]) == {
        "CONTENT_IDEA", "CRITIQUE_FEEDBACK", "QUESTION_CONFUSION", "APPRECIATION", "DEBATE_OPINION"
    }

    df_requests = pd.read_parquet(requests_file)
    assert len(df_requests) == 3
