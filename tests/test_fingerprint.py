"""Unit and integration tests for ytint Forensic Author Persona & Sockpuppet Fingerprinting Engine."""

import json
import os
import pathlib
import sys
import pytest
import numpy as np

from engine.fingerprint import (
    AuthorFingerprintEngine,
    AuthorPersona,
    SockpuppetPair,
    SockpuppetRing,
    AuthorForensicReport,
    compute_shannon_entropy,
    extract_emojis,
    cosine_similarity_1d,
    compute_circadian_entropy,
    main as fingerprint_main,
)


def test_compute_shannon_entropy():
    # Empty string has 0 entropy
    assert compute_shannon_entropy("") == 0.0
    # Single repeated word has 0 entropy
    assert compute_shannon_entropy("test test test test") == 0.0
    # Diverse text has positive entropy
    ent = compute_shannon_entropy("The quick brown fox jumps over the lazy dog")
    assert ent > 2.0


def test_extract_emojis():
    text = "Hello world! 🚀 This is awesome 💰👍"
    emojis = extract_emojis(text)
    assert len(emojis) >= 3
    assert "🚀" in emojis
    assert "💰" in emojis

    # No emojis
    assert extract_emojis("Just plain text with no emojis at all.") == []


def test_cosine_similarity_1d():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    v_zero = np.array([0.0, 0.0, 0.0])

    assert cosine_similarity_1d(v1, v2) == pytest.approx(1.0, abs=1e-5)
    assert cosine_similarity_1d(v1, v3) == pytest.approx(0.0, abs=1e-5)
    assert cosine_similarity_1d(v1, v_zero) == 0.0


def test_compute_circadian_entropy():
    # Single hour peak has 0 entropy
    single_peak = [0.0] * 24
    single_peak[12] = 1.0
    assert compute_circadian_entropy(single_peak) == pytest.approx(0.0, abs=1e-5)

    # Uniform 24-hour distribution has maximum entropy log2(24) ~= 4.585 bits
    uniform = [1.0 / 24.0] * 24
    max_entropy = compute_circadian_entropy(uniform)
    assert max_entropy == pytest.approx(4.585, abs=0.01)


def test_author_persona_dataclass_serialization():
    persona = AuthorPersona(
        author_channel_id="UC_TEST_001",
        author_display_name="Test_Author",
        total_comments=10,
        unique_videos=3,
        rfm_cohort="Loyal",
        vocab_entropy=4.2,
        avg_word_count=15.0,
        caps_ratio=0.1,
        punctuation_intensity=0.05,
        exclamation_rate=1.0,
        question_rate=0.5,
        ellipsis_rate=0.2,
        emoji_frequency=1.2,
        top_emojis=["🔥"],
        diurnal_histogram=[1.0 / 24.0] * 24,
        peak_posting_hour=14,
        circadian_entropy=4.58,
        avg_sentiment=0.35,
        avg_toxicity=0.02,
        sample_comments=["Great video!"],
        stylometric_vector=[0.5] * 10,
    )
    d = persona.to_dict()
    assert d["author_channel_id"] == "UC_TEST_001"
    assert d["author_display_name"] == "Test_Author"
    assert d["rfm_cohort"] == "Loyal"
    assert len(d["stylometric_vector"]) == 10


def test_mock_report_generation():
    engine = AuthorFingerprintEngine()
    report = engine.generate_mock_report()

    assert isinstance(report, AuthorForensicReport)
    assert report.total_authors_profiled >= 5
    assert len(report.high_confidence_pairs) >= 2
    assert len(report.clustered_rings) >= 1

    d = report.to_dict()
    assert "total_authors_profiled" in d
    assert "high_confidence_pairs" in d
    assert "clustered_rings" in d


def test_engine_build_personas_and_similarity():
    engine = AuthorFingerprintEngine()
    personas = engine.build_author_personas(min_comments=2, limit=50)
    assert len(personas) > 0

    first_key = next(iter(personas))
    p = personas[first_key]
    assert len(p.diurnal_histogram) == 24
    assert len(p.stylometric_vector) == 10

    # Compute pairs
    pairs = engine.compute_sockpuppet_pairs(personas=personas, min_score=60.0)
    assert isinstance(pairs, list)

    # Cluster rings
    rings = engine.cluster_sockpuppet_rings(pairs=pairs, personas=personas, min_similarity=75.0)
    assert isinstance(rings, list)


def test_export_report(tmp_path):
    engine = AuthorFingerprintEngine()
    report = engine.generate_mock_report()

    # JSON export
    json_path = tmp_path / "forensics.json"
    engine.export_report(report, json_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert "total_authors_profiled" in loaded
        assert len(loaded["high_confidence_pairs"]) > 0

    # CSV export
    csv_path = tmp_path / "pairs.csv"
    engine.export_report(report, csv_path)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 50


def test_cli_execution(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ytint-fingerprint", "--mock", "--list-rings"])
    fingerprint_main()
    captured = capsys.readouterr()
    assert "Forensic Author Persona & Sockpuppet Fingerprinting Engine" in captured.out
    assert "Clustered Sockpuppet & Alternate Account Rings" in captured.out
