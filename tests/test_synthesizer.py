"""Unit tests for the Natural Language Insight Synthesizer & RAG Agent (src/engine/synthesizer.py)."""

import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import patch, MagicMock

from engine.synthesizer import LLMClient, InsightSynthesizer, main


def test_llm_client_initialization_defaults():
    """Verify LLMClient initializes with sensible defaults."""
    client = LLMClient()
    assert client.provider in ["gemini", "ollama", "openai", "mock"]
    assert client.temperature == 0.3
    assert client.max_tokens == 1024


def test_llm_client_unsupported_provider():
    """Verify unsupported provider raises ValueError upon generation."""
    client = LLMClient(provider="anthropic_unsupported")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        client.generate("Hello world")


def test_llm_client_mock_generation():
    """Verify deterministic mock generation behavior across key prompt types."""
    client = LLMClient(provider="mock")

    # Executive Briefing
    exec_res = client.generate("Generate an executive briefing on channel health.")
    assert "Executive AI Intelligence Assessment" in exec_res
    assert "Channel Health & Resonance Verdict" in exec_res

    # Thread debate
    debate_res = client.generate("Summarize this debate thread and controversy.")
    assert "Thread Debate & Conflict Analysis" in debate_res
    assert "Controversy Trigger" in debate_res

    # General Q&A
    qa_res = client.generate("What is the peak posting time?")
    assert "ytint Intelligence Analysis" in qa_res


def test_llm_client_gemini_fallback_without_key():
    """Verify Gemini provider gracefully falls back to mock if no API key is provided."""
    with patch.dict("os.environ", {}, clear=True):
        client = LLMClient(provider="gemini", api_key=None)
        res = client.generate("Test prompt for Gemini")
        assert len(res) > 50


def test_llm_client_ollama_connection_fallback():
    """Verify Ollama provider gracefully falls back to mock if Ollama daemon is offline."""
    client = LLMClient(provider="ollama", base_url="http://localhost:99999")
    res = client.generate("Test prompt for Ollama")
    assert len(res) > 50


def test_llm_client_openai_fallback_without_key():
    """Verify OpenAI provider gracefully falls back to mock if no API key is provided."""
    with patch.dict("os.environ", {}, clear=True):
        client = LLMClient(provider="openai", api_key=None)
        res = client.generate("Test prompt for OpenAI")
        assert len(res) > 50


def test_synthesizer_instantiation_defaults():
    """Verify InsightSynthesizer initializes cleanly with auto-resolved paths."""
    synth = InsightSynthesizer()
    assert synth.interim_dir.exists()
    assert synth.output_dir.exists()
    assert isinstance(synth.llm, LLMClient)


def test_compile_context_dossier_schema():
    """Verify compile_context_dossier returns complete schema."""
    synth = InsightSynthesizer()
    dossier = synth.compile_context_dossier()

    expected_keys = [
        "total_comments",
        "total_videos",
        "loyalist_count",
        "driveby_count",
        "power_law_alpha",
        "bot_percentage",
        "like_inflation_percentage",
        "top_topics",
        "top_intents",
        "top_videos",
        "causal_creator_lift"
    ]
    for key in expected_keys:
        assert key in dossier

    assert dossier["total_comments"] >= 0
    assert dossier["total_videos"] >= 0
    assert isinstance(dossier["top_topics"], list)
    assert isinstance(dossier["top_intents"], dict)
    assert isinstance(dossier["top_videos"], list)


def test_synthesizer_resilience_empty_directory(tmp_path):
    """Verify synthesizer handles empty or missing directories without crashing."""
    empty_int = tmp_path / "interim"
    empty_out = tmp_path / "output"
    empty_int.mkdir()
    empty_out.mkdir()

    mock_client = LLMClient(provider="mock")
    synth = InsightSynthesizer(interim_dir=empty_int, output_dir=empty_out, llm_client=mock_client)

    dossier = synth.compile_context_dossier()
    assert dossier["total_comments"] == 0
    assert dossier["total_videos"] == 0
    assert dossier["loyalist_count"] == 0
    assert dossier["driveby_count"] == 0

    briefing = synth.synthesize_executive_briefing()
    assert len(briefing) > 100

    thread_summary = synth.summarize_thread_debate("non_existent_thread_id")
    assert "⚠️" in thread_summary

    qa_answer = synth.answer_query("How many bots are there?")
    assert len(qa_answer) > 50


def test_synthesize_executive_briefing_mock():
    """Verify executive briefing generation pipeline with mock client."""
    mock_client = LLMClient(provider="mock")
    synth = InsightSynthesizer(llm_client=mock_client)
    briefing = synth.synthesize_executive_briefing()

    assert "Executive AI Intelligence Assessment" in briefing
    assert "Channel Health & Resonance Verdict" in briefing
    assert "Strategic Recommendations" in briefing


def test_summarize_thread_debate_with_data():
    """Verify summarize_thread_debate runs on active dataset or provides clean diagnostics."""
    mock_client = LLMClient(provider="mock")
    synth = InsightSynthesizer(llm_client=mock_client)

    # Check if comments exist to pick a realistic comment ID
    df_comments = synth._safe_read_parquet("comments_clean.parquet")
    if not df_comments.empty and "comment_id" in df_comments.columns:
        test_id = str(df_comments["comment_id"].iloc[0])
        res = synth.summarize_thread_debate(test_id)
        assert len(res) > 50
        assert "Thread Debate & Conflict Analysis" in res or "⚠️" in res
    else:
        res = synth.summarize_thread_debate("dummy_id_123")
        assert "⚠️" in res


def test_answer_query_mock():
    """Verify RAG question answering pipeline with mock client."""
    mock_client = LLMClient(provider="mock")
    synth = InsightSynthesizer(llm_client=mock_client)
    ans = synth.answer_query("What is the ratio of loyalists to drive-by commenters?")

    assert len(ans) > 50
    assert "ytint Intelligence Analysis" in ans


def test_synthesizer_cli_mock(capsys):
    """Verify CLI main entrypoint executes cleanly in mock mode."""
    with patch("sys.argv", ["ytint-ai", "--mock", "--summary"]):
        main()
        captured = capsys.readouterr()
        assert "Executive AI Intelligence Assessment" in captured.out

    with patch("sys.argv", ["ytint-ai", "--mock", "--ask", "What are top themes?"]):
        main()
        captured = capsys.readouterr()
        assert "Query: 'What are top themes?'" in captured.out
        assert "ytint Intelligence Analysis" in captured.out
