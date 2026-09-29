"""Unit tests for Creator Persona Tone-Matching & Stylometric LoRA Engine (engine.persona)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.assistant import ActionRecommendation, CreatorAssistantEngine
from engine.persona import (
    ConversationPair,
    CreatorPersonaProfile,
    LoRADatasetBuilder,
    PersonaExtractor,
)


def test_persona_profile_dataclass_and_serialization(tmp_path: Path) -> None:
    profile = CreatorPersonaProfile(
        creator_channel_id="UC_TEST_123",
        creator_name="TestCreator",
        total_creator_comments=50,
        avg_word_count=22.5,
        avg_sentence_count=2.0,
        vocab_entropy=6.5,
        type_token_ratio=0.55,
        caps_ratio=0.04,
        punctuation_intensity=1.2,
        exclamation_rate=0.8,
        question_rate=0.2,
        ellipsis_rate=0.1,
        emoji_frequency=1.5,
        top_emojis=["🔥", "🙏", "✨"],
        greeting_patterns=["Hey everyone!"],
        signoff_patterns=["Appreciate you!"],
        signature_phrases=["spot on", "deep dive"],
        avg_sentiment=0.6,
        avg_toxicity=0.01,
        sample_replies=[{"text": "Sample reply 🔥"}],
    )
    d = profile.to_dict()
    assert d["creator_name"] == "TestCreator"
    assert d["total_creator_comments"] == 50
    assert "🔥" in d["top_emojis"]

    # Test JSON roundtrip
    save_path = tmp_path / "persona.json"
    profile.save(save_path)
    assert save_path.exists()

    loaded = CreatorPersonaProfile.load(save_path)
    assert loaded is not None
    assert loaded.creator_name == "TestCreator"
    assert loaded.avg_word_count == 22.5
    assert loaded.top_emojis == ["🔥", "🙏", "✨"]


def test_format_stylometric_prompt() -> None:
    profile = CreatorPersonaProfile(
        creator_channel_id="UC_TEST",
        creator_name="ChannelMaster",
        avg_word_count=18.0,
        avg_sentence_count=1.5,
        exclamation_rate=0.7,
        top_emojis=["🔥", "🙏"],
        greeting_patterns=["Hey team!"],
        signoff_patterns=["Cheers!"],
        signature_phrases=["solid catch"],
        avg_sentiment=0.45,
    )
    prompt = profile.format_stylometric_prompt(tone="playful")
    assert "CREATOR PERSONA GUIDELINES" in prompt
    assert "ChannelMaster" in prompt
    assert "PLAYFUL" in prompt
    assert "🔥" in prompt
    assert "Hey team!" in prompt
    assert "Cheers!" in prompt
    assert "solid catch" in prompt


def test_conversation_pair_formatting() -> None:
    pair = ConversationPair(
        video_id="VID_001",
        thread_root_id="root_1",
        viewer_comment_id="v_1",
        viewer_author="Bob",
        viewer_text="How did you compute the Gini coefficient at 3:15?",
        creator_comment_id="c_1",
        creator_reply_text="Great question Bob! We used numpy trapezoidal integration. 🙏",
        likes_on_reply=12,
    )

    alpaca = pair.to_alpaca()
    assert "instruction" in alpaca
    assert "Viewer (Bob):" in alpaca["input"]
    assert "Great question Bob!" in alpaca["output"]

    chatml = pair.to_chatml(creator_name="Alex")
    assert "messages" in chatml
    assert len(chatml["messages"]) == 3
    assert chatml["messages"][0]["role"] == "system"
    assert "Alex" in chatml["messages"][0]["content"]
    assert chatml["messages"][1]["role"] == "user"
    assert chatml["messages"][2]["role"] == "assistant"


def test_persona_extractor_stylometrics() -> None:
    extractor = PersonaExtractor()
    sample_texts = [
        "Hey everyone! Thanks for watching. Absolutely spot on breakdown! 🔥",
        "Great question! We will dive deeper into this next episode. Appreciate you! 🙏",
        "Fair pushback! I definitely hear your perspective on this topic. Cheers!",
    ]
    profile = extractor._calculate_stylometrics(
        texts=sample_texts,
        creator_channel_id="UC_SAMPLE",
        creator_name="SampleHost",
    )
    assert profile.creator_name == "SampleHost"
    assert profile.total_creator_comments == 3
    assert profile.avg_word_count > 5
    assert profile.exclamation_rate > 0
    assert len(profile.top_emojis) >= 1
    assert "🔥" in profile.top_emojis or "🙏" in profile.top_emojis


def test_lora_dataset_builder_exports(tmp_path: Path) -> None:
    builder = LoRADatasetBuilder()
    pairs = builder.generate_mock_pairs(limit=5, creator_name="ProCreator")
    assert len(pairs) == 5

    res = builder.export_lora_dataset(
        pairs=pairs,
        output_dir=tmp_path / "lora_out",
        creator_name="ProCreator",
    )
    assert int(res["total_pairs"]) == 5

    alpaca_file = Path(res["alpaca"])
    assert alpaca_file.exists()
    lines = alpaca_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    first_record = json.loads(lines[0])
    assert "instruction" in first_record
    assert "output" in first_record

    chatml_file = Path(res["chatml"])
    assert chatml_file.exists()
    c_lines = chatml_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(c_lines) == 5

    modelfile = Path(res["modelfile"])
    assert modelfile.exists()
    assert "ProCreator" in modelfile.read_text(encoding="utf-8")

    script_path = builder.generate_training_script(output_dir=tmp_path / "lora_out")
    assert Path(script_path).exists()
    assert "LoraConfig" in Path(script_path).read_text(encoding="utf-8")


def test_assistant_engine_persona_integration(tmp_path: Path) -> None:
    persona_file = tmp_path / "test_persona.json"
    engine = CreatorAssistantEngine(persona_path=str(persona_file))

    # Extract persona in mock mode
    profile = engine.get_or_extract_persona(creator_name="PersonaHost", mock=True)
    assert profile.creator_name == "PersonaHost"
    assert persona_file.exists()

    # Draft reply conditioned on persona
    rec = ActionRecommendation(
        comment_id="c_pin_test",
        video_id="v_1",
        author_name="SuperFan",
        author_cohort="Champions",
        text="The architectural diagram at 10:00 clarifies everything!",
        like_count=50,
        reply_count=5,
        published_at="2026-09-06T12:00:00Z",
        minutes_since_upload=20.0,
        sentiment_score=0.8,
        toxicity_score=0.01,
        action_type="PIN",
        priority_score=95.0,
        action_rationale="High-signal feedback.",
    )
    draft = engine.draft_reply(rec, tone="playful")
    assert len(draft) > 15
    assert "SuperFan" in draft
    # Check that either top emoji or exclamation mark from persona is present
    assert any(em in draft for em in profile.top_emojis) or "!" in draft

    # Test export LoRA dataset from assistant engine
    lora_out = tmp_path / "lora_export"
    res = engine.export_lora_tuning_dataset(output_dir=str(lora_out), mock=True, limit=4)
    assert int(res["total_pairs"]) == 4
    assert Path(res["alpaca"]).exists()
