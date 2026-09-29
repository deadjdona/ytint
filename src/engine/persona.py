"""ytint // Creator Persona Tone-Matching & Stylometric LoRA Engine (src/engine/persona.py)

Extracts creator stylometric fingerprints (lexical diversity, punctuation patterns,
emoji signatures, greeting/signoff habits, signature phrases), builds paired
instruction-tuning conversation datasets for LoRA fine-tuning, and provides
context-aware few-shot persona conditioning for draft reply generation.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import emoji
except ImportError:
    emoji = None

# Ensure src directory is in sys.path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

logger = logging.getLogger("ytint.persona")


# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================

@dataclass
class CreatorPersonaProfile:
    """Stylometric and behavioral fingerprint of a YouTube channel creator."""

    creator_channel_id: str
    creator_name: str
    total_creator_comments: int = 0
    avg_word_count: float = 0.0
    avg_sentence_count: float = 0.0
    vocab_entropy: float = 0.0
    type_token_ratio: float = 0.0
    caps_ratio: float = 0.0
    punctuation_intensity: float = 0.0
    exclamation_rate: float = 0.0
    question_rate: float = 0.0
    ellipsis_rate: float = 0.0
    emoji_frequency: float = 0.0
    top_emojis: List[str] = field(default_factory=list)
    greeting_patterns: List[str] = field(default_factory=list)
    signoff_patterns: List[str] = field(default_factory=list)
    signature_phrases: List[str] = field(default_factory=list)
    avg_sentiment: float = 0.0
    avg_toxicity: float = 0.0
    sample_replies: List[Dict[str, Any]] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the profile to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CreatorPersonaProfile:
        """Constructs a profile from a dictionary."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)

    def save(self, filepath: Union[str, Path]) -> str:
        """Saves the persona profile as a formatted JSON document."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Creator Persona Profile for '{self.creator_name}' to {p}")
        return str(p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Optional[CreatorPersonaProfile]:
        """Loads a persona profile from a JSON document."""
        p = Path(filepath)
        if not p.exists():
            logger.warning(f"Persona profile file does not exist: {p}")
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load persona profile from {p}: {e}")
            return None

    def format_stylometric_prompt(self, tone: str = "warm") -> str:
        """Formats the stylometric DNA into a natural-language prompt conditioning block."""
        emojis_str = " ".join(self.top_emojis[:5]) if self.top_emojis else "🔥 🙏 ✨"
        greetings_str = ", ".join([f'"{g}"' for g in self.greeting_patterns[:3]]) if self.greeting_patterns else '"Hey there!", "Thanks for watching!"'
        signoffs_str = ", ".join([f'"{s}"' for s in self.signoff_patterns[:3]]) if self.signoff_patterns else '"Appreciate you!", "Cheers!"'
        phrases_str = ", ".join([f'"{p}"' for p in self.signature_phrases[:4]]) if self.signature_phrases else 'None identified'

        excl_style = "frequent exclamation marks (enthusiastic/warm)" if self.exclamation_rate > 0.3 else "moderate punctuation"

        return (
            f"=== CREATOR PERSONA GUIDELINES ===\n"
            f"Creator Identity: {self.creator_name}\n"
            f"Tone Modulation: {tone.upper()}\n"
            f"Stylometric DNA:\n"
            f"- Typical Length: ~{int(self.avg_word_count or 18)} words ({self.avg_sentence_count:.1f} sentences).\n"
            f"- Punctuation Style: {excl_style}.\n"
            f"- Emoji Signature: {emojis_str} (Use naturally, ~{self.emoji_frequency:.1f} per comment).\n"
            f"- Authentic Greetings: {greetings_str}.\n"
            f"- Authentic Sign-offs: {signoffs_str}.\n"
            f"- Signature Expressions: {phrases_str}.\n"
            f"- Tone & Sentiment: Base positivity {self.avg_sentiment:+.2f}, zero toxicity, authentically creator-voiced.\n"
            f"CRITICAL CONSTRAINT: Do NOT sound like an AI assistant. Speak directly from the creator's viewpoint using their voice markers."
        )


@dataclass
class ConversationPair:
    """A paired (viewer_comment, creator_reply) dialogue instance."""

    video_id: str
    thread_root_id: str
    viewer_comment_id: str
    viewer_author: str
    viewer_text: str
    creator_comment_id: str
    creator_reply_text: str
    likes_on_reply: int = 0
    reply_latency_minutes: float = 0.0

    def to_alpaca(self) -> Dict[str, str]:
        """Formats into standard Alpaca instruction-tuning schema."""
        return {
            "instruction": "Reply to the following YouTube viewer comment authentically in the creator's personal voice, tone, and vocabulary.",
            "input": f"Viewer ({self.viewer_author}): \"{self.viewer_comment_text_clean()}\"",
            "output": self.creator_reply_text.strip(),
        }

    def to_chatml(self, creator_name: str = "Creator") -> Dict[str, Any]:
        """Formats into OpenAI / ChatML multi-turn message schema."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are {creator_name}, an authentic YouTube creator responding directly to viewers in your community comments.",
                },
                {
                    "role": "user",
                    "content": self.viewer_comment_text_clean(),
                },
                {
                    "role": "assistant",
                    "content": self.creator_reply_text.strip(),
                },
            ]
        }

    def viewer_comment_text_clean(self) -> str:
        """Cleans and truncates viewer comment text."""
        txt = self.viewer_text.strip().replace("\n", " ")
        return txt[:300]


# ==============================================================================
# 2. PERSONA EXTRACTOR
# ==============================================================================

class PersonaExtractor:
    """Extracts creator stylometric profiles and indexes historical creator responses."""

    def __init__(
        self,
        comments_path: Union[str, Path] = "data/interim/comments_clean.parquet",
        videos_path: Union[str, Path] = "data/interim/videos_clean.parquet",
    ) -> None:
        self.comments_path = Path(comments_path)
        self.videos_path = Path(videos_path)
        self._embedder = None

    def _get_embedder(self):
        """Lazy-loads the sentence-transformers model for semantic retrieval."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.debug(f"Could not load SentenceTransformer: {e}")
                self._embedder = False
        return self._embedder if self._embedder is not False else None

    def extract_profile(
        self,
        creator_channel_id: Optional[str] = None,
        creator_name: Optional[str] = None,
        mock: bool = False,
    ) -> CreatorPersonaProfile:
        """Analyzes comments written by the creator and constructs their stylometric profile."""
        if mock or not self.comments_path.exists():
            return self.generate_mock_profile(creator_name=creator_name or "Alex Creator")

        try:
            # Determine creator identification
            df_c = pd.read_parquet(self.comments_path)
            if df_c.empty:
                return self.generate_mock_profile(creator_name=creator_name or "Alex Creator")

            # Try to identify creator channel if not provided
            target_cid = creator_channel_id
            target_name = creator_name

            if not target_cid and self.videos_path.exists():
                try:
                    df_v = pd.read_parquet(self.videos_path, columns=["channel_id", "channel_title"])
                    if not df_v.empty and "channel_id" in df_v.columns:
                        target_cid = str(df_v.iloc[0]["channel_id"])
                        if not target_name and "channel_title" in df_v.columns:
                            target_name = str(df_v.iloc[0]["channel_title"])
                except Exception:
                    pass

            # Filter comments by creator channel id or heuristics
            creator_mask = pd.Series([False] * len(df_c), index=df_c.index)
            if target_cid and "author_channel_id" in df_c.columns:
                creator_mask |= (df_c["author_channel_id"] == target_cid)

            # Secondary heuristic: comments where creator hearted or pinned or where author name matches
            if creator_mask.sum() < 3 and target_name and "author_display_name" in df_c.columns:
                creator_mask |= df_c["author_display_name"].str.contains(target_name, case=False, na=False)

            # If still fewer than 3 comments found, find the author with creator-like badge or most hearted replies
            if creator_mask.sum() < 3:
                # Fall back to synthetic mock profile with realistic stylometrics
                logger.info("Fewer than 3 creator comments in dataset; generating enriched baseline persona.")
                return self.generate_mock_profile(creator_name=target_name or "Alex Creator")

            df_creator = df_c[creator_mask].copy()
            resolved_name = target_name or (df_creator.iloc[0].get("author_display_name", "Creator") if not df_creator.empty else "Creator")
            resolved_cid = target_cid or (df_creator.iloc[0].get("author_channel_id", "CREATOR_CID") if not df_creator.empty else "CREATOR_CID")

            # Compute stylometrics
            texts = df_creator["text"].dropna().astype(str).tolist()
            profile = self._calculate_stylometrics(
                texts=texts,
                creator_channel_id=str(resolved_cid),
                creator_name=str(resolved_name),
                df_creator=df_creator,
            )
            return profile

        except Exception as e:
            logger.warning(f"Error during persona extraction: {e}. Falling back to baseline profile.")
            return self.generate_mock_profile(creator_name=creator_name or "Alex Creator")

    def _calculate_stylometrics(
        self,
        texts: List[str],
        creator_channel_id: str,
        creator_name: str,
        df_creator: Optional[pd.DataFrame] = None,
    ) -> CreatorPersonaProfile:
        """Performs statistical NLP stylometric calculations on author comment texts."""
        if not texts:
            return self.generate_mock_profile(creator_name=creator_name)

        total_comments = len(texts)
        all_words: List[str] = []
        word_counts: List[int] = []
        sentence_counts: List[int] = []
        caps_ratios: List[float] = []
        exclamation_counts: List[int] = []
        question_counts: List[int] = []
        ellipsis_counts: List[int] = []
        emojis_collected: List[str] = []

        greetings: List[str] = []
        signoffs: List[str] = []

        # Regex patterns
        greeting_re = re.compile(r"^(hey|hi|hello|yo|good morning|thanks for watching|what's up|welcome)[^.!?\n]*", re.IGNORECASE)
        signoff_re = re.compile(r"(cheers|best|thanks again|appreciate you|stay tuned|keep building|catch you in the next one|see you)[^.!?\n]*$", re.IGNORECASE)

        for text in texts:
            clean_txt = text.strip()
            # Words
            tokens = re.findall(r"\b\w+\b", clean_txt.lower())
            all_words.extend(tokens)
            word_counts.append(len(tokens))

            # Sentences
            sents = [s for s in re.split(r"[.!?]+", clean_txt) if s.strip()]
            sentence_counts.append(max(1, len(sents)))

            # Caps ratio
            letters = [c for c in clean_txt if c.isalpha()]
            caps = [c for c in letters if c.isupper()]
            caps_ratios.append(len(caps) / max(1, len(letters)))

            # Punctuation
            exclamation_counts.append(clean_txt.count("!"))
            question_counts.append(clean_txt.count("?"))
            ellipsis_counts.append(clean_txt.count("...") + clean_txt.count("…"))

            # Emojis
            if emoji:
                found_emojis = [c["emoji"] for c in emoji.emoji_list(clean_txt)]
                emojis_collected.extend(found_emojis)

            # Greetings / Signoffs
            g_match = greeting_re.search(clean_txt)
            if g_match:
                greetings.append(g_match.group(0).strip())

            s_match = signoff_re.search(clean_txt)
            if s_match:
                signoffs.append(s_match.group(0).strip())

        # Vocabulary Entropy & TTR
        total_tokens = len(all_words)
        vocab_counts = collections.Counter(all_words)
        unique_tokens = len(vocab_counts)
        ttr = unique_tokens / max(1, total_tokens)

        entropy = 0.0
        for count in vocab_counts.values():
            p = count / max(1, total_tokens)
            entropy -= p * math.log2(p)

        # Emoji frequency
        top_emojis = [e for e, _ in collections.Counter(emojis_collected).most_common(8)]
        emoji_freq = len(emojis_collected) / max(1, total_comments)

        # Signature Phrases (frequent bigrams/trigrams)
        phrases: List[str] = []
        if len(all_words) >= 2:
            bigrams = [f"{all_words[i]} {all_words[i+1]}" for i in range(len(all_words) - 1)]
            bg_counts = collections.Counter(bigrams)
            stop_phrases = {"in the", "of the", "to the", "it is", "this is", "on the", "at the", "for the"}
            common_bg = [bg for bg, c in bg_counts.most_common(12) if c >= 2 and bg not in stop_phrases]
            phrases.extend(common_bg[:5])

        # Greeting patterns (deduplicated common)
        top_greetings = [g for g, _ in collections.Counter(greetings).most_common(4)]
        top_signoffs = [s for s, _ in collections.Counter(signoffs).most_common(4)]

        # Sentiments
        avg_sentiment = 0.45
        avg_toxicity = 0.01
        if df_creator is not None:
            if "vader_compound" in df_creator.columns:
                avg_sentiment = float(df_creator["vader_compound"].dropna().mean() or 0.45)
            if "toxicity" in df_creator.columns:
                avg_toxicity = float(df_creator["toxicity"].dropna().mean() or 0.01)

        # Sample replies
        samples = []
        for idx, text in enumerate(texts[:10]):
            samples.append({
                "index": idx,
                "text": text,
                "word_count": len(text.split()),
            })

        return CreatorPersonaProfile(
            creator_channel_id=creator_channel_id,
            creator_name=creator_name,
            total_creator_comments=total_comments,
            avg_word_count=round(float(np.mean(word_counts or [20.0])), 1),
            avg_sentence_count=round(float(np.mean(sentence_counts or [2.0])), 1),
            vocab_entropy=round(entropy, 2),
            type_token_ratio=round(ttr, 3),
            caps_ratio=round(float(np.mean(caps_ratios or [0.03])), 3),
            punctuation_intensity=round((sum(exclamation_counts) + sum(question_counts)) / max(1, total_comments), 2),
            exclamation_rate=round(sum(exclamation_counts) / max(1, total_comments), 2),
            question_rate=round(sum(question_counts) / max(1, total_comments), 2),
            ellipsis_rate=round(sum(ellipsis_counts) / max(1, total_comments), 2),
            emoji_frequency=round(emoji_freq, 2),
            top_emojis=top_emojis or ["🔥", "🙏", "✨", "❤️"],
            greeting_patterns=top_greetings or ["Hey everyone!", "Thanks for tuning in!"],
            signoff_patterns=top_signoffs or ["Appreciate you!", "Cheers!"],
            signature_phrases=phrases or ["deep dive", "great point", "stay tuned"],
            avg_sentiment=round(avg_sentiment, 2),
            avg_toxicity=round(avg_toxicity, 3),
            sample_replies=samples,
        )

    def find_similar_replies(
        self,
        query_text: str,
        profile: CreatorPersonaProfile,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieves historical creator replies that are semantically closest to the query."""
        if not profile.sample_replies:
            return []

        embedder = self._get_embedder()
        if not embedder:
            # Fallback: return top samples by word count / presence
            return profile.sample_replies[:top_k]

        try:
            sample_texts = [s.get("text", "") for s in profile.sample_replies if s.get("text")]
            if not sample_texts:
                return []

            query_vec = embedder.encode([query_text], normalize_embeddings=True)
            sample_vecs = embedder.encode(sample_texts, normalize_embeddings=True)

            similarities = np.dot(sample_vecs, query_vec.T).flatten()
            ranked_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in ranked_indices:
                sample_item = dict(profile.sample_replies[idx])
                sample_item["similarity"] = round(float(similarities[idx]), 3)
                results.append(sample_item)
            return results

        except Exception as e:
            logger.debug(f"Semantic similarity retrieval failed: {e}. Falling back to default samples.")
            return profile.sample_replies[:top_k]

    def generate_mock_profile(self, creator_name: str = "Alex Creator") -> CreatorPersonaProfile:
        """Generates a rich, realistic synthetic creator persona for demos and testing."""
        mock_replies = [
            {"index": 0, "text": "Appreciate you watching! Absolutely spot on regarding the architecture tradeoff. We'll be covering this more in part 2! 🔥", "word_count": 22},
            {"index": 1, "text": "Great question! The threshold is defined in settings.yaml under the respective stage parameters. Thanks for digging into the details! 🙏", "word_count": 21},
            {"index": 2, "text": "Fair pushback! I definitely hear your perspective on this. My goal was simply to share the raw facts, but thanks for keeping the debate honest.", "word_count": 27},
            {"index": 3, "text": "Sending massive love right back! Comments like yours genuinely make all the research and late nights worthwhile. ✨", "word_count": 18},
            {"index": 4, "text": "Pinning this right to the top so everyone can see it. Thanks so much for adding this context! 📌", "word_count": 19},
        ]
        return CreatorPersonaProfile(
            creator_channel_id="UC_CREATOR_SYNTHETIC",
            creator_name=creator_name,
            total_creator_comments=85,
            avg_word_count=21.4,
            avg_sentence_count=2.1,
            vocab_entropy=6.82,
            type_token_ratio=0.524,
            caps_ratio=0.038,
            punctuation_intensity=1.42,
            exclamation_rate=1.12,
            question_rate=0.25,
            ellipsis_rate=0.08,
            emoji_frequency=1.35,
            top_emojis=["🔥", "🙏", "✨", "📌", "❤️"],
            greeting_patterns=["Hey everyone!", "Appreciate you watching!", "Thanks for tuning in!"],
            signoff_patterns=["Appreciate you!", "Cheers!", "Stay tuned!"],
            signature_phrases=["spot on", "great question", "fair pushback", "late nights"],
            avg_sentiment=0.58,
            avg_toxicity=0.005,
            sample_replies=mock_replies,
        )


# ==============================================================================
# 3. LORA DATASET BUILDER & TUNING PIPELINE
# ==============================================================================

class LoRADatasetBuilder:
    """Extracts paired (viewer, creator) conversation trees and compiles LoRA fine-tuning datasets."""

    def __init__(
        self,
        comments_path: Union[str, Path] = "data/interim/comments_clean.parquet",
        videos_path: Union[str, Path] = "data/interim/videos_clean.parquet",
    ) -> None:
        self.comments_path = Path(comments_path)
        self.videos_path = Path(videos_path)

    def extract_conversation_pairs(
        self,
        creator_channel_id: Optional[str] = None,
        creator_name: Optional[str] = None,
        mock: bool = False,
        limit: int = 500,
    ) -> List[ConversationPair]:
        """Traverses the comment tree to extract direct (viewer_comment -> creator_reply) dialogue pairs."""
        if mock or not self.comments_path.exists():
            return self.generate_mock_pairs(limit=min(limit, 10), creator_name=creator_name or "Alex Creator")

        try:
            df_c = pd.read_parquet(self.comments_path)
            if df_c.empty or "parent_id" not in df_c.columns or "comment_id" not in df_c.columns:
                return self.generate_mock_pairs(limit=min(limit, 10), creator_name=creator_name or "Alex Creator")

            # Identify creator channel ID
            target_cid = creator_channel_id
            if not target_cid and self.videos_path.exists():
                try:
                    df_v = pd.read_parquet(self.videos_path, columns=["channel_id"])
                    if not df_v.empty and "channel_id" in df_v.columns:
                        target_cid = str(df_v.iloc[0]["channel_id"])
                except Exception:
                    pass

            # Filter creator replies
            creator_mask = (df_c["is_reply"] == 1) & (df_c["parent_id"].notna())
            if target_cid and "author_channel_id" in df_c.columns:
                creator_mask &= (df_c["author_channel_id"] == target_cid)
            elif creator_name and "author_display_name" in df_c.columns:
                creator_mask &= df_c["author_display_name"].str.contains(creator_name, case=False, na=False)

            df_replies = df_c[creator_mask]
            if df_replies.empty:
                logger.info("No creator replies matching criteria found. Falling back to synthetic training pairs.")
                return self.generate_mock_pairs(limit=min(limit, 10), creator_name=creator_name or "Alex Creator")

            # Map parent IDs to parent comment texts
            comment_map = dict(zip(df_c["comment_id"], df_c["text"]))
            author_map = dict(zip(df_c["comment_id"], df_c.get("author_display_name", df_c["comment_id"])))

            pairs: List[ConversationPair] = []
            for _, r in df_replies.iterrows():
                parent_id = str(r.get("parent_id", ""))
                parent_text = comment_map.get(parent_id, "")
                if not parent_text or len(parent_text.strip()) < 5:
                    continue

                pair = ConversationPair(
                    video_id=str(r.get("video_id", "VID_UNKNOWN")),
                    thread_root_id=parent_id,
                    viewer_comment_id=parent_id,
                    viewer_author=str(author_map.get(parent_id, "Viewer")),
                    viewer_text=str(parent_text),
                    creator_comment_id=str(r.get("comment_id", "")),
                    creator_reply_text=str(r.get("text", "")),
                    likes_on_reply=int(r.get("like_count", 0)),
                )
                pairs.append(pair)
                if len(pairs) >= limit:
                    break

            return pairs or self.generate_mock_pairs(limit=min(limit, 10), creator_name=creator_name or "Alex Creator")

        except Exception as e:
            logger.warning(f"Failed to extract conversation pairs: {e}. Generating mock pairs.")
            return self.generate_mock_pairs(limit=min(limit, 10), creator_name=creator_name or "Alex Creator")

    def export_lora_dataset(
        self,
        pairs: List[ConversationPair],
        output_dir: Union[str, Path] = "data/output/lora",
        creator_name: str = "Creator",
    ) -> Dict[str, str]:
        """Exports paired dialogues to Alpaca JSONL, ChatML JSONL, and an Ollama Modelfile."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        alpaca_path = out_dir / "creator_tuning_alpaca.jsonl"
        chatml_path = out_dir / "creator_tuning_chatml.jsonl"
        modelfile_path = out_dir / "Modelfile"

        # 1. Alpaca Format
        with open(alpaca_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_alpaca(), ensure_ascii=False) + "\n")

        # 2. ChatML Format
        with open(chatml_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_chatml(creator_name=creator_name), ensure_ascii=False) + "\n")

        # 3. Ollama Modelfile definition
        modelfile_content = (
            f"# Ollama Modelfile: Creator Persona ({creator_name})\n"
            f"FROM llama3.2\n\n"
            f"PARAMETER temperature 0.4\n"
            f"PARAMETER top_p 0.9\n"
            f"PARAMETER stop \"<|eot_id|>\"\n\n"
            f"SYSTEM \"\"\"\n"
            f"You are {creator_name}, the creator of this YouTube channel. "
            f"Respond directly to viewer comments in your authentic, warm, and engaging voice. "
            f"Use natural community catchphrases and signature emojis sparingly where appropriate.\n"
            f"\"\"\"\n"
        )
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

        logger.info(f"Exported {len(pairs)} dialogue pairs to {out_dir}")
        return {
            "alpaca": str(alpaca_path),
            "chatml": str(chatml_path),
            "modelfile": str(modelfile_path),
            "total_pairs": str(len(pairs)),
        }

    def generate_training_script(
        self,
        output_dir: Union[str, Path] = "data/output/lora",
        base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    ) -> str:
        """Generates a standalone, executable Python PEFT/LoRA training script."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = out_dir / "train_lora_peft.py"

        script_content = f'''"""Standalone PEFT/LoRA Fine-Tuning Script for Creator Persona.
Generated automatically by ytint.persona.LoRADatasetBuilder.
"""

import sys
import torch
from pathlib import Path

def train():
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
    except ImportError:
        print("Required training libraries missing. Run: pip install peft transformers trl datasets accelerate")
        sys.exit(1)

    dataset_file = Path(__file__).parent / "creator_tuning_alpaca.jsonl"
    if not dataset_file.exists():
        print(f"Dataset not found: {{dataset_file}}")
        sys.exit(1)

    print(f"Loading base model: {base_model}")
    model_id = "{base_model}"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(dataset_file), split="train")

    def format_prompts(batch):
        formatted = []
        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            text = f"### Instruction:\\n{{inst}}\\n\\n### Input:\\n{{inp}}\\n\\n### Response:\\n{{out}}"
            formatted.append(text)
        return formatted

    training_args = TrainingArguments(
        output_dir=str(Path(__file__).parent / "adapter_weights"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=format_prompts,
        max_seq_length=512,
    )

    print("Starting Creator Persona LoRA Fine-Tuning...")
    trainer.train()
    adapter_path = Path(__file__).parent / "creator_persona_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA fine-tuning complete! Adapter saved to {{adapter_path}}")

if __name__ == "__main__":
    train()
'''
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        logger.info(f"Generated standalone LoRA training script: {script_path}")
        return str(script_path)

    def generate_mock_pairs(self, limit: int = 10, creator_name: str = "Alex Creator") -> List[ConversationPair]:
        """Generates realistic synthetic (viewer, creator) conversation pairs."""
        mock_data = [
            ("The explanation of the causal uplift at 14:20 was brilliant. Best breakdown yet.",
             f"Appreciate you watching! Absolutely spot on regarding the causal uplift. We spent weeks calibrating those parameters! 🔥", 45),
            ("Is the dataset available for public research download or is it proprietary?",
             f"Great question! Yes, all parquet interim tables are accessible in the repo under data/interim/ with sample schemas. Appreciate you asking! 🙏", 28),
            ("I felt the tone was a bit dismissive of the alternative approach in section 3.",
             f"Fair pushback! I definitely hear your perspective on this. My goal was simply to share the raw benchmark numbers, but I respect that view.", 14),
            ("Been subscribed since day one, the production value on these releases keeps skyrocketing!",
             f"Sending massive love right back! Comments like yours genuinely make all the late nights and editing marathons worthwhile. ✨", 39),
            ("Can you make a dedicated follow-up on the network bow-tie decomposition?",
             f"You read my mind! That's already on the production board for next month. Stay tuned! 🚀", 31),
            ("At 8:35, did you normalize by total views or comments per hour?",
             f"Good catch! We normalized by comments per 1k views to keep category benchmarks consistent across video lengths.", 19),
            ("I don't think this method works for smaller channels with under 100 comments.",
             f"Totally valid point! For smaller sample sizes, the Poisson burst detector does require widening the time bins. Thanks for pointing that out.", 22),
            ("The emotion wheel visualization looks incredible. What library was used?",
             f"Thank you! That's rendered via matplotlib with custom polar coordinates in s99_visualize.py. Glad you like the aesthetic! 🎨", 18),
        ]

        pairs = []
        for i, (viewer_text, creator_reply, likes) in enumerate(mock_data[:limit]):
            pairs.append(
                ConversationPair(
                    video_id=f"MOCK_VID_{i % 3 + 1:03d}",
                    thread_root_id=f"mock_root_{i}",
                    viewer_comment_id=f"mock_viewer_{i}",
                    viewer_author=f"Viewer_{i+1}",
                    viewer_text=viewer_text,
                    creator_comment_id=f"mock_creator_reply_{i}",
                    creator_reply_text=creator_reply,
                    likes_on_reply=likes,
                    reply_latency_minutes=12.5 * (i + 1),
                )
            )
        return pairs
