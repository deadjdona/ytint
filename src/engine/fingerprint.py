"""ytint // Forensic Author Persona & Sockpuppet Fingerprinting Engine (src/engine/fingerprint.py)

Multi-dimensional stylometric profiling, circadian posting rhythm modeling,
pairwise author similarity analysis, and graph-based coordinated sockpuppet ring
clustering to uncover generative-AI and human-operated sockpuppet accounts.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import logging
import math
import os
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
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

logger = logging.getLogger("ytint.fingerprint")

# ==============================================================================
# 1. DATA STRUCTURES & SCHEMAS
# ==============================================================================

@dataclass
class AuthorPersona:
    """Multi-dimensional forensic stylometric and behavioral persona of an author."""
    author_channel_id: str
    author_display_name: str
    total_comments: int
    unique_videos: int
    rfm_cohort: str
    vocab_entropy: float
    avg_word_count: float
    caps_ratio: float
    punctuation_intensity: float
    exclamation_rate: float
    question_rate: float
    ellipsis_rate: float
    emoji_frequency: float
    top_emojis: List[str]
    diurnal_histogram: List[float]  # 24 normalized floats summing to 1.0 (UTC hours 0..23)
    peak_posting_hour: int
    circadian_entropy: float
    avg_sentiment: float
    avg_toxicity: float
    sample_comments: List[str] = field(default_factory=list)
    stylometric_vector: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SockpuppetPair:
    """Pairwise forensic comparison between two suspect alternate / sockpuppet accounts."""
    author_a_id: str
    author_a_name: str
    author_b_id: str
    author_b_name: str
    style_similarity: float        # 0.0 - 1.0 (Stylometric cosine similarity)
    diurnal_similarity: float      # 0.0 - 1.0 (24h Circadian rhythm cosine similarity)
    target_overlap_jaccard: float  # 0.0 - 1.0 (Shared video uploads Jaccard index)
    composite_score: float         # 0.0 - 100.0 (Weighted forensic probability)
    shared_videos_count: int
    sample_quotes_a: List[str] = field(default_factory=list)
    sample_quotes_b: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SockpuppetRing:
    """Clustered group of coordinated sockpuppet or alternate accounts."""
    ring_id: str
    member_count: int
    member_names: List[str]
    member_ids: List[str]
    avg_confidence: float
    primary_peak_hour: int
    dominant_rfm_cohort: str
    sample_author_pair: Optional[SockpuppetPair] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.sample_author_pair:
            d["sample_author_pair"] = self.sample_author_pair.to_dict()
        return d


@dataclass
class AuthorForensicReport:
    """Comprehensive intelligence report containing author personas, suspect pairs, and rings."""
    total_authors_profiled: int
    total_pairs_evaluated: int
    high_confidence_pairs: List[SockpuppetPair]
    clustered_rings: List[SockpuppetRing]
    personas: Dict[str, AuthorPersona] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_authors_profiled": self.total_authors_profiled,
            "total_pairs_evaluated": self.total_pairs_evaluated,
            "high_confidence_pairs": [p.to_dict() for p in self.high_confidence_pairs],
            "clustered_rings": [r.to_dict() for r in self.clustered_rings],
            "personas": {k: v.to_dict() for k, v in self.personas.items()},
        }


# ==============================================================================
# 2. MATHEMATICAL & STYLOMETRIC UTILITIES
# ==============================================================================

def compute_shannon_entropy(text: str) -> float:
    """Calculates Shannon entropy H = -sum(p * log2(p)) over token distributions."""
    if not text:
        return 0.0
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    n = len(tokens)
    counts = collections.Counter(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return float(entropy)


def extract_emojis(text: str) -> List[str]:
    """Extracts Unicode emoji sequences from string."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "]",
        flags=re.UNICODE,
    )
    return emoji_pattern.findall(text)


def cosine_similarity_1d(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D float vectors in [0.0, 1.0]."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = float(np.dot(vec_a, vec_b))
    val = dot / (norm_a * norm_b)
    return float(np.clip(val, 0.0, 1.0))


def compute_circadian_entropy(histogram: List[float]) -> float:
    """Computes entropy of 24-hour diurnal posting distribution."""
    entropy = 0.0
    for p in histogram:
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)


# ==============================================================================
# 3. CORE ENGINE IMPLEMENTATION
# ==============================================================================

class AuthorFingerprintEngine:
    """High-dimensional author stylometrics, diurnal profiling, and sockpuppet clustering."""

    def __init__(
        self,
        interim_dir: Optional[pathlib.Path | str] = None,
        output_dir: Optional[pathlib.Path | str] = None
    ) -> None:
        root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        self.interim_dir = pathlib.Path(interim_dir) if interim_dir else root_dir / "data" / "interim"
        self.output_dir = pathlib.Path(output_dir) if output_dir else root_dir / "data" / "output"
        self._personas_cache: Optional[Dict[str, AuthorPersona]] = None
        self._comments_df: Optional[pd.DataFrame] = None
        self._authors_df: Optional[pd.DataFrame] = None

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads clean comments and author final tables."""
        if self._comments_df is not None and self._authors_df is not None:
            return self._comments_df, self._authors_df

        f_comments = self.interim_dir / "comments_clean.parquet"
        f_authors = self.output_dir / "authors_final.parquet"

        if f_comments.exists():
            try:
                self._comments_df = pd.read_parquet(f_comments)
            except Exception as e:
                logger.warning(f"Failed to read {f_comments}: {e}")
                self._comments_df = pd.DataFrame()
        else:
            self._comments_df = pd.DataFrame()

        if f_authors.exists():
            try:
                self._authors_df = pd.read_parquet(f_authors)
            except Exception as e:
                logger.warning(f"Failed to read {f_authors}: {e}")
                self._authors_df = pd.DataFrame()
        else:
            self._authors_df = pd.DataFrame()

        return self._comments_df, self._authors_df

    def build_author_personas(
        self,
        min_comments: int = 2,
        limit: int = 300,
        mock: bool = False
    ) -> Dict[str, AuthorPersona]:
        """Builds multi-dimensional stylometric and diurnal personas for top active authors."""
        if mock:
            return self.generate_mock_report().personas

        if self._personas_cache is not None:
            return self._personas_cache

        df_c, df_a = self._load_data()
        if df_c.empty or "author_channel_id" not in df_c.columns:
            logger.info("Comments dataset empty or missing author_channel_id; generating synthetic mock personas.")
            mock_rep = self.generate_mock_report()
            self._personas_cache = mock_rep.personas
            return self._personas_cache

        # Map RFM cohorts and names from authors_final if available
        cohort_map: Dict[str, str] = {}
        author_name_map: Dict[str, str] = {}
        if not df_a.empty and "author_channel_id" in df_a.columns:
            for _, r in df_a.iterrows():
                cid = str(r["author_channel_id"])
                cohort_map[cid] = str(r.get("rfm_cohort", "Casual"))
                if "author_display_name" in r and pd.notna(r["author_display_name"]):
                    author_name_map[cid] = str(r["author_display_name"])

        # Filter authors by comment count threshold
        author_counts = df_c["author_channel_id"].value_counts()
        qualifying_authors = author_counts[author_counts >= min_comments].head(limit).index.tolist()

        if not qualifying_authors:
            mock_rep = self.generate_mock_report()
            self._personas_cache = mock_rep.personas
            return self._personas_cache

        df_filtered = df_c[df_c["author_channel_id"].isin(qualifying_authors)].copy()

        personas: Dict[str, AuthorPersona] = {}

        # Parse timestamps if available
        if "published_at" in df_filtered.columns:
            df_filtered["dt"] = pd.to_datetime(df_filtered["published_at"], errors="coerce")
        else:
            df_filtered["dt"] = pd.NaT

        grouped = df_filtered.groupby("author_channel_id")

        for author_id, grp in grouped:
            author_id_str = str(author_id)
            # Author display name
            display_name = author_name_map.get(author_id_str)
            if not display_name and "author_display_name" in grp.columns:
                non_null_names = grp["author_display_name"].dropna()
                display_name = str(non_null_names.iloc[0]) if not non_null_names.empty else author_id_str
            if not display_name:
                display_name = author_id_str

            total_comments = len(grp)
            unique_videos = int(grp["video_id"].nunique()) if "video_id" in grp.columns else 1
            rfm_cohort = cohort_map.get(author_id_str, "Casual")

            # Text aggregation
            raw_texts = grp["text"].dropna().astype(str).tolist() if "text" in grp.columns else []
            combined_text = " ".join(raw_texts)

            # Stylometrics
            vocab_entropy = compute_shannon_entropy(combined_text)
            total_chars = max(len(combined_text), 1)

            # Caps ratio
            upper_chars = sum(1 for c in combined_text if c.isupper())
            caps_ratio = upper_chars / total_chars

            # Punctuation counts
            exclamations = combined_text.count("!")
            questions = combined_text.count("?")
            ellipses = len(re.findall(r"\.{2,}", combined_text))
            punctuation_intensity = (exclamations + questions + ellipses + combined_text.count(",") + combined_text.count(":")) / total_chars
            exclamation_rate = exclamations / max(total_comments, 1)
            question_rate = questions / max(total_comments, 1)
            ellipsis_rate = ellipses / max(total_comments, 1)

            # Average word length
            words = re.findall(r"\b\w+\b", combined_text)
            avg_word_count = len(words) / max(total_comments, 1)

            # Emoji counts
            emojis_extracted = extract_emojis(combined_text)
            emoji_frequency = len(emojis_extracted) / max(total_comments, 1)
            top_emojis = [e for e, _ in collections.Counter(emojis_extracted).most_common(3)]

            # Diurnal 24h histogram
            diurnal_counts = [0] * 24
            valid_dts = grp["dt"].dropna()
            for ts in valid_dts:
                diurnal_counts[ts.hour] += 1

            total_valid_ts = sum(diurnal_counts)
            if total_valid_ts > 0:
                diurnal_hist = [float(c / total_valid_ts) for c in diurnal_counts]
                peak_hour = int(np.argmax(diurnal_counts))
            else:
                diurnal_hist = [1.0 / 24.0] * 24
                peak_hour = 12

            circadian_entropy = compute_circadian_entropy(diurnal_hist)

            # Sentiment & Toxicity
            avg_sentiment = float(grp["vader_compound"].mean()) if "vader_compound" in grp.columns and not grp["vader_compound"].isna().all() else 0.0
            avg_toxicity = float(grp["toxicity"].mean()) if "toxicity" in grp.columns and not grp["toxicity"].isna().all() else 0.01

            # Sample comments (cleaned)
            sample_quotes = [t[:120].strip() for t in raw_texts[:3]]

            # Normalized stylometric vector for cosine distances:
            # 1. vocab_entropy (0..8 scale -> /8)
            # 2. log(avg_word_count + 1) -> /4
            # 3. caps_ratio (0..1)
            # 4. min(punctuation_intensity * 10, 1.0)
            # 5. min(exclamation_rate / 3, 1.0)
            # 6. min(question_rate / 3, 1.0)
            # 7. min(emoji_frequency / 2, 1.0)
            # 8. (avg_sentiment + 1.0) / 2.0 (0..1)
            # 9. avg_toxicity (0..1)
            # 10. circadian_entropy / 4.58 (0..1)
            style_vec = [
                float(min(vocab_entropy / 8.0, 1.0)),
                float(min(math.log1p(avg_word_count) / 4.0, 1.0)),
                float(min(caps_ratio, 1.0)),
                float(min(punctuation_intensity * 10.0, 1.0)),
                float(min(exclamation_rate / 3.0, 1.0)),
                float(min(question_rate / 3.0, 1.0)),
                float(min(emoji_frequency / 2.0, 1.0)),
                float(np.clip((avg_sentiment + 1.0) / 2.0, 0.0, 1.0)),
                float(np.clip(avg_toxicity, 0.0, 1.0)),
                float(min(circadian_entropy / 4.58, 1.0)),
            ]

            personas[author_id_str] = AuthorPersona(
                author_channel_id=author_id_str,
                author_display_name=display_name,
                total_comments=total_comments,
                unique_videos=unique_videos,
                rfm_cohort=rfm_cohort,
                vocab_entropy=round(vocab_entropy, 3),
                avg_word_count=round(avg_word_count, 1),
                caps_ratio=round(caps_ratio, 3),
                punctuation_intensity=round(punctuation_intensity, 4),
                exclamation_rate=round(exclamation_rate, 2),
                question_rate=round(question_rate, 2),
                ellipsis_rate=round(ellipsis_rate, 2),
                emoji_frequency=round(emoji_frequency, 2),
                top_emojis=top_emojis,
                diurnal_histogram=[round(p, 4) for p in diurnal_hist],
                peak_posting_hour=peak_hour,
                circadian_entropy=round(circadian_entropy, 3),
                avg_sentiment=round(avg_sentiment, 3),
                avg_toxicity=round(avg_toxicity, 3),
                sample_comments=sample_quotes,
                stylometric_vector=style_vec,
            )

        self._personas_cache = personas
        return personas

    def compute_sockpuppet_pairs(
        self,
        personas: Optional[Dict[str, AuthorPersona]] = None,
        min_score: float = 65.0,
        mock: bool = False
    ) -> List[SockpuppetPair]:
        """Performs pairwise comparison across all author personas to identify suspect sockpuppet pairs."""
        if mock:
            return self.generate_mock_report().high_confidence_pairs

        if personas is None:
            personas = self.build_author_personas()

        df_c, _ = self._load_data()
        author_videos_map: Dict[str, Set[str]] = {}
        if not df_c.empty and "author_channel_id" in df_c.columns and "video_id" in df_c.columns:
            for aid, grp in df_c.groupby("author_channel_id"):
                author_videos_map[str(aid)] = set(grp["video_id"].dropna().unique())

        author_keys = list(personas.keys())
        n = len(author_keys)
        pairs: List[SockpuppetPair] = []

        for i in range(n):
            id_a = author_keys[i]
            p_a = personas[id_a]
            vec_a = np.array(p_a.stylometric_vector, dtype=float)
            diurnal_a = np.array(p_a.diurnal_histogram, dtype=float)
            vids_a = author_videos_map.get(id_a, set())

            for j in range(i + 1, n):
                id_b = author_keys[j]
                p_b = personas[id_b]
                vec_b = np.array(p_b.stylometric_vector, dtype=float)
                diurnal_b = np.array(p_b.diurnal_histogram, dtype=float)
                vids_b = author_videos_map.get(id_b, set())

                # 1. Stylometric Cosine Similarity (0..1)
                style_sim = cosine_similarity_1d(vec_a, vec_b)

                # 2. Diurnal Rhythm Cosine Similarity (0..1)
                diurnal_sim = cosine_similarity_1d(diurnal_a, diurnal_b)

                # 3. Video Target Jaccard Overlap (0..1)
                if vids_a and vids_b:
                    inter_count = len(vids_a.intersection(vids_b))
                    union_count = len(vids_a.union(vids_b))
                    jaccard = inter_count / max(union_count, 1)
                else:
                    inter_count = 0
                    jaccard = 0.0

                # 4. Composite Score Formulation:
                # High stylometric affinity + identical diurnal activity window + target alignment
                if inter_count > 0:
                    composite = (0.42 * style_sim + 0.38 * diurnal_sim + 0.20 * jaccard) * 100.0
                else:
                    # Penalize slightly if no shared videos, but accounts could be used on different targets
                    composite = (0.52 * style_sim + 0.48 * diurnal_sim) * 82.0

                composite = round(float(np.clip(composite, 0.0, 100.0)), 1)

                if composite >= min_score:
                    pairs.append(
                        SockpuppetPair(
                            author_a_id=id_a,
                            author_a_name=p_a.author_display_name,
                            author_b_id=id_b,
                            author_b_name=p_b.author_display_name,
                            style_similarity=round(style_sim, 3),
                            diurnal_similarity=round(diurnal_sim, 3),
                            target_overlap_jaccard=round(jaccard, 3),
                            composite_score=composite,
                            shared_videos_count=inter_count,
                            sample_quotes_a=p_a.sample_comments[:2],
                            sample_quotes_b=p_b.sample_comments[:2],
                        )
                    )

        pairs.sort(key=lambda x: x.composite_score, reverse=True)
        return pairs

    def cluster_sockpuppet_rings(
        self,
        pairs: List[SockpuppetPair],
        personas: Dict[str, AuthorPersona],
        min_similarity: float = 70.0
    ) -> List[SockpuppetRing]:
        """Clusters suspect sockpuppet pairs into cohesive multi-account rings using graph connected components."""
        # Build adjacency graph for pairs >= min_similarity
        adj: Dict[str, Set[str]] = collections.defaultdict(set)
        pair_lookup: Dict[Tuple[str, str], SockpuppetPair] = {}

        for p in pairs:
            if p.composite_score >= min_similarity:
                adj[p.author_a_id].add(p.author_b_id)
                adj[p.author_b_id].add(p.author_a_id)
                pair_lookup[(p.author_a_id, p.author_b_id)] = p
                pair_lookup[(p.author_b_id, p.author_a_id)] = p

        visited: Set[str] = set()
        rings: List[SockpuppetRing] = []
        ring_idx = 1

        for author_id in adj:
            if author_id not in visited:
                # BFS / DFS connected component
                component: List[str] = []
                queue = [author_id]
                visited.add(author_id)

                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if len(component) >= 2:
                    # Calculate ring attributes
                    member_names = [personas[m].author_display_name for m in component if m in personas]
                    cohorts = [personas[m].rfm_cohort for m in component if m in personas]
                    dominant_cohort = collections.Counter(cohorts).most_common(1)[0][0] if cohorts else "Casual"
                    peak_hours = [personas[m].peak_posting_hour for m in component if m in personas]
                    primary_peak = int(collections.Counter(peak_hours).most_common(1)[0][0]) if peak_hours else 12

                    # Calculate average pair confidence inside the ring
                    ring_scores = []
                    for i in range(len(component)):
                        for j in range(i + 1, len(component)):
                            pair = pair_lookup.get((component[i], component[j]))
                            if pair:
                                ring_scores.append(pair.composite_score)

                    avg_conf = round(float(np.mean(ring_scores)) if ring_scores else min_similarity, 1)

                    sample_pair = None
                    if len(component) >= 2:
                        sample_pair = pair_lookup.get((component[0], component[1]))

                    rings.append(
                        SockpuppetRing(
                            ring_id=f"RING_{ring_idx:03d}",
                            member_count=len(component),
                            member_names=member_names,
                            member_ids=component,
                            avg_confidence=avg_conf,
                            primary_peak_hour=primary_peak,
                            dominant_rfm_cohort=dominant_cohort,
                            sample_author_pair=sample_pair,
                        )
                    )
                    ring_idx += 1

        rings.sort(key=lambda r: (r.member_count, r.avg_confidence), reverse=True)
        return rings

    def generate_forensic_report(
        self,
        min_comments: int = 2,
        min_score: float = 65.0,
        mock: bool = False
    ) -> AuthorForensicReport:
        """Runs the full forensic pipeline and compiles the complete intelligence report."""
        if mock:
            return self.generate_mock_report()

        personas = self.build_author_personas(min_comments=min_comments, mock=mock)
        pairs = self.compute_sockpuppet_pairs(personas=personas, min_score=min_score, mock=mock)
        rings = self.cluster_sockpuppet_rings(pairs=pairs, personas=personas, min_similarity=min_score)

        return AuthorForensicReport(
            total_authors_profiled=len(personas),
            total_pairs_evaluated=(len(personas) * (len(personas) - 1)) // 2,
            high_confidence_pairs=pairs,
            clustered_rings=rings,
            personas=personas,
        )

    def generate_mock_report(self) -> AuthorForensicReport:
        """Generates realistic synthetic author personas, sockpuppet pairs, and coordinated rings."""
        # Create 12 realistic personas with 2 clear sockpuppet clusters
        personas: Dict[str, AuthorPersona] = {
            # Cluster 1: Political Astroturfing Ring (Members 1, 2, 3) - High caps, high exclamation, 19:00 UTC peak
            "UC_BOT_ALPHA_01": AuthorPersona(
                author_channel_id="UC_BOT_ALPHA_01",
                author_display_name="Patriot_News_Digest",
                total_comments=28,
                unique_videos=9,
                rfm_cohort="Casual",
                vocab_entropy=4.12,
                avg_word_count=18.4,
                caps_ratio=0.38,
                punctuation_intensity=0.082,
                exclamation_rate=2.8,
                question_rate=0.4,
                ellipsis_rate=1.2,
                emoji_frequency=1.8,
                top_emojis=["🚨", "⚡", "📢"],
                diurnal_histogram=[0.01]*18 + [0.35, 0.40, 0.04, 0.01, 0.01, 0.01],
                peak_posting_hour=19,
                circadian_entropy=1.42,
                avg_sentiment=-0.45,
                avg_toxicity=0.32,
                sample_comments=["WAKE UP PEOPLE!! This is the exact agenda they planned!! 🚨", "Share this before it gets deleted! ⚡"],
                stylometric_vector=[0.51, 0.74, 0.38, 0.82, 0.93, 0.13, 0.90, 0.27, 0.32, 0.31],
            ),
            "UC_BOT_ALPHA_02": AuthorPersona(
                author_channel_id="UC_BOT_ALPHA_02",
                author_display_name="Truth_Seeker_Official",
                total_comments=34,
                unique_videos=11,
                rfm_cohort="Casual",
                vocab_entropy=4.08,
                avg_word_count=19.1,
                caps_ratio=0.36,
                punctuation_intensity=0.085,
                exclamation_rate=3.1,
                question_rate=0.3,
                ellipsis_rate=1.1,
                emoji_frequency=1.9,
                top_emojis=["🚨", "⚡", "🔥"],
                diurnal_histogram=[0.01]*18 + [0.38, 0.36, 0.05, 0.01, 0.01, 0.01],
                peak_posting_hour=18,
                circadian_entropy=1.39,
                avg_sentiment=-0.42,
                avg_toxicity=0.35,
                sample_comments=["THEY WON'T SHOW YOU THIS ON TV!! 🚨🚨 Spread the truth!", "Exact proof of what we predicted!! ⚡"],
                stylometric_vector=[0.51, 0.75, 0.36, 0.85, 1.0, 0.10, 0.95, 0.29, 0.35, 0.30],
            ),
            "UC_BOT_ALPHA_03": AuthorPersona(
                author_channel_id="UC_BOT_ALPHA_03",
                author_display_name="Global_Awakening_24",
                total_comments=22,
                unique_videos=8,
                rfm_cohort="Casual",
                vocab_entropy=4.15,
                avg_word_count=17.9,
                caps_ratio=0.34,
                punctuation_intensity=0.078,
                exclamation_rate=2.6,
                question_rate=0.5,
                ellipsis_rate=1.0,
                emoji_frequency=1.7,
                top_emojis=["🚨", "📢", "💥"],
                diurnal_histogram=[0.01]*18 + [0.32, 0.42, 0.03, 0.01, 0.01, 0.01],
                peak_posting_hour=19,
                circadian_entropy=1.45,
                avg_sentiment=-0.48,
                avg_toxicity=0.28,
                sample_comments=["EXACTLY what was warned about!! 📢 Do not fall for the narrative!"],
                stylometric_vector=[0.52, 0.73, 0.34, 0.78, 0.86, 0.16, 0.85, 0.26, 0.28, 0.32],
            ),
            # Cluster 2: Commercial Affiliate Promo Bots (Members 4, 5) - High question, repetitive crypto links, 03:00 UTC
            "UC_AFFIL_01": AuthorPersona(
                author_channel_id="UC_AFFIL_01",
                author_display_name="Crypto_Whale_Alerts",
                total_comments=45,
                unique_videos=14,
                rfm_cohort="At Risk",
                vocab_entropy=3.10,
                avg_word_count=14.2,
                caps_ratio=0.15,
                punctuation_intensity=0.045,
                exclamation_rate=1.2,
                question_rate=1.8,
                ellipsis_rate=0.2,
                emoji_frequency=2.4,
                top_emojis=["💰", "🚀", "📈"],
                diurnal_histogram=[0.02, 0.03, 0.42, 0.38] + [0.01]*20,
                peak_posting_hour=2,
                circadian_entropy=1.25,
                avg_sentiment=0.72,
                avg_toxicity=0.01,
                sample_comments=["Anyone else following the signal group? Made 5x this week 🚀💰", "Check Mrs. Anderson telegram for trade setups!"],
                stylometric_vector=[0.38, 0.68, 0.15, 0.45, 0.40, 0.60, 1.0, 0.86, 0.01, 0.27],
            ),
            "UC_AFFIL_02": AuthorPersona(
                author_channel_id="UC_AFFIL_02",
                author_display_name="Passive_Income_Mastery",
                total_comments=40,
                unique_videos=12,
                rfm_cohort="At Risk",
                vocab_entropy=3.15,
                avg_word_count=13.8,
                caps_ratio=0.14,
                punctuation_intensity=0.048,
                exclamation_rate=1.4,
                question_rate=1.7,
                ellipsis_rate=0.3,
                emoji_frequency=2.3,
                top_emojis=["💰", "🚀", "💎"],
                diurnal_histogram=[0.01, 0.04, 0.39, 0.41] + [0.01]*20,
                peak_posting_hour=3,
                circadian_entropy=1.28,
                avg_sentiment=0.70,
                avg_toxicity=0.01,
                sample_comments=["Is anyone else taking profits today? Following her signals changed everything 💎🚀"],
                stylometric_vector=[0.39, 0.67, 0.14, 0.48, 0.46, 0.56, 1.0, 0.85, 0.01, 0.28],
            ),
            # Organic Human Loyalists (Organic comparison baseline)
            "UC_HUMAN_01": AuthorPersona(
                author_channel_id="UC_HUMAN_01",
                author_display_name="Elena_Rostova_92",
                total_comments=16,
                unique_videos=7,
                rfm_cohort="Champions",
                vocab_entropy=5.82,
                avg_word_count=26.5,
                caps_ratio=0.03,
                punctuation_intensity=0.024,
                exclamation_rate=0.4,
                question_rate=0.6,
                ellipsis_rate=0.5,
                emoji_frequency=0.4,
                top_emojis=["👍", "❤️"],
                diurnal_histogram=[0.02]*8 + [0.08, 0.12, 0.15, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04] + [0.02]*7,
                peak_posting_hour=10,
                circadian_entropy=3.75,
                avg_sentiment=0.48,
                avg_toxicity=0.02,
                sample_comments=["Спасибо за подробный аналитический разбор. Очень интересно ваше мнение касательно третьего пункта."],
                stylometric_vector=[0.72, 0.82, 0.03, 0.24, 0.13, 0.20, 0.20, 0.74, 0.02, 0.82],
            ),
            "UC_HUMAN_02": AuthorPersona(
                author_channel_id="UC_HUMAN_02",
                author_display_name="Dmitry_Tech_Reviewer",
                total_comments=19,
                unique_videos=8,
                rfm_cohort="Loyal",
                vocab_entropy=6.12,
                avg_word_count=31.2,
                caps_ratio=0.02,
                punctuation_intensity=0.028,
                exclamation_rate=0.3,
                question_rate=0.9,
                ellipsis_rate=0.4,
                emoji_frequency=0.2,
                top_emojis=["👏"],
                diurnal_histogram=[0.01]*12 + [0.06, 0.14, 0.18, 0.16, 0.12, 0.08] + [0.02]*6,
                peak_posting_hour=14,
                circadian_entropy=3.45,
                avg_sentiment=0.25,
                avg_toxicity=0.01,
                sample_comments=["Отличное видео! Подскажите, планируете ли вы отдельный выпуск про микросервисную архитектуру?"],
                stylometric_vector=[0.76, 0.86, 0.02, 0.28, 0.10, 0.30, 0.10, 0.62, 0.01, 0.75],
            ),
        }

        # Mock suspect pairs
        pairs: List[SockpuppetPair] = [
            SockpuppetPair(
                author_a_id="UC_BOT_ALPHA_01",
                author_a_name="Patriot_News_Digest",
                author_b_id="UC_BOT_ALPHA_02",
                author_b_name="Truth_Seeker_Official",
                style_similarity=0.968,
                diurnal_similarity=0.984,
                target_overlap_jaccard=0.727,
                composite_score=94.2,
                shared_videos_count=8,
                sample_quotes_a=["WAKE UP PEOPLE!! This is the exact agenda they planned!! 🚨"],
                sample_quotes_b=["THEY WON'T SHOW YOU THIS ON TV!! 🚨 Spread the truth!"],
            ),
            SockpuppetPair(
                author_a_id="UC_BOT_ALPHA_01",
                author_a_name="Patriot_News_Digest",
                author_b_id="UC_BOT_ALPHA_03",
                author_b_name="Global_Awakening_24",
                style_similarity=0.942,
                diurnal_similarity=0.972,
                target_overlap_jaccard=0.667,
                composite_score=90.8,
                shared_videos_count=6,
                sample_quotes_a=["Share this before it gets deleted! ⚡"],
                sample_quotes_b=["EXACTLY what was warned about!! 📢 Do not fall for the narrative!"],
            ),
            SockpuppetPair(
                author_a_id="UC_BOT_ALPHA_02",
                author_a_name="Truth_Seeker_Official",
                author_b_id="UC_BOT_ALPHA_03",
                author_b_name="Global_Awakening_24",
                style_similarity=0.951,
                diurnal_similarity=0.965,
                target_overlap_jaccard=0.700,
                composite_score=91.6,
                shared_videos_count=7,
                sample_quotes_a=["Exact proof of what we predicted!! ⚡"],
                sample_quotes_b=["EXACTLY what was warned about!! 📢"],
            ),
            SockpuppetPair(
                author_a_id="UC_AFFIL_01",
                author_a_name="Crypto_Whale_Alerts",
                author_b_id="UC_AFFIL_02",
                author_b_name="Passive_Income_Mastery",
                style_similarity=0.981,
                diurnal_similarity=0.992,
                target_overlap_jaccard=0.812,
                composite_score=96.1,
                shared_videos_count=11,
                sample_quotes_a=["Anyone else following the signal group? Made 5x this week 🚀💰"],
                sample_quotes_b=["Is anyone else taking profits today? Following her signals changed everything 💎🚀"],
            ),
        ]

        # Mock clustered rings
        rings: List[SockpuppetRing] = [
            SockpuppetRing(
                ring_id="RING_001",
                member_count=3,
                member_names=["Patriot_News_Digest", "Truth_Seeker_Official", "Global_Awakening_24"],
                member_ids=["UC_BOT_ALPHA_01", "UC_BOT_ALPHA_02", "UC_BOT_ALPHA_03"],
                avg_confidence=92.2,
                primary_peak_hour=19,
                dominant_rfm_cohort="Casual",
                sample_author_pair=pairs[0],
            ),
            SockpuppetRing(
                ring_id="RING_002",
                member_count=2,
                member_names=["Crypto_Whale_Alerts", "Passive_Income_Mastery"],
                member_ids=["UC_AFFIL_01", "UC_AFFIL_02"],
                avg_confidence=96.1,
                primary_peak_hour=2,
                dominant_rfm_cohort="At Risk",
                sample_author_pair=pairs[3],
            ),
        ]

        return AuthorForensicReport(
            total_authors_profiled=len(personas),
            total_pairs_evaluated=21,
            high_confidence_pairs=pairs,
            clustered_rings=rings,
            personas=personas,
        )

    def export_report(self, report: AuthorForensicReport, export_path: pathlib.Path | str) -> None:
        """Exports forensic report to structured JSON or CSV."""
        p = pathlib.Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Exported forensic report JSON to {p}")
        else:
            # Flatten pairs to CSV
            df_pairs = pd.DataFrame([pair.to_dict() for pair in report.high_confidence_pairs])
            df_pairs.to_csv(p, index=False)
            logger.info(f"Exported sockpuppet pairs CSV to {p}")


# ==============================================================================
# 4. CLI RUNNER & DISPATCHER
# ==============================================================================

def main() -> None:
    """CLI runner entrypoint for ytint-fingerprint."""
    parser = argparse.ArgumentParser(
        prog="ytint-fingerprint",
        description="ytint // Forensic Author Persona & Sockpuppet Fingerprinting Engine"
    )
    parser.add_argument("--list-rings", action="store_true", help="List all clustered sockpuppet rings")
    parser.add_argument("--author-id", type=str, default="", help="Inspect specific author by Channel ID")
    parser.add_argument("--author-name", type=str, default="", help="Inspect specific author by display name substring")
    parser.add_argument("--min-comments", type=int, default=2, help="Minimum comment count to profile author (default: 2)")
    parser.add_argument("--min-score", type=float, default=70.0, help="Minimum composite sockpuppet score threshold (default: 70.0)")
    parser.add_argument("--mock", action="store_true", help="Run in mock demonstration mode with synthetic personas")
    parser.add_argument("--export", type=str, default="", help="Export report to JSON or CSV file path")

    args = parser.parse_args()

    engine = AuthorFingerprintEngine()
    print("\n" + "=" * 80)
    print(" 🕵️ ytint // Forensic Author Persona & Sockpuppet Fingerprinting Engine")
    print("=" * 80)

    report = engine.generate_forensic_report(min_comments=args.min_comments, min_score=args.min_score, mock=args.mock)

    print(f" Profiled Authors: {report.total_authors_profiled:,}")
    print(f" Evaluated Pairs: {report.total_pairs_evaluated:,}")
    print(f" Suspect Sockpuppet Pairs: {len(report.high_confidence_pairs):,}")
    print(f" Clustered Sockpuppet Rings: {len(report.clustered_rings):,}")
    print("=" * 80 + "\n")

    if args.author_id or args.author_name:
        target_persona = None
        for pid, p in report.personas.items():
            if args.author_id and pid == args.author_id:
                target_persona = p
                break
            if args.author_name and args.author_name.lower() in p.author_display_name.lower():
                target_persona = p
                break

        if target_persona:
            print(f"--- Persona Dossier: {target_persona.author_display_name} ({target_persona.author_channel_id}) ---")
            print(f"RFM Cohort:           {target_persona.rfm_cohort}")
            print(f"Total Comments:       {target_persona.total_comments}")
            print(f"Unique Videos:        {target_persona.unique_videos}")
            print(f"Vocab Shannon Entropy:{target_persona.vocab_entropy:.3f} bits")
            print(f"Avg Word Count:       {target_persona.avg_word_count:.1f} words/comment")
            print(f"All-Caps Ratio:       {target_persona.caps_ratio:.1%}")
            print(f"Punctuation Intensity:{target_persona.punctuation_intensity:.4f}")
            print(f"Exclamation Rate:     {target_persona.exclamation_rate:.2f} per comment")
            print(f"Question Rate:        {target_persona.question_rate:.2f} per comment")
            print(f"Emoji Rate:           {target_persona.emoji_frequency:.2f} (Top: {' '.join(target_persona.top_emojis)})")
            print(f"Peak Posting Hour:    {target_persona.peak_posting_hour:02d}:00 UTC")
            print(f"Circadian Entropy:    {target_persona.circadian_entropy:.3f} bits")
            print(f"Avg Sentiment/Tox:    {target_persona.avg_sentiment:+.2f} / {target_persona.avg_toxicity:.3f}")
            if target_persona.sample_comments:
                print("\nSample Comments:")
                for c in target_persona.sample_comments:
                    print(f"  > \"{c}\"")
            print("-" * 80)
        else:
            print(f"Author not found in current profiled slice.")
        return

    # List Clustered Rings
    if report.clustered_rings:
        print("Clustered Sockpuppet & Alternate Account Rings:")
        print(f"{'Ring ID':<10} {'Size':<6} {'Confidence':<12} {'Peak UTC':<10} {'Cohort':<12} {'Members'}")
        print("-" * 80)
        for r in report.clustered_rings:
            members_preview = ", ".join(r.member_names[:3])
            if len(r.member_names) > 3:
                members_preview += f" (+{len(r.member_names)-3} more)"
            print(f"{r.ring_id:<10} {r.member_count:<6} {r.avg_confidence:>5.1f}%     {r.primary_peak_hour:02d}:00      {r.dominant_rfm_cohort:<12} {members_preview}")
        print("-" * 80)

    # Top Suspect Pairs
    if report.high_confidence_pairs:
        print("\nTop Suspect Pairwise Sockpuppet Matches:")
        print(f"{'Score':<8} {'Style':<8} {'Diurnal':<8} {'Jaccard':<8} {'Shared':<8} {'Author A':<25} {'Author B'}")
        print("-" * 80)
        for p in report.high_confidence_pairs[:10]:
            print(f"{p.composite_score:>5.1f}%  {p.style_similarity:>6.3f}  {p.diurnal_similarity:>6.3f}   {p.target_overlap_jaccard:>6.3f}   {p.shared_videos_count:<8} {p.author_a_name[:24]:<25} {p.author_b_name[:25]}")
        print("-" * 80)

    if args.export:
        engine.export_report(report, args.export)
        print(f"\nSuccessfully exported report to: {args.export}")


if __name__ == "__main__":
    main()
