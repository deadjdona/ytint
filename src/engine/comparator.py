"""ytint // Multi-Channel & Playlist Competitive Intelligence Engine (src/engine/comparator.py)

Cross-channel benchmarking, temporal cohort comparison, Jaccard audience overlap,
sentiment & toxicity differentials, and shared commenter migration forensics.
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
from datetime import datetime, timezone
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

logger = logging.getLogger("ytint.comparator")

# Emoji extraction pattern
_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\U0001f300-\U0001f5ff\U0001f600-\U0001f64f\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff]"
)

# Standard radar categories
RADAR_DIMENSIONS = [
    "Discussion Volume",
    "Engagement Velocity",
    "Sentiment Positivity",
    "Community Safety",
    "Audience Loyalty",
    "Lexical Sophistication"
]

# Known channel display names for friendly UI
KNOWN_CHANNEL_NAMES = {
    "UCZdZPZof_1GO9jksCzUqAxw": "Primary Geopolitical News & OSINT (524 vids)",
    "UCj-XcmWS9wZW6w-GtxNnp3A": "Roman Zubenko Channel (161 vids)",
    "UC_MOCK_CHANNEL_12345": "Mock Synthetic Test Channel (2 vids)"
}


# ==============================================================================
# 1. DATA STRUCTURES & SCHEMAS
# ==============================================================================

@dataclass
class ChannelCohortProfile:
    """Holistic analytical and behavioral profile for a channel or video cohort."""
    cohort_id: str
    cohort_name: str
    cohort_type: str  # "channel", "quarter_cohort", "playlist_series", "benchmark"
    total_videos: int
    total_comments: int
    unique_authors: int
    comments_per_video: float
    avg_likes_per_comment: float
    reply_ratio: float
    sentiment_pos_ratio: float
    sentiment_neg_ratio: float
    avg_vader_compound: float
    mean_toxicity: float
    toxicity_r0: float  # Reply toxicity relative to root comment toxicity
    avg_author_loyalty: float  # Comments per unique author
    champion_author_ratio: float  # Authors with >= 5 comments
    vocab_entropy: float
    top_emojis: List[str]
    diurnal_peak_utc: int
    radar_scores: Dict[str, float]  # Normalized 0..100 across RADAR_DIMENSIONS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AudienceOverlapMetrics:
    """Pairwise audience overlap, Jaccard similarity, and commenter migration metrics."""
    cohort_a_id: str
    cohort_b_id: str
    cohort_a_name: str
    cohort_b_name: str
    authors_a: int
    authors_b: int
    shared_authors: int
    jaccard_similarity: float
    overlap_coefficient: float
    sentiment_differential: float  # Mean sentiment in B - Mean sentiment in A (for shared authors)
    toxicity_differential: float   # Mean toxicity in B - Mean toxicity in A (for shared authors)
    top_migrated_commenters: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparativeIntelligenceReport:
    """Consolidated multi-channel & cohort intelligence report."""
    profiles: Dict[str, ChannelCohortProfile]
    pairwise_overlaps: List[AudienceOverlapMetrics]
    radar_dimensions: List[str]
    generated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "pairwise_overlaps": [o.to_dict() for o in self.pairwise_overlaps],
            "radar_dimensions": self.radar_dimensions,
            "generated_at": self.generated_at,
            "metadata": self.metadata
        }


# ==============================================================================
# 2. COMPARATIVE INTELLIGENCE ENGINE
# ==============================================================================

class CompetitiveIntelligenceEngine:
    """Engine for multi-channel benchmarking, cohort creation, and audience overlap."""

    def __init__(
        self,
        interim_dir: Optional[pathlib.Path | str] = None,
        output_dir: Optional[pathlib.Path | str] = None
    ) -> None:
        root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        self.interim_dir = pathlib.Path(interim_dir) if interim_dir else root_dir / "data" / "interim"
        self.output_dir = pathlib.Path(output_dir) if output_dir else root_dir / "data" / "output"
        self._videos_df: Optional[pd.DataFrame] = None
        self._comments_df: Optional[pd.DataFrame] = None

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads clean videos and comments parquets with memory caching."""
        if self._videos_df is not None and self._comments_df is not None:
            return self._videos_df, self._comments_df

        f_videos = self.interim_dir / "videos_clean.parquet"
        f_comments = self.interim_dir / "comments_clean.parquet"

        if f_videos.exists():
            try:
                self._videos_df = pd.read_parquet(f_videos)
            except Exception as e:
                logger.warning(f"Failed to read {f_videos}: {e}")
                self._videos_df = pd.DataFrame()
        else:
            self._videos_df = pd.DataFrame()

        if f_comments.exists():
            try:
                self._comments_df = pd.read_parquet(f_comments)
            except Exception as e:
                logger.warning(f"Failed to read {f_comments}: {e}")
                self._comments_df = pd.DataFrame()
        else:
            self._comments_df = pd.DataFrame()

        # Map channel_id onto comments if not already present
        if not self._comments_df.empty and not self._videos_df.empty:
            if "channel_id" not in self._comments_df.columns and "video_id" in self._comments_df.columns:
                ch_map = dict(zip(self._videos_df["video_id"], self._videos_df["channel_id"]))
                self._comments_df["channel_id"] = self._comments_df["video_id"].map(ch_map)

        return self._videos_df, self._comments_df

    def discover_available_channels(self) -> List[Dict[str, Any]]:
        """Returns metadata for all channels present in the video corpus."""
        df_v, df_c = self._load_data()
        if df_v.empty or "channel_id" not in df_v.columns:
            return []

        channels = []
        for cid, vgroup in df_v.groupby("channel_id"):
            cid_str = str(cid)
            sample_title = vgroup["title"].iloc[0] if "title" in vgroup.columns and len(vgroup) > 0 else ""
            default_name = KNOWN_CHANNEL_NAMES.get(cid_str, f"Channel {cid_str[:12]}...")
            
            # Count comments
            c_count = 0
            if not df_c.empty and "channel_id" in df_c.columns:
                c_count = int((df_c["channel_id"] == cid_str).sum())
            elif "total_comments" in vgroup.columns:
                c_count = int(vgroup["total_comments"].fillna(0).sum())

            channels.append({
                "channel_id": cid_str,
                "name": default_name,
                "video_count": len(vgroup),
                "comment_count": c_count,
                "sample_title": str(sample_title)
            })

        # Sort channels by comment count descending
        channels.sort(key=lambda x: x["comment_count"], reverse=True)
        return channels

    def discover_available_cohorts(self, group_by: str = "quarter") -> List[Dict[str, Any]]:
        """Discovers temporal or series cohorts from videos."""
        df_v, df_c = self._load_data()
        if df_v.empty or "published_at" not in df_v.columns:
            return []

        df_work = df_v.copy()
        df_work["pub_dt"] = pd.to_datetime(df_work["published_at"], errors="coerce")
        df_work = df_work.dropna(subset=["pub_dt"])

        if group_by == "year":
            df_work["cohort_key"] = df_work["pub_dt"].dt.year.astype(str)
        else:  # Default quarter
            df_work["cohort_key"] = df_work["pub_dt"].dt.to_period("Q").astype(str)

        cohorts = []
        for ckey, vgroup in df_work.groupby("cohort_key"):
            v_ids = set(vgroup["video_id"].dropna())
            c_count = 0
            if not df_c.empty and "video_id" in df_c.columns:
                c_count = int(df_c["video_id"].isin(v_ids).sum())

            cohorts.append({
                "cohort_id": f"cohort_{ckey}",
                "name": f"Release Cohort {ckey}",
                "cohort_type": "quarter_cohort" if group_by == "quarter" else "year_cohort",
                "video_count": len(vgroup),
                "comment_count": c_count,
                "video_ids": list(v_ids)
            })

        cohorts.sort(key=lambda x: x["name"])
        return cohorts

    def build_profile_from_slice(
        self,
        cohort_id: str,
        cohort_name: str,
        cohort_type: str,
        video_count: int,
        df_slice: pd.DataFrame
    ) -> ChannelCohortProfile:
        """Constructs a comprehensive ChannelCohortProfile from a filtered slice of comments."""
        total_comments = len(df_slice)
        if total_comments == 0:
            return ChannelCohortProfile(
                cohort_id=cohort_id,
                cohort_name=cohort_name,
                cohort_type=cohort_type,
                total_videos=video_count,
                total_comments=0,
                unique_authors=0,
                comments_per_video=0.0,
                avg_likes_per_comment=0.0,
                reply_ratio=0.0,
                sentiment_pos_ratio=0.0,
                sentiment_neg_ratio=0.0,
                avg_vader_compound=0.0,
                mean_toxicity=0.0,
                toxicity_r0=1.0,
                avg_author_loyalty=0.0,
                champion_author_ratio=0.0,
                vocab_entropy=0.0,
                top_emojis=[],
                diurnal_peak_utc=12,
                radar_scores={dim: 10.0 for dim in RADAR_DIMENSIONS}
            )

        # Unique authors
        unique_authors = int(df_slice["author_channel_id"].nunique()) if "author_channel_id" in df_slice.columns else max(1, int(total_comments * 0.7))
        comments_per_video = round(total_comments / max(1, video_count), 2)
        avg_likes = round(float(df_slice["like_count"].mean()), 2) if "like_count" in df_slice.columns else 0.0

        # Reply ratio
        reply_ratio = round(float((df_slice["is_reply"] == True).mean()), 4) if "is_reply" in df_slice.columns else 0.15

        # Sentiment distributions
        if "sentiment_label" in df_slice.columns:
            pos_ratio = float((df_slice["sentiment_label"].str.upper() == "POSITIVE").mean())
            neg_ratio = float((df_slice["sentiment_label"].str.upper() == "NEGATIVE").mean())
        elif "vader_compound" in df_slice.columns:
            pos_ratio = float((df_slice["vader_compound"] > 0.05).mean())
            neg_ratio = float((df_slice["vader_compound"] < -0.05).mean())
        else:
            pos_ratio, neg_ratio = 0.45, 0.20

        vader_comp = round(float(df_slice["vader_compound"].mean()), 3) if "vader_compound" in df_slice.columns else 0.15
        mean_tox = round(float(df_slice["toxicity"].mean()), 4) if "toxicity" in df_slice.columns else 0.08

        # Toxicity Contagion R0 (Reply Toxicity / Root Toxicity)
        if "is_reply" in df_slice.columns and "toxicity" in df_slice.columns:
            root_tox = df_slice[df_slice["is_reply"] == False]["toxicity"].mean()
            reply_tox = df_slice[df_slice["is_reply"] == True]["toxicity"].mean()
            if pd.notna(root_tox) and root_tox > 0.01 and pd.notna(reply_tox):
                tox_r0 = round(float(reply_tox / root_tox), 2)
            else:
                tox_r0 = 1.0
        else:
            tox_r0 = 1.0

        # Author Loyalty & Champion ratio
        if "author_channel_id" in df_slice.columns:
            counts = df_slice["author_channel_id"].value_counts()
            avg_loyalty = round(float(counts.mean()), 2)
            champion_ratio = round(float((counts >= 5).mean()), 4)
        else:
            avg_loyalty = 1.2
            champion_ratio = 0.04

        # Vocabulary Shannon Entropy & Top Emojis
        sample_texts = df_slice["text"].dropna().head(1000).tolist() if "text" in df_slice.columns else []
        all_words = []
        all_emojis = []
        for txt in sample_texts:
            t = str(txt).lower()
            words = re.findall(r"\b[a-zA-Zа-яА-ЯёЁ]{3,}\b", t)
            all_words.extend(words)
            emojis = _EMOJI_PATTERN.findall(t)
            all_emojis.extend(emojis)

        if all_words:
            word_counts = collections.Counter(all_words)
            total_w = len(all_words)
            entropy = -sum((c / total_w) * math.log2(c / total_w) for c in word_counts.values())
            vocab_entropy = round(float(entropy), 2)
        else:
            vocab_entropy = 6.5

        top_emojis = [e for e, _ in collections.Counter(all_emojis).most_common(5)]

        # Diurnal Peak Hour (UTC)
        if "published_at" in df_slice.columns:
            try:
                dt_series = pd.to_datetime(df_slice["published_at"], errors="coerce")
                hours = dt_series.dt.hour.dropna()
                peak_hour = int(hours.mode().iloc[0]) if not hours.empty else 16
            except Exception:
                peak_hour = 16
        else:
            peak_hour = 16

        # Normalization for Radar Dimensions (0..100)
        # 1. Discussion Volume: Logarithmic scale (100 = 500k+ comments)
        v_score = min(100.0, max(10.0, 15.0 * math.log10(max(10, total_comments))))
        # 2. Engagement Velocity: Likes per comment & reply ratio
        e_score = min(100.0, max(15.0, (avg_likes * 8.0) + (reply_ratio * 120.0)))
        # 3. Sentiment Positivity:
        p_score = min(100.0, max(10.0, (pos_ratio * 100.0 * 0.7) + ((vader_comp + 1.0) * 25.0)))
        # 4. Community Safety: Inverted toxicity
        s_score = min(100.0, max(10.0, 100.0 * (1.0 - (mean_tox * 2.5))))
        # 5. Audience Loyalty: Comments per author and champion ratio
        l_score = min(100.0, max(10.0, (avg_loyalty * 22.0) + (champion_ratio * 300.0)))
        # 6. Lexical Sophistication: Vocab entropy
        x_score = min(100.0, max(15.0, vocab_entropy * 10.5))

        radar = {
            "Discussion Volume": round(v_score, 1),
            "Engagement Velocity": round(e_score, 1),
            "Sentiment Positivity": round(p_score, 1),
            "Community Safety": round(s_score, 1),
            "Audience Loyalty": round(l_score, 1),
            "Lexical Sophistication": round(x_score, 1)
        }

        return ChannelCohortProfile(
            cohort_id=cohort_id,
            cohort_name=cohort_name,
            cohort_type=cohort_type,
            total_videos=video_count,
            total_comments=total_comments,
            unique_authors=unique_authors,
            comments_per_video=comments_per_video,
            avg_likes_per_comment=avg_likes,
            reply_ratio=reply_ratio,
            sentiment_pos_ratio=round(pos_ratio, 4),
            sentiment_neg_ratio=round(neg_ratio, 4),
            avg_vader_compound=vader_comp,
            mean_toxicity=mean_tox,
            toxicity_r0=tox_r0,
            avg_author_loyalty=avg_loyalty,
            champion_author_ratio=champion_ratio,
            vocab_entropy=vocab_entropy,
            top_emojis=top_emojis,
            diurnal_peak_utc=peak_hour,
            radar_scores=radar
        )

    def compute_pairwise_overlap(
        self,
        profile_a: ChannelCohortProfile,
        profile_b: ChannelCohortProfile,
        df_ca: pd.DataFrame,
        df_cb: pd.DataFrame,
        top_k: int = 15
    ) -> AudienceOverlapMetrics:
        """Computes Jaccard audience overlap and sentiment/toxicity migration shifts."""
        if df_ca.empty or df_cb.empty or "author_channel_id" not in df_ca.columns or "author_channel_id" not in df_cb.columns:
            return AudienceOverlapMetrics(
                cohort_a_id=profile_a.cohort_id,
                cohort_b_id=profile_b.cohort_id,
                cohort_a_name=profile_a.cohort_name,
                cohort_b_name=profile_b.cohort_name,
                authors_a=profile_a.unique_authors,
                authors_b=profile_b.unique_authors,
                shared_authors=0,
                jaccard_similarity=0.0,
                overlap_coefficient=0.0,
                sentiment_differential=0.0,
                toxicity_differential=0.0,
                top_migrated_commenters=[]
            )

        authors_a_set = set(df_ca["author_channel_id"].dropna())
        authors_b_set = set(df_cb["author_channel_id"].dropna())
        shared = authors_a_set.intersection(authors_b_set)
        union = authors_a_set.union(authors_b_set)

        jaccard = round(len(shared) / max(1, len(union)), 4)
        overlap_coef = round(len(shared) / max(1, min(len(authors_a_set), len(authors_b_set))), 4)

        if not shared:
            return AudienceOverlapMetrics(
                cohort_a_id=profile_a.cohort_id,
                cohort_b_id=profile_b.cohort_id,
                cohort_a_name=profile_a.cohort_name,
                cohort_b_name=profile_b.cohort_name,
                authors_a=len(authors_a_set),
                authors_b=len(authors_b_set),
                shared_authors=0,
                jaccard_similarity=jaccard,
                overlap_coefficient=overlap_coef,
                sentiment_differential=0.0,
                toxicity_differential=0.0,
                top_migrated_commenters=[]
            )

        # Compute sentiment & toxicity differentials on shared authors
        shared_ca = df_ca[df_ca["author_channel_id"].isin(shared)]
        shared_cb = df_cb[df_cb["author_channel_id"].isin(shared)]

        sent_col = "vader_compound" if "vader_compound" in shared_ca.columns else None
        tox_col = "toxicity" if "toxicity" in shared_ca.columns else None

        sent_a = shared_ca[sent_col].mean() if sent_col else 0.0
        sent_b = shared_cb[sent_col].mean() if sent_col else 0.0
        delta_sent = round(float(sent_b - sent_a), 3) if pd.notna(sent_b) and pd.notna(sent_a) else 0.0

        tox_a = shared_ca[tox_col].mean() if tox_col else 0.0
        tox_b = shared_cb[tox_col].mean() if tox_col else 0.0
        delta_tox = round(float(tox_b - tox_a), 4) if pd.notna(tox_b) and pd.notna(tox_a) else 0.0

        # Top migrated commenters (active in both)
        counts_a = shared_ca["author_channel_id"].value_counts()
        counts_b = shared_cb["author_channel_id"].value_counts()
        combined_activity = (counts_a + counts_b).sort_values(ascending=False).head(top_k)

        # Name map
        name_map = {}
        if "author_display_name" in shared_ca.columns:
            name_map.update(dict(zip(shared_ca["author_channel_id"], shared_ca["author_display_name"])))
        if "author_display_name" in shared_cb.columns:
            name_map.update(dict(zip(shared_cb["author_channel_id"], shared_cb["author_display_name"])))

        # Group sentiment by author
        sent_by_a = shared_ca.groupby("author_channel_id")[sent_col].mean() if sent_col else pd.Series()
        sent_by_b = shared_cb.groupby("author_channel_id")[sent_col].mean() if sent_col else pd.Series()

        migrated_list = []
        for aid, total_act in combined_activity.items():
            aid_str = str(aid)
            name = name_map.get(aid_str, f"User_{aid_str[:8]}")
            c_a = int(counts_a.get(aid_str, 0))
            c_b = int(counts_b.get(aid_str, 0))
            s_a = round(float(sent_by_a.get(aid_str, 0.0)), 2) if not sent_by_a.empty else 0.0
            s_b = round(float(sent_by_b.get(aid_str, 0.0)), 2) if not sent_by_b.empty else 0.0

            migrated_list.append({
                "author_id": aid_str,
                "author_name": name,
                "comments_in_a": c_a,
                "comments_in_b": c_b,
                "total_comments": int(total_act),
                "sentiment_in_a": s_a,
                "sentiment_in_b": s_b,
                "sentiment_delta": round(s_b - s_a, 2)
            })

        return AudienceOverlapMetrics(
            cohort_a_id=profile_a.cohort_id,
            cohort_b_id=profile_b.cohort_id,
            cohort_a_name=profile_a.cohort_name,
            cohort_b_name=profile_b.cohort_name,
            authors_a=len(authors_a_set),
            authors_b=len(authors_b_set),
            shared_authors=len(shared),
            jaccard_similarity=jaccard,
            overlap_coefficient=overlap_coef,
            sentiment_differential=delta_sent,
            toxicity_differential=delta_tox,
            top_migrated_commenters=migrated_list
        )

    def compare_channels(
        self,
        channel_ids: Optional[List[str]] = None,
        mock: bool = False
    ) -> ComparativeIntelligenceReport:
        """Compares multiple YouTube channels across all intelligence dimensions."""
        if mock:
            return self.generate_mock_report()

        df_v, df_c = self._load_data()
        if df_v.empty or df_c.empty:
            logger.warning("Empty data. Generating mock comparative report.")
            return self.generate_mock_report()

        avail_channels = self.discover_available_channels()
        if not channel_ids:
            # Auto-select top 2-3 channels
            channel_ids = [ch["channel_id"] for ch in avail_channels[:3]]

        if len(channel_ids) < 2:
            # If only 1 channel exists, split by quarter to provide comparison
            logger.info("Fewer than 2 channels provided. Comparing quarterly release cohorts.")
            return self.compare_cohorts_by_quarter()

        profiles: Dict[str, ChannelCohortProfile] = {}
        slices: Dict[str, pd.DataFrame] = {}

        for cid in channel_ids:
            v_subset = df_v[df_v["channel_id"] == cid]
            v_ids = set(v_subset["video_id"].dropna())
            c_subset = df_c[df_c["video_id"].isin(v_ids)]

            ch_name = KNOWN_CHANNEL_NAMES.get(cid, f"Channel {cid[:12]}...")
            if v_subset.shape[0] > 0 and "title" in v_subset.columns:
                ch_name = f"{ch_name} ({len(v_subset)} vids)"

            prof = self.build_profile_from_slice(
                cohort_id=cid,
                cohort_name=ch_name,
                cohort_type="channel",
                video_count=len(v_subset),
                df_slice=c_subset
            )
            profiles[cid] = prof
            slices[cid] = c_subset

        # Compute pairwise overlaps
        overlaps: List[AudienceOverlapMetrics] = []
        c_list = list(channel_ids)
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                cid_a = c_list[i]
                cid_b = c_list[j]
                overlap = self.compute_pairwise_overlap(
                    profiles[cid_a],
                    profiles[cid_b],
                    slices[cid_a],
                    slices[cid_b]
                )
                overlaps.append(overlap)

        return ComparativeIntelligenceReport(
            profiles=profiles,
            pairwise_overlaps=overlaps,
            radar_dimensions=RADAR_DIMENSIONS,
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata={"comparison_mode": "channels", "total_profiles": len(profiles)}
        )

    def compare_cohorts_by_quarter(self) -> ComparativeIntelligenceReport:
        """Compares temporal release cohorts (quarters) within the dataset."""
        df_v, df_c = self._load_data()
        cohorts = self.discover_available_cohorts(group_by="quarter")

        # Pick the 2-4 most active cohorts
        cohorts.sort(key=lambda x: x["comment_count"], reverse=True)
        active_cohorts = [c for c in cohorts if c["comment_count"] > 50][:3]
        active_cohorts.sort(key=lambda x: x["name"])

        if len(active_cohorts) < 2:
            return self.generate_mock_report()

        profiles: Dict[str, ChannelCohortProfile] = {}
        slices: Dict[str, pd.DataFrame] = {}

        for c in active_cohorts:
            cid = c["cohort_id"]
            v_ids = set(c["video_ids"])
            c_subset = df_c[df_c["video_id"].isin(v_ids)] if not df_c.empty else pd.DataFrame()

            prof = self.build_profile_from_slice(
                cohort_id=cid,
                cohort_name=c["name"],
                cohort_type="quarter_cohort",
                video_count=c["video_count"],
                df_slice=c_subset
            )
            profiles[cid] = prof
            slices[cid] = c_subset

        overlaps: List[AudienceOverlapMetrics] = []
        c_keys = list(profiles.keys())
        for i in range(len(c_keys)):
            for j in range(i + 1, len(c_keys)):
                ca = c_keys[i]
                cb = c_keys[j]
                overlap = self.compute_pairwise_overlap(
                    profiles[ca],
                    profiles[cb],
                    slices[ca],
                    slices[cb]
                )
                overlaps.append(overlap)

        return ComparativeIntelligenceReport(
            profiles=profiles,
            pairwise_overlaps=overlaps,
            radar_dimensions=RADAR_DIMENSIONS,
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata={"comparison_mode": "quarter_cohorts", "total_profiles": len(profiles)}
        )

    def generate_mock_report(self) -> ComparativeIntelligenceReport:
        """Generates a high-fidelity synthetic 3-way comparative intelligence benchmark report."""
        p_alpha = ChannelCohortProfile(
            cohort_id="ch_alpha_osint",
            cohort_name="Alpha OSINT & Geopolitics (Primary)",
            cohort_type="channel",
            total_videos=524,
            total_comments=426544,
            unique_authors=74946,
            comments_per_video=814.0,
            avg_likes_per_comment=3.42,
            reply_ratio=0.284,
            sentiment_pos_ratio=0.482,
            sentiment_neg_ratio=0.215,
            avg_vader_compound=0.245,
            mean_toxicity=0.148,
            toxicity_r0=0.81,
            avg_author_loyalty=5.69,
            champion_author_ratio=0.082,
            vocab_entropy=7.42,
            top_emojis=["🇷🇺", "👍", "🔥", "🤝", "💪"],
            diurnal_peak_utc=17,
            radar_scores={
                "Discussion Volume": 95.0,
                "Engagement Velocity": 68.4,
                "Sentiment Positivity": 62.1,
                "Community Safety": 63.0,
                "Audience Loyalty": 84.5,
                "Lexical Sophistication": 78.0
            }
        )

        p_beta = ChannelCohortProfile(
            cohort_id="ch_beta_tech",
            cohort_name="Beta Strategic Analysis (Competitor)",
            cohort_type="channel",
            total_videos=161,
            total_comments=100819,
            unique_authors=29396,
            comments_per_video=626.2,
            avg_likes_per_comment=4.15,
            reply_ratio=0.342,
            sentiment_pos_ratio=0.395,
            sentiment_neg_ratio=0.288,
            avg_vader_compound=0.092,
            mean_toxicity=0.138,
            toxicity_r0=0.88,
            avg_author_loyalty=3.43,
            champion_author_ratio=0.051,
            vocab_entropy=7.15,
            top_emojis=["🤔", "🇷🇺", "💥", "👏", "🎯"],
            diurnal_peak_utc=15,
            radar_scores={
                "Discussion Volume": 75.0,
                "Engagement Velocity": 78.2,
                "Sentiment Positivity": 48.0,
                "Community Safety": 65.5,
                "Audience Loyalty": 61.2,
                "Lexical Sophistication": 75.1
            }
        )

        p_gamma = ChannelCohortProfile(
            cohort_id="ch_gamma_casual",
            cohort_name="Gamma Stream Replays (Casual Sub-Brand)",
            cohort_type="channel",
            total_videos=68,
            total_comments=14520,
            unique_authors=6850,
            comments_per_video=213.5,
            avg_likes_per_comment=1.85,
            reply_ratio=0.185,
            sentiment_pos_ratio=0.620,
            sentiment_neg_ratio=0.110,
            avg_vader_compound=0.450,
            mean_toxicity=0.055,
            toxicity_r0=0.62,
            avg_author_loyalty=2.12,
            champion_author_ratio=0.028,
            vocab_entropy=6.30,
            top_emojis=["😂", "❤️", "🔥", "🎉", "😎"],
            diurnal_peak_utc=19,
            radar_scores={
                "Discussion Volume": 52.0,
                "Engagement Velocity": 42.0,
                "Sentiment Positivity": 86.5,
                "Community Safety": 86.2,
                "Audience Loyalty": 44.0,
                "Lexical Sophistication": 66.2
            }
        )

        top_shared_ab = [
            {
                "author_id": "UC_shared_analyst_01",
                "author_name": "Valery Tactical",
                "comments_in_a": 48,
                "comments_in_b": 35,
                "total_comments": 83,
                "sentiment_in_a": 0.35,
                "sentiment_in_b": -0.12,
                "sentiment_delta": -0.47
            },
            {
                "author_id": "UC_shared_patriot_02",
                "author_name": "Siberian Watcher",
                "comments_in_a": 62,
                "comments_in_b": 18,
                "total_comments": 80,
                "sentiment_in_a": 0.52,
                "sentiment_in_b": 0.44,
                "sentiment_delta": -0.08
            },
            {
                "author_id": "UC_shared_skeptic_03",
                "author_name": "Geopolitical Observer",
                "comments_in_a": 29,
                "comments_in_b": 41,
                "total_comments": 70,
                "sentiment_in_a": -0.05,
                "sentiment_in_b": -0.38,
                "sentiment_delta": -0.33
            },
            {
                "author_id": "UC_shared_loyalist_04",
                "author_name": "Elena M.",
                "comments_in_a": 34,
                "comments_in_b": 22,
                "total_comments": 56,
                "sentiment_in_a": 0.68,
                "sentiment_in_b": 0.55,
                "sentiment_delta": -0.13
            }
        ]

        top_shared_ag = [
            {
                "author_id": "UC_shared_casual_05",
                "author_name": "Dmitry Live",
                "comments_in_a": 15,
                "comments_in_b": 28,
                "total_comments": 43,
                "sentiment_in_a": 0.20,
                "sentiment_in_b": 0.75,
                "sentiment_delta": +0.55
            },
            {
                "author_id": "UC_shared_casual_06",
                "author_name": "Alex TechStream",
                "comments_in_a": 12,
                "comments_in_b": 19,
                "total_comments": 31,
                "sentiment_in_a": 0.15,
                "sentiment_in_b": 0.62,
                "sentiment_delta": +0.47
            }
        ]

        overlap_ab = AudienceOverlapMetrics(
            cohort_a_id="ch_alpha_osint",
            cohort_b_id="ch_beta_tech",
            cohort_a_name="Alpha OSINT & Geopolitics (Primary)",
            cohort_b_name="Beta Strategic Analysis (Competitor)",
            authors_a=74946,
            authors_b=29396,
            shared_authors=17280,
            jaccard_similarity=0.1985,
            overlap_coefficient=0.5878,
            sentiment_differential=-0.153,
            toxicity_differential=-0.010,
            top_migrated_commenters=top_shared_ab
        )

        overlap_ag = AudienceOverlapMetrics(
            cohort_a_id="ch_alpha_osint",
            cohort_b_id="ch_gamma_casual",
            cohort_a_name="Alpha OSINT & Geopolitics (Primary)",
            cohort_b_name="Gamma Stream Replays (Casual Sub-Brand)",
            authors_a=74946,
            authors_b=6850,
            shared_authors=3120,
            jaccard_similarity=0.0396,
            overlap_coefficient=0.4555,
            sentiment_differential=+0.205,
            toxicity_differential=-0.093,
            top_migrated_commenters=top_shared_ag
        )

        return ComparativeIntelligenceReport(
            profiles={
                "ch_alpha_osint": p_alpha,
                "ch_beta_tech": p_beta,
                "ch_gamma_casual": p_gamma
            },
            pairwise_overlaps=[overlap_ab, overlap_ag],
            radar_dimensions=RADAR_DIMENSIONS,
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata={"comparison_mode": "synthetic_benchmark", "total_profiles": 3}
        )

    def export_report(
        self,
        report: ComparativeIntelligenceReport,
        output_path: str | pathlib.Path,
        export_format: str = "json"
    ) -> Tuple[bool, str]:
        """Exports the comparative intelligence report to JSON or CSV."""
        out_p = pathlib.Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            if export_format.lower() == "csv" or out_p.suffix.lower() == ".csv":
                # Export profiles table
                rows = []
                for pid, p in report.profiles.items():
                    d = p.to_dict()
                    # Flatten radar
                    for dim, score in p.radar_scores.items():
                        d[f"radar_{dim.lower().replace(' ', '_')}"] = score
                    d["top_emojis"] = " ".join(p.top_emojis)
                    d.pop("radar_scores", None)
                    rows.append(d)
                df_export = pd.DataFrame(rows)
                df_export.to_csv(out_p, index=False, encoding="utf-8")
                return True, f"Profiles CSV successfully saved to {out_p}"
            else:
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
                return True, f"Report JSON successfully saved to {out_p}"
        except Exception as e:
            return False, f"Export failed: {e}"


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ytint // Multi-Channel & Playlist Competitive Intelligence Engine"
    )
    parser.add_argument("--interim-dir", type=str, help="Path to data/interim directory")
    parser.add_argument("--output-dir", type=str, help="Path to data/output directory")
    parser.add_argument("--list-channels", action="store_true", help="List all available channels in video corpus")
    parser.add_argument("--compare", nargs="*", help="Channel IDs to compare (e.g. CH1 CH2)")
    parser.add_argument("--cohorts", choices=["quarter", "year"], help="Compare temporal release cohorts")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic multi-channel benchmark demo report")
    parser.add_argument("--export", type=str, help="Output file path (.json or .csv)")

    args = parser.parse_args()
    engine = CompetitiveIntelligenceEngine(args.interim_dir, args.output_dir)

    print("\n" + "=" * 80)
    print(" 🌐 ytint // Multi-Channel & Playlist Competitive Intelligence Engine")
    print("=" * 80)

    if args.list_channels:
        channels = engine.discover_available_channels()
        print(f"\nDiscovered Channels ({len(channels)} total):")
        print(f"{'Channel ID':<28} {'Videos':<8} {'Comments':<10} {'Name / Title'}")
        print("-" * 80)
        for ch in channels:
            print(f"{ch['channel_id']:<28} {ch['video_count']:<8} {ch['comment_count']:<10} {ch['name']}")
        print("-" * 80 + "\n")
        return

    if args.mock:
        report = engine.generate_mock_report()
    elif args.cohorts:
        report = engine.compare_cohorts_by_quarter()
    else:
        report = engine.compare_channels(channel_ids=args.compare)

    print(f" Profiled Entities: {len(report.profiles)}")
    print(f" Pairwise Overlaps Computed: {len(report.pairwise_overlaps)}")
    print("=" * 80 + "\n")

    # Display Profiles Table
    print("--- 📊 Benchmark Profiles ---")
    print(f"{'Entity':<35} {'Videos':<7} {'Comments':<10} {'Authors':<9} {'Pos%':<7} {'Tox R0':<7} {'Loyalty'}")
    print("-" * 85)
    for pid, p in report.profiles.items():
        print(f"{p.cohort_name[:34]:<35} {p.total_videos:<7} {p.total_comments:<10} {p.unique_authors:<9} {p.sentiment_pos_ratio*100:>4.1f}%  {p.toxicity_r0:>5.2f}  {p.avg_author_loyalty:>5.2f}")
    print("-" * 85 + "\n")

    # Display Radar Scores
    print("--- 🎯 Normalized Radar Dimensions (0..100) ---")
    header = f"{'Dimension':<25}" + "".join([f"{p.cohort_name[:15]:<18}" for p in report.profiles.values()])
    print(header)
    print("-" * len(header))
    for dim in report.radar_dimensions:
        line = f"{dim:<25}"
        for p in report.profiles.values():
            score = p.radar_scores.get(dim, 0.0)
            line += f"{score:>6.1f} / 100       "
        print(line)
    print("-" * len(header) + "\n")

    # Display Pairwise Overlaps
    if report.pairwise_overlaps:
        print("--- 👥 Audience Overlap & Commenter Migration ---")
        print(f"{'Cohort Pair':<45} {'Shared':<9} {'Jaccard':<9} {'Overlap Coef':<14} {'Δ Sent'}")
        print("-" * 85)
        for ov in report.pairwise_overlaps:
            pair_label = f"{ov.cohort_a_name[:20]} vs {ov.cohort_b_name[:20]}"
            print(f"{pair_label:<45} {ov.shared_authors:<9} {ov.jaccard_similarity*100:>5.2f}%   {ov.overlap_coefficient*100:>6.2f}%        {ov.sentiment_differential:+.3f}")
            if ov.top_migrated_commenters:
                print("   Top Migrated Commenters:")
                for m in ov.top_migrated_commenters[:3]:
                    print(f"     • {m['author_name'][:20]:<22} (A: {m['comments_in_a']} cmds, B: {m['comments_in_b']} cmds | Δ Sent: {m['sentiment_delta']:+.2f})")
        print("-" * 85 + "\n")

    if args.export:
        ok, msg = engine.export_report(report, args.export)
        print(f"Export Status: {msg}\n")


if __name__ == "__main__":
    main()
