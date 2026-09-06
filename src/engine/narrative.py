"""ytint // Temporal Narrative Scene Reaction Forensics Engine (src/engine/narrative.py)

Cross-modal intelligence engine aligning audience comments and reactions
with video playback timelines. Classifies scene reactions into validated
taxonomies (humor, surprise, critique, emotional, navigation), pinpoints
confusion hotspots and outrage triggers, and compiles audience quote montages.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger("ytint.narrative")

# ==============================================================================
# 1. REACTION TAXONOMY & REGEX PATTERNS
# ==============================================================================

REACTION_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "humor_laughter": {
        "label": "😂 Humor & Laughter",
        "color": "#00e599",
        "patterns": [
            r"[😂🤣😆😹]",
            r"\b(lol+|lmao+|rofl+|смешн\w*|угар\w*|ор\b|ржу|хах\w*|кек\w*|ору|мем\w*|funny|hilarious|comedy)\b"
        ]
    },
    "shock_surprise": {
        "label": "😱 Shock & Surprise",
        "color": "#ffaa00",
        "patterns": [
            r"[😱🤯😳😮]",
            r"\b(wtf|omg|holy\s+shit|шок\w*|жесть|офигеть|капец|нихера|plot\s*twist|unbelievable|shocking)\b"
        ]
    },
    "emotional_touching": {
        "label": "❤️ Emotional & Touching",
        "color": "#a855f7",
        "patterns": [
            r"[❤️🥺😭😍💔]",
            r"\b(плачу|душевн\w*|слез\w*|слёз\w*|трогательн\w*|wholesome|crying|beautiful|heartwarming|sad|emotional)\b"
        ]
    },
    "critique_analytical": {
        "label": "🤔 Critique & Confusion",
        "color": "#0066fe",
        "patterns": [
            r"[🤔🧐❓]",
            r"\b(ошибк\w*|ляп\w*|нелогичн\w*|почему|зачем|логика|сюжет|plot\s*hole|mistake|logic|confused|wait\s+what|doesn't\s+make\s+sense)\b"
        ]
    },
    "chapter_navigation": {
        "label": "⏱️ Chapter & Music",
        "color": "#64748b",
        "patterns": [
            r"[⏱️🕒🎵🎧]",
            r"\b(таймкод\w*|трек|музык\w*|начало|конец|интро|аутро|intro|outro|music|song|track|timestamp)\b"
        ]
    }
}

TIMESTAMP_REGEX = re.compile(
    r"(?:(?:https?://[^\s]*[?&]t=(\d+)s?)|(?:\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b))",
    re.IGNORECASE
)

QUESTION_REGEX = re.compile(
    r"(?:\?+|почему|зачем|как так|в смысле|wait what|why did|how come|is that true)",
    re.IGNORECASE
)


def format_seconds(seconds: int) -> str:
    """Formats an integer second count into clean MM:SS or HH:MM:SS format."""
    sec = max(0, int(seconds))
    if sec >= 3600:
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        m = sec // 60
        s = sec % 60
        return f"{m:02d}:{s:02d}"


def extract_timestamps_from_text(text: str) -> List[int]:
    """Extracts all timestamp mentions in seconds from a raw comment string."""
    if not text or not isinstance(text, str):
        return []
    
    clean_text = html.unescape(text)
    results = []
    
    for match in TIMESTAMP_REGEX.finditer(clean_text):
        url_t = match.group(1)
        if url_t:
            results.append(int(url_t))
        else:
            h_str, m_str, s_str = match.group(2), match.group(3), match.group(4)
            h = int(h_str) if h_str else 0
            m = int(m_str) if m_str else 0
            s = int(s_str) if s_str else 0
            total_sec = h * 3600 + m * 60 + s
            results.append(total_sec)
            
    return sorted(list(set(results)))


def classify_comment_reaction(text: str) -> str:
    """Classifies a comment into one of the reaction taxonomy buckets."""
    text_lower = str(text).lower()
    for r_type, config in REACTION_TAXONOMY.items():
        for pat in config["patterns"]:
            if re.search(pat, text_lower, re.IGNORECASE):
                return r_type
    return "general_reaction"


# ==============================================================================
# 2. DATA MODELS
# ==============================================================================

@dataclass
class TimestampQuote:
    """An exemplar audience quote pinned to a specific narrative moment."""
    comment_id: str
    author_name: str
    second: int
    formatted_time: str
    like_count: int
    sentiment: float
    toxicity: float
    reaction_type: str
    text_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneReactionCluster:
    """A discrete temporal scene bucket along the video timeline."""
    bin_index: int
    start_sec: int
    end_sec: int
    formatted_time: str
    comment_count: int
    dominant_reaction: str
    reaction_label: str
    reaction_color: str
    mean_sentiment: float
    mean_toxicity: float
    is_confusion_hotspot: bool
    is_humor_peak: bool
    is_outrage_spark: bool
    reaction_counts: Dict[str, int]
    quotes: List[TimestampQuote] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["quotes"] = [q.to_dict() if isinstance(q, TimestampQuote) else q for q in self.quotes]
        return d


@dataclass
class ConfusionHotspot:
    """A scene moment exhibiting elevated viewer confusion or plot critiques."""
    start_sec: int
    end_sec: int
    formatted_time: str
    question_count: int
    mean_sentiment: float
    critique_ratio: float
    sample_questions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoNarrativeReport:
    """Comprehensive narrative scene analysis for a video upload."""
    video_id: str
    video_title: str
    total_timestamp_comments: int
    total_scenes: int
    video_duration_sec: int
    dominant_reaction: str
    reaction_breakdown: Dict[str, int]
    scenes: List[SceneReactionCluster]
    confusion_hotspots: List[ConfusionHotspot]
    humor_peaks: List[SceneReactionCluster]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "total_timestamp_comments": self.total_timestamp_comments,
            "total_scenes": self.total_scenes,
            "video_duration_sec": self.video_duration_sec,
            "dominant_reaction": self.dominant_reaction,
            "reaction_breakdown": self.reaction_breakdown,
            "scenes": [s.to_dict() for s in self.scenes],
            "confusion_hotspots": [h.to_dict() for h in self.confusion_hotspots],
            "humor_peaks": [p.to_dict() for p in self.humor_peaks],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Converts scene clusters into a tabular DataFrame for analytics and charting."""
        rows = []
        for s in self.scenes:
            rows.append({
                "bin_index": s.bin_index,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "formatted_time": s.formatted_time,
                "comment_count": s.comment_count,
                "dominant_reaction": s.dominant_reaction,
                "reaction_label": s.reaction_label,
                "mean_sentiment": s.mean_sentiment,
                "mean_toxicity": s.mean_toxicity,
                "is_confusion_hotspot": s.is_confusion_hotspot,
                "is_humor_peak": s.is_humor_peak,
                "is_outrage_spark": s.is_outrage_spark,
                "top_quote": s.quotes[0].text_snippet if s.quotes else ""
            })
        return pd.DataFrame(rows)


# ==============================================================================
# 3. CORE NARRATIVE ENGINE
# ==============================================================================

class NarrativeForensicsEngine:
    """Core analytical engine for timestamp reaction alignment and scene forensics."""

    def __init__(
        self,
        interim_dir: Optional[pathlib.Path | str] = None,
        output_dir: Optional[pathlib.Path | str] = None,
    ) -> None:
        self.interim_dir = pathlib.Path(interim_dir or "data/interim")
        self.output_dir = pathlib.Path(output_dir or "data/output")
        
        self.df_videos: pd.DataFrame = pd.DataFrame()
        self.df_reaction_map: pd.DataFrame = pd.DataFrame()
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Loads video metadata and reaction map artifacts if present."""
        vid_p = self.interim_dir / "videos_clean.parquet"
        if vid_p.exists():
            try:
                self.df_videos = pd.read_parquet(vid_p)
            except Exception as e:
                logger.warning(f"Failed to load {vid_p}: {e}")

        rx_p = self.output_dir / "video_reaction_map.parquet"
        if rx_p.exists():
            try:
                self.df_reaction_map = pd.read_parquet(rx_p)
            except Exception as e:
                logger.warning(f"Failed to load {rx_p}: {e}")

    def get_available_videos(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Returns videos ranked by total timestamp mentions."""
        if self.df_reaction_map.empty or "video_id" not in self.df_reaction_map.columns:
            return []

        counts = self.df_reaction_map.groupby("video_id")["comment_count"].sum().reset_index()
        counts = counts.sort_values(by="comment_count", ascending=False).head(limit)

        results = []
        for _, row in counts.iterrows():
            vid = row["video_id"]
            cnt = int(row["comment_count"])
            title = vid
            if not self.df_videos.empty and "video_id" in self.df_videos.columns:
                match = self.df_videos[self.df_videos["video_id"] == vid]
                if not match.empty and "title" in match.columns:
                    title = str(match["title"].iloc[0])
            results.append({
                "video_id": vid,
                "title": title,
                "timestamp_comments": cnt
            })
        return results

    def _load_video_comments_with_timestamps(self, video_id: str) -> pd.DataFrame:
        """Extracts and parses all comments mentioning timestamps for a given video."""
        comments_p = self.interim_dir / "comments_clean.parquet"
        if not comments_p.exists():
            return pd.DataFrame()

        try:
            import pyarrow.dataset as ds
            dataset = ds.dataset(str(comments_p), format="parquet")
            table = dataset.to_table(
                filter=ds.field("video_id") == video_id,
                columns=["comment_id", "video_id", "author_display_name", "like_count", "vader_compound", "toxicity", "text"]
            )
            df = table.to_pandas()
        except Exception:
            # Fallback to direct pandas read
            try:
                df = pd.read_parquet(comments_p)
                df = df[df["video_id"] == video_id]
            except Exception as e:
                logger.warning(f"Could not load comments for video {video_id}: {e}")
                return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Extract seconds from text
        extracted_rows = []
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            ts_list = extract_timestamps_from_text(text)
            if ts_list:
                reaction = classify_comment_reaction(text)
                for sec in ts_list:
                    extracted_rows.append({
                        "comment_id": row["comment_id"],
                        "author_name": row.get("author_display_name", "Anonymous"),
                        "second": sec,
                        "like_count": int(row.get("like_count", 0)),
                        "sentiment": float(row.get("vader_compound", 0.0)),
                        "toxicity": float(row.get("toxicity", 0.0)),
                        "reaction_type": reaction,
                        "text": text
                    })

        return pd.DataFrame(extracted_rows)

    def analyze_video_narrative(
        self,
        video_id: str,
        bin_seconds: int = 15,
        mock_if_empty: bool = True
    ) -> VideoNarrativeReport:
        """Performs full temporal scene segmentation and narrative forensics on a video."""
        # Retrieve video title
        title = video_id
        if not self.df_videos.empty and "video_id" in self.df_videos.columns:
            m = self.df_videos[self.df_videos["video_id"] == video_id]
            if not m.empty and "title" in m.columns:
                title = str(m["title"].iloc[0])

        df_c = self._load_video_comments_with_timestamps(video_id)

        if df_c.empty:
            if mock_if_empty:
                logger.info(f"No timestamp comments found for {video_id}. Generating mock chronicle.")
                return self.generate_mock_narrative(video_id=video_id, title=title, bin_seconds=bin_seconds)
            else:
                return VideoNarrativeReport(
                    video_id=video_id,
                    video_title=title,
                    total_timestamp_comments=0,
                    total_scenes=0,
                    video_duration_sec=0,
                    dominant_reaction="None",
                    reaction_breakdown={},
                    scenes=[],
                    confusion_hotspots=[],
                    humor_peaks=[]
                )

        max_sec = int(df_c["second"].max())
        duration_sec = max(max_sec + bin_seconds, bin_seconds * 10)
        num_bins = math.ceil(duration_sec / bin_seconds)

        scenes: List[SceneReactionCluster] = []
        confusion_hotspots: List[ConfusionHotspot] = []
        humor_peaks: List[SceneReactionCluster] = []
        reaction_counts_total: Dict[str, int] = {}

        # Slicing into temporal bins
        for b_idx in range(num_bins):
            start_s = b_idx * bin_seconds
            end_s = start_s + bin_seconds
            mask = (df_c["second"] >= start_s) & (df_c["second"] < end_s)
            bin_df = df_c[mask]

            if bin_df.empty:
                continue

            comment_count = len(bin_df)
            rx_counts = bin_df["reaction_type"].value_counts().to_dict()
            for rk, rv in rx_counts.items():
                reaction_counts_total[rk] = reaction_counts_total.get(rk, 0) + rv

            # Identify dominant reaction (excluding general_reaction if other specific reactions exist)
            specific_counts = {k: v for k, v in rx_counts.items() if k != "general_reaction"}
            if specific_counts:
                dominant_rx = max(specific_counts, key=specific_counts.get)  # type: ignore
            else:
                dominant_rx = "general_reaction"

            rx_info = REACTION_TAXONOMY.get(dominant_rx, {
                "label": "💬 General Comment",
                "color": "#94a3b8"
            })

            mean_sentiment = float(bin_df["sentiment"].mean())
            mean_toxicity = float(bin_df["toxicity"].mean())

            # Detect confusion / question density
            question_count = int(bin_df["text"].apply(lambda t: bool(QUESTION_REGEX.search(str(t)))).sum())
            critique_count = rx_counts.get("critique_analytical", 0)
            is_confusion = (question_count >= 2) or (critique_count >= 2) or (dominant_rx == "critique_analytical" and comment_count >= 2)

            # Detect humor peak
            is_humor = (dominant_rx == "humor_laughter" and comment_count >= 2)

            # Detect outrage spark
            is_outrage = (mean_toxicity >= 0.25) or (mean_sentiment <= -0.40 and comment_count >= 2)

            # Extract top exemplar quotes sorted by like count
            sorted_quotes = bin_df.sort_values(by="like_count", ascending=False)
            quotes: List[TimestampQuote] = []
            for _, q_row in sorted_quotes.head(5).iterrows():
                raw_text = str(q_row["text"])
                snippet = raw_text[:110] + ("..." if len(raw_text) > 110 else "")
                quotes.append(TimestampQuote(
                    comment_id=str(q_row["comment_id"]),
                    author_name=str(q_row["author_name"]),
                    second=int(q_row["second"]),
                    formatted_time=format_seconds(int(q_row["second"])),
                    like_count=int(q_row["like_count"]),
                    sentiment=round(float(q_row["sentiment"]), 3),
                    toxicity=round(float(q_row["toxicity"]), 3),
                    reaction_type=str(q_row["reaction_type"]),
                    text_snippet=snippet
                ))

            cluster = SceneReactionCluster(
                bin_index=b_idx,
                start_sec=start_s,
                end_sec=end_s,
                formatted_time=f"{format_seconds(start_s)} - {format_seconds(end_s)}",
                comment_count=comment_count,
                dominant_reaction=dominant_rx,
                reaction_label=rx_info["label"],
                reaction_color=rx_info["color"],
                mean_sentiment=round(mean_sentiment, 3),
                mean_toxicity=round(mean_toxicity, 3),
                is_confusion_hotspot=is_confusion,
                is_humor_peak=is_humor,
                is_outrage_spark=is_outrage,
                reaction_counts=rx_counts,
                quotes=quotes
            )
            scenes.append(cluster)

            if is_confusion:
                sample_qs = [q.text_snippet for q in quotes if QUESTION_REGEX.search(q.text_snippet)][:3]
                if not sample_qs and quotes:
                    sample_qs = [quotes[0].text_snippet]
                confusion_hotspots.append(ConfusionHotspot(
                    start_sec=start_s,
                    end_sec=end_s,
                    formatted_time=cluster.formatted_time,
                    question_count=question_count,
                    mean_sentiment=round(mean_sentiment, 3),
                    critique_ratio=round(critique_count / max(1, comment_count), 2),
                    sample_questions=sample_qs
                ))

            if is_humor:
                humor_peaks.append(cluster)

        overall_dominant = "general_reaction"
        if reaction_counts_total:
            overall_dominant = max(reaction_counts_total, key=reaction_counts_total.get)  # type: ignore

        return VideoNarrativeReport(
            video_id=video_id,
            video_title=title,
            total_timestamp_comments=len(df_c),
            total_scenes=len(scenes),
            video_duration_sec=duration_sec,
            dominant_reaction=overall_dominant,
            reaction_breakdown=reaction_counts_total,
            scenes=scenes,
            confusion_hotspots=confusion_hotspots,
            humor_peaks=humor_peaks
        )

    def generate_mock_narrative(
        self,
        video_id: str = "MOCK_VIDEO_001",
        title: str = "Demo Video: Architectural Walkthrough",
        bin_seconds: int = 15
    ) -> VideoNarrativeReport:
        """Generates realistic synthetic multi-scene chronicle data for offline testing."""
        mock_scene_specs = [
            (30, 45, "chapter_navigation", 4, 0.40, 0.01, False, False, False, "Intro section timestamp markers"),
            (75, 90, "humor_laughter", 12, 0.75, 0.02, False, True, False, "Hahaha look at his face at 1:20 😂"),
            (120, 135, "critique_analytical", 8, -0.20, 0.05, True, False, False, "Wait, why does the calculation say 42 here?"),
            (180, 195, "shock_surprise", 9, -0.45, 0.12, False, False, False, "OMG did that really just happen?? 😱"),
            (240, 255, "humor_laughter", 15, 0.88, 0.01, False, True, False, "ROFL the cat walking across the desk 🤣"),
            (300, 315, "critique_analytical", 6, -0.35, 0.08, True, False, False, "At 5:10 you forgot to mention the exception handling"),
            (360, 375, "emotional_touching", 11, 0.92, 0.01, False, False, False, "That tribute at 6:10 brought tears to my eyes ❤️"),
            (420, 435, "shock_surprise", 7, 0.10, 0.35, False, False, True, "Wait that statement is completely contradictory!"),
        ]

        scenes: List[SceneReactionCluster] = []
        confusion_hotspots: List[ConfusionHotspot] = []
        humor_peaks: List[SceneReactionCluster] = []
        rx_totals: Dict[str, int] = {}

        for idx, (start_s, end_s, rx_type, count, sent, tox, is_conf, is_hum, is_out, quote_txt) in enumerate(mock_scene_specs):
            rx_info = REACTION_TAXONOMY.get(rx_type, {"label": rx_type, "color": "#94a3b8"})
            rx_totals[rx_type] = rx_totals.get(rx_type, 0) + count

            quote = TimestampQuote(
                comment_id=f"mock_c_{idx}",
                author_name=f"@User_{idx+10}",
                second=start_s + 5,
                formatted_time=format_seconds(start_s + 5),
                like_count=count * 3,
                sentiment=sent,
                toxicity=tox,
                reaction_type=rx_type,
                text_snippet=quote_txt
            )

            cluster = SceneReactionCluster(
                bin_index=idx,
                start_sec=start_s,
                end_sec=end_s,
                formatted_time=f"{format_seconds(start_s)} - {format_seconds(end_s)}",
                comment_count=count,
                dominant_reaction=rx_type,
                reaction_label=rx_info["label"],
                reaction_color=rx_info["color"],
                mean_sentiment=sent,
                mean_toxicity=tox,
                is_confusion_hotspot=is_conf,
                is_humor_peak=is_hum,
                is_outrage_spark=is_out,
                reaction_counts={rx_type: count},
                quotes=[quote]
            )
            scenes.append(cluster)

            if is_conf:
                confusion_hotspots.append(ConfusionHotspot(
                    start_sec=start_s,
                    end_sec=end_s,
                    formatted_time=cluster.formatted_time,
                    question_count=3,
                    mean_sentiment=sent,
                    critique_ratio=0.75,
                    sample_questions=[quote_txt]
                ))
            if is_hum:
                humor_peaks.append(cluster)

        total_comments = sum(s.comment_count for s in scenes)
        return VideoNarrativeReport(
            video_id=video_id,
            video_title=title,
            total_timestamp_comments=total_comments,
            total_scenes=len(scenes),
            video_duration_sec=480,
            dominant_reaction="humor_laughter",
            reaction_breakdown=rx_totals,
            scenes=scenes,
            confusion_hotspots=confusion_hotspots,
            humor_peaks=humor_peaks
        )

    def export_report(self, report: VideoNarrativeReport, output_path: pathlib.Path | str) -> Tuple[bool, str]:
        """Exports the video narrative chronicle to CSV or JSON."""
        try:
            out_p = pathlib.Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".json":
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            else:
                df = report.to_dataframe()
                df.to_csv(out_p, index=False)
            return True, f"Successfully exported narrative chronicle to '{out_p}'."
        except Exception as e:
            return False, f"Failed to export chronicle: {e}"


# ==============================================================================
# 4. CLI RUNNER & ENTRYPOINT
# ==============================================================================

def main(args: Optional[List[str]] = None) -> int:
    """CLI Entrypoint for ytint-narrative."""
    parser = argparse.ArgumentParser(
        prog="ytint-narrative",
        description="ytint // Temporal Narrative Scene Reaction Forensics Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  ytint-narrative --list-videos
  ytint-narrative --video-id 6h547XdZYiQ
  ytint-narrative --video-id 6h547XdZYiQ --step-secs 30
  ytint-narrative --video-id 6h547XdZYiQ --hotspots
  ytint-narrative --video-id 6h547XdZYiQ --export scratch/narrative.csv
  ytint-narrative --mock
"""
    )

    parser.add_argument("--video-id", "-v", type=str, default=None, help="Target YouTube video ID")
    parser.add_argument("--list-videos", "-l", action="store_true", help="List videos with timestamp reaction mentions")
    parser.add_argument("--step-secs", "-s", type=int, default=15, help="Temporal scene bucket size in seconds (default: 15)")
    parser.add_argument("--hotspots", action="store_true", help="Display only detected confusion drops and outrage spikes")
    parser.add_argument("--export", "-e", type=str, default=None, help="Export chronicle to CSV or JSON file path")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode")

    parsed = parser.parse_args(args)
    engine = NarrativeForensicsEngine()

    # Mode 1: List videos
    if parsed.list_videos:
        vids = engine.get_available_videos()
        if not vids:
            print("No video timestamp reactions found in data/output/video_reaction_map.parquet.")
            return 0
        print(f"\nDiscovered {len(vids)} Videos with Timestamp Reactions:")
        print(f"{'Video ID':<14} {'Reactions':>10}  {'Title'}")
        print("=" * 72)
        for v in vids:
            print(f"{v['video_id']:<14} {v['timestamp_comments']:>10,}  {v['title'][:44]}")
        print()
        return 0

    # Mode 2: Mock mode
    if parsed.mock or not parsed.video_id:
        v_id = parsed.video_id or "MOCK_VIDEO_001"
        report = engine.generate_mock_narrative(video_id=v_id, bin_seconds=parsed.step_secs)
    else:
        report = engine.analyze_video_narrative(video_id=parsed.video_id, bin_seconds=parsed.step_secs)

    if parsed.export:
        ok, msg = engine.export_report(report, parsed.export)
        if ok:
            print(f"[SUCCESS] {msg}")
            return 0
        else:
            print(f"[ERROR] {msg}", file=sys.stderr)
            return 1

    # Print Narrative Chronicle
    print("=" * 76)
    print(f" ⏱️ ytint // Narrative Scene Reaction Forensics: {report.video_title}")
    print(f" Video ID: {report.video_id} | Total Timestamp Comments: {report.total_timestamp_comments:,}")
    print(f" Timeline Scenes: {report.total_scenes} | Duration: {format_seconds(report.video_duration_sec)}")
    print(f" Dominant Reaction: {report.dominant_reaction}")
    print("=" * 76 + "\n")

    if parsed.hotspots:
        print("🔍 Detected Viewer Confusion Hotspots:")
        if not report.confusion_hotspots:
            print("  (No confusion hotspots detected)")
        for h in report.confusion_hotspots:
            print(f"  • {h.formatted_time} : Questions={h.question_count} | Sentiment={h.mean_sentiment:+.2f}")
            for q in h.sample_questions:
                print(f"    - \"{q}\"")
        print()
        return 0

    print(f"{'Time Window':<16} {'Comments':>8} {'Reaction Type':<24} {'Sentiment':>10} {'Toxicity':>10}")
    print("-" * 76)
    for s in report.scenes:
        flag = ""
        if s.is_confusion_hotspot:
            flag = " [❓ CONFUSION]"
        elif s.is_humor_peak:
            flag = " [😂 HUMOR]"
        elif s.is_outrage_spark:
            flag = " [🚨 OUTRAGE]"

        rx_str = f"{s.reaction_label}{flag}"
        print(f"{s.formatted_time:<16} {s.comment_count:>8} {rx_str:<24} {s.mean_sentiment:>+10.2f} {s.mean_toxicity:>10.2f}")
        if s.quotes:
            print(f"    > \"{s.quotes[0].text_snippet}\" ({s.quotes[0].author_name}, 👍 {s.quotes[0].like_count})")

    print("\n" + "=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
