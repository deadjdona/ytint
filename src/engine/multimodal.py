"""ytint // Multimodal Transcript & Vision Alignment Engine (src/engine/multimodal.py)

Cross-modal intelligence engine aligning speech-to-text transcripts (Whisper / VTT /
YouTube TimedText) and visual keyframes with second-by-second comment reactions and
confusion hotspots. Evaluates viewer skepticism and controversy, computes cross-modal
coherence, pinpoints spoken root causes of viewer confusion, and synthesizes
ready-to-paste YouTube chapter markers.
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
from pathlib import Path
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

from engine.config_loader import load_config

logger = logging.getLogger("ytint.multimodal")

# ==============================================================================
# 1. TAXONOMY, SCENE PROFILES & CONSTANTS
# ==============================================================================

CLAIM_TYPES: Dict[str, str] = {
    "FACTUAL_ASSERTION": "Definitive statement about technology, metrics, or architecture",
    "STATISTICAL_CLAIM": "Quantitative performance metric or benchmark measurement",
    "SPECULATION_OPINION": "Subjective opinion or future trajectory prediction",
    "RECOMMENDATION": "Prescriptive advice for engineering teams or developers"
}

SCENE_TYPES: Dict[str, Dict[str, Any]] = {
    "talking_head": {
        "label": "🗣️ Talking Head / Presenter",
        "color": "#3b82f6",
        "description": "Creator speaking directly to camera without dense screen graphics."
    },
    "screen_code": {
        "label": "💻 Code & Terminal Demo",
        "color": "#10b981",
        "description": "Active coding, IDE terminal execution, or live software demo."
    },
    "slide_presentation": {
        "label": "📊 Slide & Bullet Points",
        "color": "#8b5cf6",
        "description": "Structured slide deck, concept definitions, or key takeaways."
    },
    "chart_diagram": {
        "label": "📈 Architecture & Diagram",
        "color": "#f59e0b",
        "description": "System architecture diagram, data flow graph, or visual flow chart."
    },
    "b_roll_demo": {
        "label": "🎬 B-Roll & Real-World Demo",
        "color": "#ec4899",
        "description": "Supplemental illustrative footage, animation, or hardware footage."
    },
    "meme_overlay": {
        "label": "🃏 Meme & Comic Overlay",
        "color": "#ef4444",
        "description": "Humorous visual insert, reaction gif, or comedic sound-effect card."
    }
}

CONFUSION_PATTERNS = [
    r"\b(wait\s+what|why\s+did|how\s+come|what\s+does\s+\w+\s+mean|confus\w*|lost\s+me|don't\s+get|makes?\s+no\s+sense)\b",
    r"\b(почему|зачем|как\s+так|в\s+смысле|не\s+понял|не\s+понятно|объясни\w*|где\s+логика|что\s+за)\b",
    r"\?{2,}"
]

SKEPTICISM_PATTERNS = [
    r"\b(doubt|fake|misleading|clickbait|cap|wrong|disagree|flawed|bs|scam|no\s+way)\b",
    r"\b(не\s+верю|чушь|ложь|бред|сомневаюсь|ошибка|неправда|врань\w*)\b"
]

ENDORSEMENT_PATTERNS = [
    r"\b(agree|true|confirmed|facts|exactly|based|legit|accurate|helpful|nailed)\b",
    r"\b(согласен|согласна|факт|точно|правда|верно|полезно|красава|база)\b"
]

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    # Russian stop words
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
    "всегда", "конечно", "всю", "между"
}


# ==============================================================================
# 2. DATA STRUCTURES & MODELS
# ==============================================================================

def format_seconds(seconds: int | float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS."""
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


def parse_timestamp_str(ts_str: str) -> float:
    """Parses 'MM:SS' or 'HH:MM:SS' or '00:01:23.456' to float seconds."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0
    return 0.0


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extracts top non-stopword alphanumeric keywords from text."""
    if not text:
        return []
    words = re.findall(r"\b[A-Za-zА-Яа-я0-9_-]{3,}\b", text.lower())
    filtered = [w for w in words if w not in STOP_WORDS and not w.isdigit()]
    if not filtered:
        return []
    counts: Dict[str, int] = {}
    for w in filtered:
        counts[w] = counts.get(w, 0) + 1
    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:top_n]]


@dataclass
class TranscriptSegment:
    """Individual spoken speech-to-text transcript unit."""
    segment_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    formatted_time: str
    text: str
    word_count: int
    speaker: Optional[str] = None
    confidence: float = 0.95
    keywords: List[str] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.start_sec

    @property
    def duration(self) -> float:
        return self.duration_sec

    @property
    def end(self) -> float:
        return self.end_sec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualFrame:
    """Sampled visual frame descriptor at a specific video playback second."""
    timestamp_sec: float
    slide_title: str
    visual_description: str
    ocr_text: str = ""
    scene_type: str = "slide_presentation"
    visual_complexity: float = 0.5
    tags: List[str] = field(default_factory=list)
    color_hex: str = "#3b82f6"

    @property
    def timestamp(self) -> float:
        return self.timestamp_sec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


VisualKeyframe = VisualFrame


@dataclass
class SpokenClaim:
    """Extracted substantive claim made during video speech."""
    claim_id: str
    timestamp_sec: float
    formatted_time: str
    claim_type: str
    claim_text: str
    endorsement_ratio: float
    skepticism_ratio: float
    controversy_score: float
    matched_comments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfusionHotspot:
    """Scene moment where audience expressed questions or confusion."""
    hotspot_id: str
    timestamp_sec: float
    formatted_time: str
    spoken_text: str
    question_count: int
    question_density: float
    confusion_rating: str  # "HIGH_CONFUSION", "MODERATE_CONFUSION", "MINOR_CURIOSITY"
    coaching_recommendation: str
    sample_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SynthesizedChapter:
    """Synthesized YouTube-compliant chapter marker."""
    timestamp_sec: int
    formatted_timestamp: str
    title: str
    rationale: str
    coherence_score: float
    confusion_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlignedSceneWindow:
    """Multi-modal synchronized temporal window."""
    window_id: int
    video_id: str
    start_sec: int
    end_sec: int
    formatted_time: str
    transcript_text: str
    spoken_keywords: List[str]
    visual_scene_type: str
    visual_complexity: float
    screen_ocr: str
    comment_count: int
    dominant_reaction: str
    reaction_breakdown: Dict[str, int]
    avg_sentiment: float
    avg_toxicity: float
    confusion_count: int
    confusion_score: float
    coherence_score: float
    cognitive_overload_flag: bool
    root_cause_phrase: Optional[str] = None
    sample_comments: List[str] = field(default_factory=list)
    suggested_chapter_title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultimodalAlignmentReport:
    """Comprehensive multimodal alignment intelligence report."""
    video_id: str
    video_title: str
    duration_sec: float
    total_segments: int
    total_words: int
    words_per_minute: float
    total_claims: int
    total_confusion_hotspots: int
    overall_coherence_score: float = 0.85
    cognitive_overload_count: int = 0
    segments: List[TranscriptSegment] = field(default_factory=list)
    claims: List[SpokenClaim] = field(default_factory=list)
    visual_frames: List[VisualFrame] = field(default_factory=list)
    aligned_moments: List[Dict[str, Any]] = field(default_factory=list)
    confusion_hotspots: List[ConfusionHotspot] = field(default_factory=list)
    chapters: List[SynthesizedChapter] = field(default_factory=list)
    aligned_windows: List[AlignedSceneWindow] = field(default_factory=list)

    @property
    def duration_secs(self) -> int:
        return int(self.duration_sec)

    @property
    def total_windows(self) -> int:
        return len(self.aligned_windows)

    @property
    def total_transcript_words(self) -> int:
        return self.total_words

    @property
    def total_keyframes(self) -> int:
        return len(self.visual_frames)

    @property
    def total_aligned_comments(self) -> int:
        return sum(w.comment_count for w in self.aligned_windows)

    @property
    def confusion_hotspots_count(self) -> int:
        return len(self.confusion_hotspots)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


MultimodalReport = MultimodalAlignmentReport


# ==============================================================================
# 3. PARSERS & MOCK GENERATORS
# ==============================================================================

def parse_vtt_or_srt(content: str) -> List[TranscriptSegment]:
    """Parses standard WebVTT or SRT subtitle strings into TranscriptSegment objects."""
    segments: List[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    seg_id = 0
    
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_line_idx = -1
        for idx, line in enumerate(lines):
            if "-->" in line:
                time_line_idx = idx
                break
        if time_line_idx == -1:
            continue
            
        time_line = lines[time_line_idx]
        parts = time_line.split("-->")
        if len(parts) != 2:
            continue
            
        start_sec = parse_timestamp_str(parts[0].strip().split()[0])
        end_sec = parse_timestamp_str(parts[1].strip().split()[0])
        duration = max(0.5, round(end_sec - start_sec, 2))
        
        text_lines = lines[time_line_idx + 1:]
        clean_text = " ".join(text_lines)
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        clean_text = html.unescape(clean_text).strip()
        
        if clean_text:
            words = clean_text.split()
            segments.append(TranscriptSegment(
                segment_id=seg_id,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration,
                formatted_time=f"{format_seconds(start_sec)} - {format_seconds(end_sec)}",
                text=clean_text,
                word_count=len(words),
                keywords=extract_keywords(clean_text)
            ))
            seg_id += 1
            
    return segments


def parse_whisper_json(data: Dict[str, Any] | List[Any]) -> List[TranscriptSegment]:
    """Parses OpenAI Whisper or faster-whisper JSON payloads."""
    segments: List[TranscriptSegment] = []
    raw_segments = data.get("segments", []) if isinstance(data, dict) else data
    seg_id = 0
    
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start + 3.0))
        duration = max(0.5, round(end - start, 2))
        text = str(item.get("text", "")).strip()
        speaker = item.get("speaker")
        confidence = float(item.get("confidence", item.get("avg_logprob", 0.95)))
        if text:
            words = text.split()
            segments.append(TranscriptSegment(
                segment_id=seg_id,
                start_sec=start,
                end_sec=end,
                duration_sec=duration,
                formatted_time=f"{format_seconds(start)} - {format_seconds(end)}",
                text=text,
                word_count=len(words),
                speaker=speaker,
                confidence=confidence,
                keywords=extract_keywords(text)
            ))
            seg_id += 1
            
    return segments


def generate_mock_transcript(video_id: str, title: str, duration_sec: int, topics: List[str]) -> List[TranscriptSegment]:
    """Generates realistic synthetic transcript segments matching video context."""
    segments: List[TranscriptSegment] = []
    topic_kw = topics[0] if topics else "System Architecture"
    clean_title = title if title else "Deep Dive Tutorial"
    
    dialogue_script = [
        ("Welcome everyone! Today we are doing a deep dive into {title}. Let's break down the foundational concepts.", 18.0),
        ("Before we begin, remember to check the GitHub repository and source code linked in the video description.", 14.0),
        ("Now let's examine the primary data pipeline and how {topic} operates under high concurrency.", 22.0),
        ("Here on the screen, notice how the vector representations are computed and normalized in memory.", 25.0),
        ("A common mistake people make is overlooking the quadratic complexity in pairwise similarity calculations.", 28.0),
        ("So if we look at this function, we pass the uncurried lambda directly into the accumulator buffer.", 24.0),
        ("Notice how the memory allocation spikes if the garbage collection threshold is left at default.", 26.0),
        ("Let's switch to the terminal and execute the performance benchmark suite to see real throughput numbers.", 30.0),
        ("As you can see from the latency output, response times dropped from 250 milliseconds down to 12 milliseconds.", 28.0),
        ("Next up, let's explore how distributed state synchronization is coordinated across worker nodes.", 32.0),
        ("This brings us to our architectural trade-offs: consistency guarantees versus network partition tolerance.", 30.0),
        ("To summarize what we've built today: we verified the core model, benchmarked latency, and optimized throughput.", 26.0),
        ("Thanks for watching! If you found this breakdown valuable, leave a comment with your questions below!", 20.0)
    ]
    
    curr_time = 0.0
    script_idx = 0
    seg_id = 0
    
    while curr_time < duration_sec:
        template, dur = dialogue_script[script_idx % len(dialogue_script)]
        text = template.format(title=clean_title, topic=topic_kw)
        if script_idx >= len(dialogue_script):
            cycle = (script_idx // len(dialogue_script)) + 1
            text += f" (Phase {cycle} optimization checkpoint)"
            
        actual_dur = min(dur, max(2.0, duration_sec - curr_time))
        end_time = round(curr_time + actual_dur, 1)
        words = text.split()
        
        segments.append(TranscriptSegment(
            segment_id=seg_id,
            start_sec=round(curr_time, 1),
            end_sec=end_time,
            duration_sec=round(actual_dur, 1),
            formatted_time=f"{format_seconds(curr_time)} - {format_seconds(end_time)}",
            text=text,
            word_count=len(words),
            keywords=extract_keywords(text)
        ))
        curr_time += actual_dur + 1.5
        script_idx += 1
        seg_id += 1
        
    return segments


def generate_mock_keyframes(video_id: str, duration_sec: int, step_sec: int = 30) -> List[VisualFrame]:
    """Generates visual keyframe states across the video timeline."""
    keyframes: List[VisualFrame] = []
    
    scene_sequence = [
        ("talking_head", 0.25, "Presenter Intro", "Speaker talking directly to camera", ["intro", "presenter"]),
        ("slide_presentation", 0.45, "Overview Agenda", "Slide deck with agenda: 1. Architecture 2. Pipeline", ["agenda", "slides"]),
        ("chart_diagram", 0.70, "Flow Diagram", "Architecture: Raw Ingest -> Feature Store -> Inference Grid", ["architecture", "flowchart"]),
        ("screen_code", 0.85, "Python Source Code", "def optimize_embeddings(corpus: np.ndarray) -> np.ndarray:", ["code", "python", "optimization"]),
        ("screen_code", 0.90, "Benchmark Terminal", "$ python -m benchmark --workers 8 --batch-size 512", ["terminal", "benchmark", "shell"]),
        ("chart_diagram", 0.65, "Performance Charts", "Benchmark Results: Latency vs Concurrency Curve", ["latency", "metrics"]),
        ("b_roll_demo", 0.50, "Telemetry Dashboard", "System Telemetry & Live Polling Dashboard", ["telemetry", "dashboard"]),
        ("talking_head", 0.30, "Outro & Takeaways", "Conclusion & Final Recommendations", ["outro", "presenter"])
    ]
    
    curr_sec = 0
    seq_idx = 0
    while curr_sec < duration_sec:
        scene_type, complexity, title, desc, tags = scene_sequence[seq_idx % len(scene_sequence)]
        color = SCENE_TYPES.get(scene_type, {}).get("color", "#3b82f6")
        
        keyframes.append(VisualFrame(
            timestamp_sec=float(curr_sec),
            slide_title=title,
            visual_description=desc,
            ocr_text=desc,
            scene_type=scene_type,
            visual_complexity=complexity,
            tags=tags,
            color_hex=color
        ))
        curr_sec += step_sec
        seq_idx += 1
        
    return keyframes


# ==============================================================================
# 4. MULTIMODAL ALIGNMENT ENGINE
# ==============================================================================

class MultimodalAlignmentEngine:
    """Core computational engine for multimodal transcript, vision and comment alignment."""

    def __init__(
        self,
        interim_dir: Optional[Path | str | Dict[str, Any]] = None,
        output_dir: Optional[Path | str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        if isinstance(interim_dir, dict):
            config_dict = interim_dir
            interim_dir = None
        self.config = config_dict or load_config()
        self.interim_dir = Path(interim_dir) if interim_dir else Path(self.config["paths"]["interim_dir"])
        self.output_dir = Path(output_dir) if output_dir else Path(self.config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._comments_df: Optional[pd.DataFrame] = None
        self._videos_df: Optional[pd.DataFrame] = None
        self._topics_df: Optional[pd.DataFrame] = None

    def _load_base_data(self) -> None:
        """Loads interim cleaned comments, videos, and topic metadata if not cached."""
        if self._comments_df is None:
            comments_path = self.interim_dir / "comments_clean.parquet"
            if comments_path.exists():
                try:
                    self._comments_df = pd.read_parquet(comments_path)
                except Exception as e:
                    logger.warning(f"Failed to read comments_clean.parquet: {e}")
                    self._comments_df = pd.DataFrame()
            else:
                self._comments_df = pd.DataFrame()
                
        if self._videos_df is None:
            videos_path = self.interim_dir / "videos_clean.parquet"
            if videos_path.exists():
                try:
                    self._videos_df = pd.read_parquet(videos_path)
                except Exception as e:
                    logger.warning(f"Failed to read videos_clean.parquet: {e}")
                    self._videos_df = pd.DataFrame()
            else:
                self._videos_df = pd.DataFrame()
                
        if self._topics_df is None:
            topics_path = self.output_dir / "topic_metadata.parquet"
            if topics_path.exists():
                try:
                    self._topics_df = pd.read_parquet(topics_path)
                except Exception as e:
                    logger.warning(f"Failed to read topic_metadata.parquet: {e}")
                    self._topics_df = pd.DataFrame()
            else:
                self._topics_df = pd.DataFrame()

    def get_available_videos(self) -> List[Dict[str, Any]]:
        """Returns list of available video candidates with titles and comment counts."""
        self._load_base_data()
        results = []
        if self._videos_df is not None and not self._videos_df.empty:
            for _, row in self._videos_df.iterrows():
                vid = str(row.get("video_id", ""))
                title = str(row.get("title", vid))
                duration = int(row.get("duration", 600)) if pd.notna(row.get("duration")) else 600
                comment_count = int(row.get("comment_count", 0)) if pd.notna(row.get("comment_count")) else 0
                results.append({
                    "video_id": vid,
                    "title": title,
                    "duration": duration,
                    "comment_count": comment_count
                })
        elif self._comments_df is not None and not self._comments_df.empty and "video_id" in self._comments_df.columns:
            counts = self._comments_df["video_id"].value_counts()
            for vid, count in counts.items():
                results.append({
                    "video_id": str(vid),
                    "title": f"Video {vid}",
                    "duration": 600,
                    "comment_count": int(count)
                })
        else:
            results.append({
                "video_id": "mock_demo_vid_01",
                "title": "Deep Learning Architecture & Embedding Vectors Tutorial",
                "duration": 720,
                "comment_count": 85
            })
        return results

    def _extract_video_comments_with_timestamps(self, video_id: str) -> pd.DataFrame:
        """Extracts and parses comments mentioning timestamps for the specified video."""
        self._load_base_data()
        if self._comments_df is None or self._comments_df.empty:
            return pd.DataFrame()
            
        df = self._comments_df[self._comments_df["video_id"] == video_id].copy()
        if df.empty:
            return pd.DataFrame()
            
        text_col = "text" if "text" in df.columns else ("text_original" if "text_original" in df.columns else None)
        if not text_col:
            return pd.DataFrame()
            
        ts_regex = re.compile(
            r"(?:(?:https?://[^\s]*[?&]t=(\d+)s?)|(?:\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b))",
            re.IGNORECASE
        )
        
        extracted_rows = []
        for _, row in df.iterrows():
            text_str = str(row[text_col])
            cid = str(row.get("comment_id", ""))
            author = str(row.get("author_name", row.get("channel_id", "Viewer")))
            vader = float(row.get("vader_compound", 0.0)) if pd.notna(row.get("vader_compound")) else 0.0
            tox = float(row.get("toxicity", 0.0)) if pd.notna(row.get("toxicity")) else 0.0
            likes = int(row.get("like_count", 0)) if pd.notna(row.get("like_count")) else 0
            
            for match in ts_regex.finditer(text_str):
                url_t = match.group(1)
                if url_t:
                    sec = int(url_t)
                else:
                    h = int(match.group(2)) if match.group(2) else 0
                    m = int(match.group(3)) if match.group(3) else 0
                    s = int(match.group(4)) if match.group(4) else 0
                    sec = h * 3600 + m * 60 + s
                    
                is_question = any(re.search(pat, text_str, re.IGNORECASE) for pat in CONFUSION_PATTERNS)
                is_skeptic = any(re.search(pat, text_str, re.IGNORECASE) for pat in SKEPTICISM_PATTERNS)
                is_endorse = any(re.search(pat, text_str, re.IGNORECASE) for pat in ENDORSEMENT_PATTERNS)
                
                extracted_rows.append({
                    "comment_id": cid,
                    "video_id": video_id,
                    "author": author,
                    "second": sec,
                    "text": text_str,
                    "vader_compound": vader,
                    "toxicity": tox,
                    "like_count": likes,
                    "is_question": is_question,
                    "is_skeptic": is_skeptic,
                    "is_endorse": is_endorse
                })
                
        return pd.DataFrame(extracted_rows)

    def generate_alignment_report(
        self,
        video_id: str,
        transcript_path: Optional[str | Path] = None,
        window_sec: float = 15.0,
        controversy_threshold: float = 0.3,
        mock: bool = True
    ) -> MultimodalAlignmentReport:
        """Streamlit-facing alignment engine producing complete MultimodalAlignmentReport."""
        return self.analyze_video(
            video_id=video_id,
            step_secs=int(window_sec),
            transcript_path=transcript_path,
            force_mock=mock
        )

    def analyze_video(
        self,
        video_id: str,
        step_secs: int = 30,
        transcript_path: Optional[str | Path] = None,
        keyframes_path: Optional[str | Path] = None,
        force_mock: bool = False
    ) -> MultimodalAlignmentReport:
        """Executes full multimodal alignment, coherence modeling, and chapter generation."""
        self._load_base_data()
        
        # 1. Resolve Video Metadata
        video_title = f"Video {video_id}"
        duration_sec = 600
        topics = ["Neural Embeddings", "Vector Search"]
        
        if self._videos_df is not None and not self._videos_df.empty:
            v_match = self._videos_df[self._videos_df["video_id"] == video_id]
            if not v_match.empty:
                r = v_match.iloc[0]
                video_title = str(r.get("title", video_title))
                duration_sec = int(r.get("duration", duration_sec)) if pd.notna(r.get("duration")) else duration_sec
                if duration_sec <= 0:
                    duration_sec = 600
                    
        if self._topics_df is not None and not self._topics_df.empty and "topic_name" in self._topics_df.columns:
            topics = self._topics_df["topic_name"].dropna().tolist()[:3]
            
        # 2. Ingest or Generate Transcripts
        transcript_segments: List[TranscriptSegment] = []
        if transcript_path and Path(transcript_path).exists() and not force_mock:
            p = Path(transcript_path)
            content = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix.lower() in [".vtt", ".srt"]:
                transcript_segments = parse_vtt_or_srt(content)
            elif p.suffix.lower() == ".json":
                try:
                    data = json.loads(content)
                    transcript_segments = parse_whisper_json(data)
                except Exception as e:
                    logger.warning(f"Error parsing json transcript: {e}")
                    
        if not transcript_segments:
            transcript_segments = generate_mock_transcript(video_id, video_title, duration_sec, topics)
            
        # 3. Ingest or Generate Keyframes
        keyframes: List[VisualFrame] = []
        if keyframes_path and Path(keyframes_path).exists() and not force_mock:
            try:
                k_data = json.loads(Path(keyframes_path).read_text(encoding="utf-8", errors="replace"))
                for k in k_data:
                    keyframes.append(VisualFrame(
                        timestamp_sec=float(k.get("timestamp_sec", k.get("timestamp", 0))),
                        slide_title=str(k.get("slide_title", "Scene")),
                        visual_description=str(k.get("visual_description", "")),
                        ocr_text=str(k.get("ocr_text", k.get("text_ocr", ""))),
                        scene_type=str(k.get("scene_type", "screen_code")),
                        visual_complexity=float(k.get("visual_complexity", 0.5)),
                        tags=list(k.get("tags", [])),
                        color_hex=SCENE_TYPES.get(k.get("scene_type", ""), {}).get("color", "#3b82f6")
                    ))
            except Exception as e:
                logger.warning(f"Error parsing keyframes json: {e}")
                
        if not keyframes:
            keyframes = generate_mock_keyframes(video_id, duration_sec, step_secs)
            
        # 4. Extract Timestamped Comments
        comments_ts_df = self._extract_video_comments_with_timestamps(video_id)
        
        # 5. Temporal Alignment into Binned Scene Windows & Moments
        num_windows = max(1, math.ceil(duration_sec / step_secs))
        aligned_windows: List[AlignedSceneWindow] = []
        aligned_moments: List[Dict[str, Any]] = []
        confusion_hotspots: List[ConfusionHotspot] = []
        claims: List[SpokenClaim] = []
        
        cognitive_overload_count = 0
        coherence_sum = 0.0
        
        for w_idx in range(num_windows):
            w_start = w_idx * step_secs
            w_end = min(duration_sec, (w_idx + 1) * step_secs)
            time_str = f"{format_seconds(w_start)} - {format_seconds(w_end)}"
            
            # Transcript overlap
            t_texts = []
            t_keywords = []
            matched_seg_ids = []
            for seg in transcript_segments:
                if (seg.start_sec < w_end) and (seg.end_sec > w_start):
                    t_texts.append(seg.text)
                    t_keywords.extend(seg.keywords)
                    matched_seg_ids.append(seg.segment_id)
            w_transcript = " ".join(t_texts)
            spoken_kw = list(dict.fromkeys(t_keywords))[:8]
            
            # Active Keyframe mapping
            active_keyframes = [kf for kf in keyframes if w_start <= kf.timestamp_sec < w_end]
            if not active_keyframes:
                closest_kf = min(keyframes, key=lambda kf: abs(kf.timestamp_sec - (w_start + w_end) / 2))
                active_keyframes = [closest_kf]
                
            dominant_kf = active_keyframes[0]
            scene_type = dominant_kf.scene_type
            complexity = round(float(np.mean([kf.visual_complexity for kf in active_keyframes])), 2)
            ocr_text = " | ".join([kf.ocr_text for kf in active_keyframes if kf.ocr_text])
            
            # Comment overlap
            w_comments = []
            w_matched_comment_dicts = []
            w_vader = 0.0
            w_tox = 0.0
            w_confusion = 0
            w_skeptic = 0
            w_endorse = 0
            w_reactions: Dict[str, int] = {
                "humor_laughter": 0,
                "shock_surprise": 0,
                "emotional_touching": 0,
                "critique_analytical": 0,
                "chapter_navigation": 0
            }
            
            if not comments_ts_df.empty:
                c_slice = comments_ts_df[(comments_ts_df["second"] >= w_start) & (comments_ts_df["second"] < w_end)]
                if not c_slice.empty:
                    w_comments = c_slice["text"].tolist()[:5]
                    w_vader = float(c_slice["vader_compound"].mean())
                    w_tox = float(c_slice["toxicity"].mean())
                    w_confusion = int(c_slice["is_question"].sum())
                    w_skeptic = int(c_slice["is_skeptic"].sum())
                    w_endorse = int(c_slice["is_endorse"].sum())
                    
                    for _, crow in c_slice.iterrows():
                        txt = str(crow["text"])
                        w_matched_comment_dicts.append({
                            "author": crow["author"],
                            "text": txt,
                            "sentiment": crow["vader_compound"],
                            "likes": crow["like_count"]
                        })
                        txt_l = txt.lower()
                        if any(re.search(p, txt_l) for p in [r"[😂🤣😆]", r"lol|lmao|смешн"]):
                            w_reactions["humor_laughter"] += 1
                        elif any(re.search(p, txt_l) for p in [r"[😱🤯]", r"wtf|omg|шок|жесть"]):
                            w_reactions["shock_surprise"] += 1
                        elif any(re.search(p, txt_l) for p in [r"[❤️🥺]", r"душевн|wholesome|трогательн"]):
                            w_reactions["emotional_touching"] += 1
                        elif any(re.search(p, txt_l) for p in [r"[🤔🧐]", r"почему|логика|сюжет"]):
                            w_reactions["critique_analytical"] += 1
                        elif any(re.search(p, txt_l) for p in [r"[⏱️🕒]", r"таймкод|трек|timestamp"]):
                            w_reactions["chapter_navigation"] += 1
            else:
                if w_idx % 4 == 1:
                    w_reactions["critique_analytical"] = 3
                    w_confusion = 2
                    w_skeptic = 1
                    w_comments = ["Wait, why did the memory buffer overflow here?", "Can someone explain this step?"]
                    w_matched_comment_dicts = [
                        {"author": "TechExplorer", "text": "Wait, why did the memory buffer overflow here?", "sentiment": -0.3, "likes": 4},
                        {"author": "DevLearner", "text": "Can someone explain this step?", "sentiment": -0.1, "likes": 1}
                    ]
                elif w_idx % 4 == 2:
                    w_reactions["humor_laughter"] = 4
                    w_endorse = 3
                    w_comments = ["haha that error was hilarious 😂", "classic typo moment"]
                    w_matched_comment_dicts = [
                        {"author": "LaughingCode", "text": "haha that error was hilarious 😂", "sentiment": 0.8, "likes": 12}
                    ]
                else:
                    w_reactions["chapter_navigation"] = 1
                    
            c_count = len(w_matched_comment_dicts)
            dom_reaction = max(w_reactions.items(), key=lambda x: x[1])[0] if sum(w_reactions.values()) > 0 else "neutral_observation"
            confusion_score = round(min(1.0, w_confusion / max(1, c_count)), 2)
            skep_ratio = round(min(1.0, w_skeptic / max(1, c_count)), 2)
            
            # Coherence scoring
            coherence = 0.85
            ocr_kw = extract_keywords(ocr_text)
            overlap_kw = set(spoken_kw).intersection(set(ocr_kw))
            if overlap_kw:
                coherence += 0.08
                
            if complexity >= 0.70 and confusion_score >= 0.35:
                coherence -= 0.30
                cog_flag = True
                cognitive_overload_count += 1
            else:
                cog_flag = False
                
            coherence = round(float(np.clip(coherence - (confusion_score * 0.20), 0.10, 1.0)), 2)
            coherence_sum += coherence
            
            root_cause = None
            if confusion_score > 0.3 or cog_flag:
                sentences = re.split(r"[.!?]\s+", w_transcript)
                if sentences and len(sentences[0]) > 10:
                    root_cause = sentences[0].strip()
                else:
                    root_cause = f"Dense technical transition at {format_seconds(w_start)}"
                    
                # Create ConfusionHotspot
                rating = "HIGH_CONFUSION" if confusion_score >= 0.5 else "MODERATE_CONFUSION"
                rec = (
                    f"Consider adding a pinned comment or YouTube Chapter Title clarifying '{spoken_kw[0] if spoken_kw else 'this concept'}' "
                    f"to de-escalate viewer perplexity."
                )
                confusion_hotspots.append(ConfusionHotspot(
                    hotspot_id=f"hs_{w_idx}",
                    timestamp_sec=float(w_start),
                    formatted_time=format_seconds(w_start),
                    spoken_text=root_cause,
                    question_count=w_confusion,
                    question_density=confusion_score,
                    confusion_rating=rating,
                    coaching_recommendation=rec,
                    sample_questions=w_comments[:2]
                ))
                
            chapter_title = None
            if w_idx == 0:
                chapter_title = "00:00 Introduction & Overview"
            elif ocr_kw:
                chapter_title = f"{format_seconds(w_start)} {ocr_kw[0].capitalize()} Breakdown"
            elif spoken_kw:
                chapter_title = f"{format_seconds(w_start)} {spoken_kw[0].capitalize()} Analysis"
            else:
                chapter_title = f"{format_seconds(w_start)} Scene {w_idx + 1}"
                
            aligned_windows.append(AlignedSceneWindow(
                window_id=w_idx,
                video_id=video_id,
                start_sec=w_start,
                end_sec=w_end,
                formatted_time=time_str,
                transcript_text=w_transcript,
                spoken_keywords=spoken_kw,
                visual_scene_type=scene_type,
                visual_complexity=complexity,
                screen_ocr=ocr_text,
                comment_count=c_count,
                dominant_reaction=dom_reaction,
                reaction_breakdown=w_reactions,
                avg_sentiment=round(w_vader, 3),
                avg_toxicity=round(w_tox, 3),
                confusion_count=w_confusion,
                confusion_score=confusion_score,
                coherence_score=coherence,
                cognitive_overload_flag=cog_flag,
                root_cause_phrase=root_cause,
                sample_comments=w_comments,
                suggested_chapter_title=chapter_title
            ))
            
            # Aligned Moment structure for Streamlit subplots
            aligned_moments.append({
                "start_sec": w_start,
                "end_sec": w_end,
                "segment_id": matched_seg_ids[0] if matched_seg_ids else w_idx,
                "average_sentiment": round(w_vader, 3),
                "skepticism_ratio": skep_ratio,
                "matched_comment_count": c_count,
                "matched_comments": w_matched_comment_dicts
            })
            
            # Synthesize Spoken Claims on key scenes
            if w_idx % 2 == 1:
                claim_types_list = list(CLAIM_TYPES.keys())
                ctype = claim_types_list[(w_idx // 2) % len(claim_types_list)]
                ctext = w_transcript[:90] if w_transcript else f"Key algorithmic claim at {format_seconds(w_start)}"
                claims.append(SpokenClaim(
                    claim_id=f"claim_{w_idx}",
                    timestamp_sec=float(w_start),
                    formatted_time=format_seconds(w_start),
                    claim_type=ctype,
                    claim_text=ctext,
                    endorsement_ratio=round(min(1.0, w_endorse / max(1, c_count)), 2),
                    skepticism_ratio=skep_ratio,
                    controversy_score=round((skep_ratio + (1.0 - coherence)) / 2.0, 2),
                    matched_comments=w_matched_comment_dicts
                ))
                
        avg_coherence = round(coherence_sum / max(1, num_windows), 2)
        chapters = self._synthesize_chapters(aligned_windows, duration_sec)
        total_words = sum(s.word_count for s in transcript_segments)
        wpm = round(total_words / max(0.1, duration_sec / 60.0), 1)
        
        report = MultimodalAlignmentReport(
            video_id=video_id,
            video_title=video_title,
            duration_sec=float(duration_sec),
            total_segments=len(transcript_segments),
            total_words=total_words,
            words_per_minute=wpm,
            total_claims=len(claims),
            total_confusion_hotspots=len(confusion_hotspots),
            overall_coherence_score=avg_coherence,
            cognitive_overload_count=cognitive_overload_count,
            segments=transcript_segments,
            claims=claims,
            visual_frames=keyframes,
            aligned_moments=aligned_moments,
            confusion_hotspots=confusion_hotspots,
            chapters=chapters,
            aligned_windows=aligned_windows
        )
        
        self.export_parquet(report)
        return report

    def _synthesize_chapters(self, windows: List[AlignedSceneWindow], duration_sec: int) -> List[SynthesizedChapter]:
        """Clusters scene transitions into 5–10 high-value YouTube chapters."""
        chapters: List[SynthesizedChapter] = []
        first_win = windows[0]
        first_kw = first_win.spoken_keywords[0] if first_win.spoken_keywords else "Introduction"
        chapters.append(SynthesizedChapter(
            timestamp_sec=0,
            formatted_timestamp="00:00",
            title=f"Introduction & {first_kw.capitalize()}",
            rationale="Opening hook and agenda overview",
            coherence_score=first_win.coherence_score,
            confusion_level="Low"
        ))
        
        min_chapter_interval = max(60, duration_sec // 8)
        last_sec = 0
        
        for win in windows[1:]:
            if (win.start_sec - last_sec) >= min_chapter_interval:
                kw_title = win.spoken_keywords[0].capitalize() if win.spoken_keywords else f"Section at {win.formatted_time}"
                if win.screen_ocr:
                    kw_title = win.screen_ocr.split(":")[0].strip()
                    
                conf_lvl = "High" if win.confusion_score >= 0.5 else ("Moderate" if win.confusion_score >= 0.25 else "Low")
                
                chapters.append(SynthesizedChapter(
                    timestamp_sec=win.start_sec,
                    formatted_timestamp=format_seconds(win.start_sec),
                    title=f"{kw_title}",
                    rationale=f"Scene shift to {SCENE_TYPES.get(win.visual_scene_type, {}).get('label', win.visual_scene_type)}",
                    coherence_score=win.coherence_score,
                    confusion_level=conf_lvl
                ))
                last_sec = win.start_sec
                
        return chapters

    def export_parquet(self, report: MultimodalAlignmentReport) -> Tuple[Path, Path]:
        """Saves aligned windows and synthesized chapters to Parquet files."""
        win_rows = []
        for w in report.aligned_windows:
            win_rows.append({
                "video_id": w.video_id,
                "window_id": w.window_id,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "formatted_time": w.formatted_time,
                "transcript_snippet": w.transcript_text[:150],
                "spoken_keywords": ", ".join(w.spoken_keywords),
                "visual_scene_type": w.visual_scene_type,
                "visual_complexity": w.visual_complexity,
                "screen_ocr": w.screen_ocr,
                "comment_count": w.comment_count,
                "dominant_reaction": w.dominant_reaction,
                "avg_sentiment": w.avg_sentiment,
                "avg_toxicity": w.avg_toxicity,
                "confusion_score": w.confusion_score,
                "coherence_score": w.coherence_score,
                "cognitive_overload_flag": w.cognitive_overload_flag,
                "root_cause_phrase": w.root_cause_phrase or "",
                "sample_comments": " || ".join(w.sample_comments)
            })
        win_df = pd.DataFrame(win_rows)
        win_path = self.output_dir / "multimodal_alignment.parquet"
        win_df.to_parquet(win_path, index=False)
        
        chap_rows = []
        for ch in report.chapters:
            chap_rows.append({
                "video_id": report.video_id,
                "timestamp_sec": ch.timestamp_sec,
                "formatted_timestamp": ch.formatted_timestamp,
                "title": ch.title,
                "rationale": ch.rationale,
                "coherence_score": ch.coherence_score,
                "confusion_level": ch.confusion_level
            })
        chap_df = pd.DataFrame(chap_rows)
        chap_path = self.output_dir / "multimodal_chapters.parquet"
        chap_df.to_parquet(chap_path, index=False)
        
        return win_path, chap_path

    def export_json(self, report: MultimodalAlignmentReport, out_file: Path | str) -> Path:
        """Serializes report to structured JSON format."""
        out_p = Path(out_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(report)
        out_p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_p


MultimodalEngine = MultimodalAlignmentEngine


# ==============================================================================
# 5. CLI INTERFACE (ytint-multimodal)
# ==============================================================================

def main():
    """CLI entrypoint for ytint-multimodal."""
    parser = argparse.ArgumentParser(
        description="ytint-multimodal // Multimodal Transcript & Vision Alignment Engine"
    )
    parser.add_argument("--status", action="store_true", help="Print available videos and multimodal status")
    parser.add_argument("--video-id", type=str, default="", help="Target video ID to align")
    parser.add_argument("--step-secs", type=int, default=30, help="Temporal scene window length in seconds (default: 30)")
    parser.add_argument("--transcript", type=str, default=None, help="Path to transcript file (.vtt, .srt, .json)")
    parser.add_argument("--keyframes", type=str, default=None, help="Path to visual keyframes metadata file (.json)")
    parser.add_argument("--align", action="store_true", help="Execute alignment and print scene ledger")
    parser.add_argument("--chapters", action="store_true", help="Synthesize and print formatted YouTube chapters")
    parser.add_argument("--coherence", action="store_true", help="Display cross-modal coherence scorecard")
    parser.add_argument("--hotspots", action="store_true", help="Extract viewer confusion hotspots and root causes")
    parser.add_argument("--mock", action="store_true", help="Force synthetic transcript & keyframe simulation")
    parser.add_argument("--export", type=str, default=None, help="Export alignment dossier to JSON file path")
    
    args = parser.parse_args()
    engine = MultimodalAlignmentEngine()
    
    if args.status:
        vids = engine.get_available_videos()
        print("🎬 ========================================================")
        print("   ytint-multimodal // Multimodal Alignment Engine Status")
        print("============================================================")
        print(f"Total Available Videos: {len(vids)}")
        for v in vids[:8]:
            print(f"  • [{v['video_id']}] {v['title'][:50]}... ({v['comment_count']} comments, {v['duration']}s)")
        print("\nSupported Scene Types:")
        for k, v in SCENE_TYPES.items():
            print(f"  {v['label']} (Color: {v['color']})")
        print("\nSupported Claim Types:")
        for k, v in CLAIM_TYPES.items():
            print(f"  • {k}: {v}")
        return
        
    vids = engine.get_available_videos()
    target_vid = args.video_id
    if not target_vid:
        if vids:
            target_vid = vids[0]["video_id"]
        else:
            target_vid = "mock_demo_vid_01"
            
    print(f"🚀 Aligning Multimodal Media for Video [{target_vid}]...")
    report = engine.analyze_video(
        video_id=target_vid,
        step_secs=args.step_secs,
        transcript_path=args.transcript,
        keyframes_path=args.keyframes,
        force_mock=args.mock
    )
    
    print("\n📊 ========================================================")
    print(f"   MULTIMODAL DOSSIER: {report.video_title}")
    print("============================================================")
    print(f"Duration: {format_seconds(report.duration_secs)} | Windows: {report.total_windows} ({args.step_secs}s step)")
    print(f"Spoken Words: {report.total_transcript_words} | Cadence: {report.words_per_minute} WPM")
    print(f"Cross-Modal Coherence Score: {report.overall_coherence_score * 100:.1f}%")
    print(f"Cognitive Overload Episodes: {report.cognitive_overload_count}")
    print(f"Audience Confusion Hotspots: {report.confusion_hotspots_count}")
    print(f"Substantive Spoken Claims: {report.total_claims}")
    
    if args.chapters or not (args.align or args.coherence or args.hotspots):
        print("\n🔖 SYNTHESIZED YOUTUBE CHAPTERS:")
        print("------------------------------------------------------------")
        for ch in report.chapters:
            print(f"{ch.formatted_timestamp} {ch.title}  (Coherence: {ch.coherence_score * 100:.0f}%, Confusion: {ch.confusion_level})")
            
    if args.coherence or args.align:
        print("\n⏱️ SCENE WINDOW ALIGNMENT LEDGER:")
        print("------------------------------------------------------------")
        for w in report.aligned_windows[:10]:
            flag = " ⚠️ [OVERLOAD]" if w.cognitive_overload_flag else ""
            print(f"[{w.formatted_time}] {SCENE_TYPES.get(w.visual_scene_type, {}).get('label', w.visual_scene_type)}")
            print(f"  🗣️ Audio: \"{w.transcript_text[:70]}...\"")
            print(f"  📊 Coherence: {w.coherence_score * 100:.0f}% | Complexity: {w.visual_complexity * 100:.0f}% | Confusion: {w.confusion_score * 100:.0f}%{flag}")
            if w.root_cause_phrase:
                print(f"  🔍 Root Cause: \"{w.root_cause_phrase}\"")
                
    if args.hotspots:
        print("\n🔍 VIEWER CONFUSION HOTSPOTS & ROOT-CAUSE ATTRIBUTION:")
        print("------------------------------------------------------------")
        if report.confusion_hotspots:
            for h in report.confusion_hotspots:
                print(f"• Hotspot at [{h.formatted_time}] ({h.confusion_rating}, Density: {h.question_density * 100:.0f}%)")
                print(f"  Spoken Context: \"{h.spoken_text}\"")
                print(f"  💡 Coaching: {h.coaching_recommendation}")
                if h.sample_questions:
                    print(f"  Viewer Quotes: \"{h.sample_questions[0]}\"")
        else:
            print("  No high-confusion hotspots detected across video timeline.")
            
    if args.export:
        out_path = engine.export_json(report, args.export)
        print(f"\n✅ Multimodal dossier successfully exported to: {out_path}")


if __name__ == "__main__":
    main()
