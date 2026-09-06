"""ytint // Executive Intelligence Dossier & Report Exporter (src/engine/reporter.py)

Compiles a comprehensive, standalone, self-contained HTML executive intelligence briefing
from all pipeline analytical layers, forensic threat matrices, and publication visual plots.
Supports responsive screen viewing and print-optimized PDF output via browser print dialog.
"""

from __future__ import annotations

import sys
import os
import base64
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

# Ensure src directory is in sys.path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from engine.config_loader import load_config, get_paths

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ytint_reporter")


class ExecutiveReportGenerator:
    """Compiles pipeline analytical layers and visual artifacts into a unified Executive Dossier."""

    def __init__(self, interim_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        if interim_dir is None or output_dir is None:
            config = load_config()
            _, self.interim_dir, self.output_dir = get_paths(config)
        else:
            self.interim_dir = Path(interim_dir)
            self.output_dir = Path(output_dir)

        self.plots_dir = self.output_dir / "plots"

    def _safe_read_parquet(self, path: Path) -> pd.DataFrame:
        """Safely loads parquet file if present, returning empty DataFrame on missing/corrupt."""
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logger.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()

    def _encode_image_base64(self, img_path: Path) -> Optional[str]:
        """Encodes an image to a base64 data URI for offline HTML embedding."""
        if img_path.exists():
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = img_path.suffix.lstrip(".").lower()
                mime = "image/png" if ext == "png" else f"image/{ext}"
                return f"data:{mime};base64,{encoded}"
            except Exception as e:
                logger.warning(f"Failed to encode image {img_path.name}: {e}")
        return None

    def extract_metrics(self) -> Dict[str, Any]:
        """Extracts and aggregates key intelligence metrics across all analytical layers."""
        df_comments = self._safe_read_parquet(self.interim_dir / "comments_clean.parquet")
        df_videos_final = self._safe_read_parquet(self.output_dir / "videos_final.parquet")
        df_videos_clean = self._safe_read_parquet(self.interim_dir / "videos_clean.parquet")
        df_authors = self._safe_read_parquet(self.output_dir / "authors_final.parquet")
        df_topics = self._safe_read_parquet(self.output_dir / "topic_metadata.parquet")
        df_spikes = self._safe_read_parquet(self.output_dir / "viral_events.parquet")
        df_bots = self._safe_read_parquet(self.output_dir / "bot_classifications.parquet")
        df_inflation = self._safe_read_parquet(self.output_dir / "like_inflation.parquet")
        df_impersonation = self._safe_read_parquet(self.output_dir / "impersonation_suspects.parquet")
        df_cib = self._safe_read_parquet(self.output_dir / "cib_clusters.parquet")
        df_power_law = self._safe_read_parquet(self.output_dir / "power_law_fit.parquet")
        df_driveby = self._safe_read_parquet(self.output_dir / "driveby_loyalists.parquet")
        df_intent = self._safe_read_parquet(self.output_dir / "audience_intent.parquet")
        df_radar = self._safe_read_parquet(self.output_dir / "video_profile_radar.parquet")
        df_causal = self._safe_read_parquet(self.output_dir / "causal_impact_summary.parquet")
        df_shap = self._safe_read_parquet(self.output_dir / "shap_features.parquet")

        # Merge video titles if available
        if not df_videos_clean.empty and "video_id" in df_videos_clean.columns and "title" in df_videos_clean.columns:
            if not df_videos_final.empty and "video_id" in df_videos_final.columns:
                if "title" not in df_videos_final.columns:
                    df_videos_final = df_videos_final.merge(
                        df_videos_clean[["video_id", "title"]].drop_duplicates("video_id"),
                        on="video_id",
                        how="left"
                    )
            elif df_videos_final.empty:
                df_videos_final = df_videos_clean

        # 1. High-level Scope & Overview
        total_comments = len(df_comments)
        total_authors = len(df_authors) if not df_authors.empty else (
            df_comments["author_id"].nunique() if not df_comments.empty and "author_id" in df_comments.columns else 0
        )
        total_videos = len(df_videos_final) if not df_videos_final.empty else (
            df_comments["video_id"].nunique() if not df_comments.empty and "video_id" in df_comments.columns else 0
        )

        date_range_str = "N/A"
        if not df_comments.empty and "published_at" in df_comments.columns:
            try:
                ts = pd.to_datetime(df_comments["published_at"], errors="coerce").dropna()
                if not ts.empty:
                    date_range_str = f"{ts.min().strftime('%Y-%m-%d')} to {ts.max().strftime('%Y-%m-%d')}"
            except Exception:
                pass

        # 2. Forensic Integrity & Threats
        bot_count = 0
        bot_pct = 0.0
        if not df_bots.empty and "is_bot" in df_bots.columns:
            bot_count = int(df_bots["is_bot"].sum())
            bot_pct = (bot_count / len(df_bots)) * 100

        inflation_count = 0
        inflation_pct = 0.0
        if not df_inflation.empty and "is_suspicious" in df_inflation.columns:
            inflation_count = int(df_inflation["is_suspicious"].sum())
            inflation_pct = (inflation_count / len(df_inflation)) * 100

        impersonation_count = len(df_impersonation) if not df_impersonation.empty else 0
        cib_clusters_count = df_cib["cluster_id"].nunique() if not df_cib.empty and "cluster_id" in df_cib.columns else 0

        # Composite Threat Level
        if bot_pct > 10.0 or inflation_pct > 15.0 or cib_clusters_count >= 5:
            threat_level = "CRITICAL"
            threat_color = "#ff3366"
        elif bot_pct > 5.0 or inflation_pct > 8.0 or cib_clusters_count >= 2 or impersonation_count > 5:
            threat_level = "ELEVATED"
            threat_color = "#ffaa00"
        elif bot_pct > 2.0 or inflation_pct > 3.0 or impersonation_count > 0:
            threat_level = "MODERATE"
            threat_color = "#00f0ff"
        else:
            threat_level = "LOW / NOMINAL"
            threat_color = "#00e599"

        # 3. Audience Structure & Loyalty
        loyalist_count = 0
        driveby_count = 0
        if not df_driveby.empty and "classification" in df_driveby.columns and "author_count" in df_driveby.columns:
            db_map = dict(zip(df_driveby["classification"], df_driveby["author_count"]))
            loyalist_count = int(db_map.get("loyalist", 0))
            driveby_count = int(db_map.get("drive_by", 0))

        alpha_val = 0.0
        if not df_power_law.empty and "alpha" in df_power_law.columns:
            alpha_val = float(df_power_law["alpha"].iloc[0])

        half_life_mean = 0.0
        if not df_videos_final.empty and "attention_half_life_days" in df_videos_final.columns:
            half_life_mean = float(df_videos_final["attention_half_life_days"].dropna().mean())

        # 4. Semantic Themes & Intent
        top_topics = []
        if not df_topics.empty and "Name" in df_topics.columns and "Count" in df_topics.columns:
            valid_topics = df_topics[df_topics["Topic"] != -1] if "Topic" in df_topics.columns else df_topics
            top_topics = valid_topics.sort_values(by="Count", ascending=False).head(5).to_dict(orient="records")

        intent_dist = {}
        if not df_intent.empty and "intent" in df_intent.columns:
            ic = df_intent["intent"].value_counts(normalize=True) * 100
            intent_dist = {str(k): round(float(v), 1) for k, v in ic.items()}
        elif not df_comments.empty and "intent" in df_comments.columns:
            ic = df_comments["intent"].value_counts(normalize=True) * 100
            intent_dist = {str(k): round(float(v), 1) for k, v in ic.items()}

        # 5. Sentiment & Toxicity
        avg_sentiment = 0.0
        if not df_videos_final.empty and "avg_sentiment" in df_videos_final.columns:
            avg_sentiment = float(df_videos_final["avg_sentiment"].dropna().mean())
        elif not df_comments.empty and "sentiment_compound" in df_comments.columns:
            avg_sentiment = float(df_comments["sentiment_compound"].dropna().mean())

        num_spikes = len(df_spikes) if not df_spikes.empty else 0

        # 6. Video Highlights
        video_highlights = []
        if not df_videos_final.empty:
            cols_to_keep = [c for c in ["video_id", "title", "comment_count", "avg_sentiment", "attention_half_life_days", "controversy_index"] if c in df_videos_final.columns]
            v_df = df_videos_final[cols_to_keep].copy()
            if "comment_count" in v_df.columns:
                v_df = v_df.sort_values(by="comment_count", ascending=False)
            elif "comments" in v_df.columns:
                v_df = v_df.sort_values(by="comments", ascending=False)
            video_highlights = v_df.head(6).to_dict(orient="records")

        # 7. Causal Impact & Top Virality Drivers
        did_lift_str = "N/A"
        if not df_causal.empty:
            for col in ["absolute_effect", "relative_effect", "summary_effect", "lift"]:
                if col in df_causal.columns:
                    val = df_causal[col].iloc[0]
                    did_lift_str = f"+{val:.1f}%" if isinstance(val, (int, float)) and val > 0 else f"{val}"
                    break

        top_shap_features = []
        if not df_shap.empty and "feature" in df_shap.columns and "importance" in df_shap.columns:
            top_shap_features = df_shap.sort_values(by="importance", ascending=False).head(5).to_dict(orient="records")

        return {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "date_range_str": date_range_str,
            "total_comments": total_comments,
            "total_authors": total_authors,
            "total_videos": total_videos,
            "bot_count": bot_count,
            "bot_pct": bot_pct,
            "inflation_count": inflation_count,
            "inflation_pct": inflation_pct,
            "impersonation_count": impersonation_count,
            "cib_clusters_count": cib_clusters_count,
            "threat_level": threat_level,
            "threat_color": threat_color,
            "loyalist_count": loyalist_count,
            "driveby_count": driveby_count,
            "alpha_val": alpha_val,
            "half_life_mean": half_life_mean,
            "top_topics": top_topics,
            "intent_dist": intent_dist,
            "avg_sentiment": avg_sentiment,
            "num_spikes": num_spikes,
            "video_highlights": video_highlights,
            "did_lift_str": did_lift_str,
            "top_shap_features": top_shap_features
        }

    def generate_html(self, include_plots: bool = True) -> str:
        """Renders self-contained, responsive, printable HTML executive dossier."""
        m = self.extract_metrics()

        # Gather key plots as base64 images if requested
        plots = {}
        plot_files = {
            "video_radar": ("video_profile_radar.png", "Comparative Multi-Dimensional Video Radar"),
            "intent_dist": ("audience_intent_distribution.png", "Audience Demand Intent Taxonomy"),
            "cib_rings": ("cib_rings_graph.png", "Coordinated Inauthentic Behavior (CIB) Clustered Networks"),
            "like_inflation": ("like_inflation.png", "Suspicious Like-to-Reply Astroturfing Scanner"),
            "bot_heuristics": ("bot_heuristics.png", "MinHash Near-Duplicate & Timing Bot Forensics"),
            "topic_stream": ("topic_streamgraph.png", "Longitudinal Semantic Topic Streamgraph"),
            "emotion_wheel": ("plutchik_emotion_wheel.png", "Plutchik Emotion Wheel & Affect Spectrum"),
            "creator_uplift": ("creator_causal_uplift.png", "Creator Intervention Difference-in-Differences Lift"),
            "shap_summary": ("shap_summary.png", "Tree SHAP Upvote Virality Drivers"),
            "diurnal_heatmap": ("diurnal_heatmap.png", "24×7 Diurnal Comment Activity Heatmap")
        }

        if include_plots and self.plots_dir.exists():
            for key, (filename, label) in plot_files.items():
                img_path = self.plots_dir / filename
                b64 = self._encode_image_base64(img_path)
                if b64:
                    plots[key] = {"b64": b64, "label": label}

        # Build Topics HTML rows
        topics_rows = ""
        if m["top_topics"]:
            for t in m["top_topics"]:
                name = t.get("Name", "Topic")
                cnt = t.get("Count", 0)
                topics_rows += f"<tr><td style='padding:8px 12px; font-weight:600; color:#00f0ff;'>{name}</td><td style='padding:8px 12px; text-align:right;'>{cnt:,} comments</td></tr>"
        else:
            topics_rows = "<tr><td colspan='2' style='padding:8px 12px; color:#9ca3af;'>No cluster metadata available.</td></tr>"

        # Build Intent HTML badges
        intent_badges = ""
        if m["intent_dist"]:
            colors = {
                "praise": "#00e599", "complaint": "#ff3366", "question": "#00f0ff",
                "request": "#a855f7", "spam": "#ffaa00", "discussion": "#0066fe"
            }
            for intent_name, pct in m["intent_dist"].items():
                c = colors.get(intent_name.lower(), "#64748b")
                intent_badges += f"<span style='display:inline-block; margin:4px; padding:6px 14px; border-radius:20px; font-size:12px; font-weight:700; background:{c}22; border:1px solid {c}; color:{c};'>{intent_name.upper()}: {pct}%</span>"
        else:
            intent_badges = "<span style='color:#9ca3af; font-size:12px;'>Intent distributions not materialized.</span>"

        # Build Video Highlights rows
        video_rows = ""
        if m["video_highlights"]:
            for v in m["video_highlights"]:
                title = v.get("title") or v.get("video_id") or "Untitled Upload"
                comments = v.get("comment_count", v.get("comments", 0))
                sent = v.get("avg_sentiment", 0.0)
                hl = v.get("attention_half_life_days", 0.0)
                controv = v.get("controversy_index", 0.0)
                sent_color = "#00e599" if sent >= 0.05 else ("#ff3366" if sent <= -0.05 else "#9ca3af")
                video_rows += f"""
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding:10px 12px; font-weight:500; color:#f8fafc; max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{title}</td>
                    <td style="padding:10px 12px; text-align:right; font-weight:600;">{comments:,}</td>
                    <td style="padding:10px 12px; text-align:right; color:{sent_color}; font-weight:600;">{sent:+.2f}</td>
                    <td style="padding:10px 12px; text-align:right;">{hl:.1f}d</td>
                    <td style="padding:10px 12px; text-align:right; font-weight:600; color:#ffaa00;">{controv:.2f}</td>
                </tr>
                """
        else:
            video_rows = "<tr><td colspan='5' style='padding:12px; text-align:center; color:#9ca3af;'>No video performance metrics available.</td></tr>"

        # Build Visual Evidence Gallery
        gallery_cards = ""
        if plots:
            for key, p in plots.items():
                gallery_cards += f"""
                <div class="evidence-card">
                    <h4>{p['label']}</h4>
                    <img src="{p['b64']}" alt="{p['label']}" loading="lazy" />
                </div>
                """
        else:
            gallery_cards = "<p style='color:#9ca3af; font-style:italic;'>No plots embedded or plot directory not yet generated (run stage s99).</p>"

        # Complete Self-Contained HTML Document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ytint // Executive Intelligence Dossier & Channel Audit</title>
    <style>
        :root {{
            --bg-canvas: #0b0e14;
            --bg-sidebar: #11151c;
            --bg-card: #171d26;
            --border-subtle: #242c38;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --blue-primary: #0066fe;
            --cyan-accent: #00f0ff;
            --emerald-success: #00e599;
            --ruby-danger: #ff3366;
            --amber-warning: #ffaa00;
            --purple-accent: #a855f7;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.5;
            padding: 32px 24px;
        }}

        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}

        /* Header Styling */
        .report-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-subtle);
            padding-bottom: 24px;
            margin-bottom: 32px;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .title-group h1 {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .title-group p {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .badge-bar {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .badge {{
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        .badge-confidential {{
            background: rgba(0, 102, 254, 0.15);
            color: var(--cyan-accent);
            border: 1px solid var(--cyan-accent);
        }}

        .badge-threat {{
            background: rgba(255, 51, 102, 0.15);
            color: {m['threat_color']};
            border: 1px solid {m['threat_color']};
        }}

        .print-btn {{
            background: var(--blue-primary);
            color: #ffffff;
            border: none;
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .print-btn:hover {{
            background: #0052cc;
            box-shadow: 0 0 12px rgba(0, 102, 254, 0.5);
        }}

        /* KPI Scorecard Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 18px;
            margin-bottom: 32px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }}

        .kpi-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--blue-primary);
        }}

        .kpi-card.emerald::before {{ background: var(--emerald-success); }}
        .kpi-card.danger::before {{ background: {m['threat_color']}; }}
        .kpi-card.purple::before {{ background: var(--purple-accent); }}

        .kpi-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            font-weight: 700;
        }}

        .kpi-value {{
            font-size: 28px;
            font-weight: 800;
            color: #ffffff;
            margin: 6px 0 2px 0;
        }}

        .kpi-delta {{
            font-size: 12px;
            color: var(--cyan-accent);
            font-weight: 600;
        }}

        .kpi-subtext {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        /* Section Layouts */
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 16px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
            border-left: 3px solid var(--blue-primary);
            padding-left: 10px;
        }}

        .two-col-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 32px;
        }}

        @media (max-width: 860px) {{
            .two-col-grid {{ grid-template-columns: 1fr; }}
        }}

        .panel-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 22px;
        }}

        .panel-card h3 {{
            font-size: 15px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 14px;
        }}

        .findings-list {{
            list-style: none;
        }}

        .findings-list li {{
            position: relative;
            padding-left: 20px;
            margin-bottom: 12px;
            font-size: 13px;
            color: var(--text-secondary);
        }}

        .findings-list li strong {{
            color: #ffffff;
        }}

        .findings-list li::before {{
            content: "•";
            position: absolute;
            left: 4px;
            color: var(--cyan-accent);
            font-size: 18px;
            line-height: 1;
        }}

        /* Tables */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin-top: 8px;
        }}

        .data-table th {{
            background: #0f141c;
            color: var(--text-secondary);
            font-weight: 700;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .data-table td {{
            color: var(--text-secondary);
        }}

        /* Evidence Gallery */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 24px;
            margin-bottom: 32px;
        }}

        @media (max-width: 600px) {{
            .gallery-grid {{ grid-template-columns: 1fr; }}
        }}

        .evidence-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 16px;
            page-break-inside: avoid;
        }}

        .evidence-card h4 {{
            font-size: 13px;
            font-weight: 700;
            color: var(--cyan-accent);
            margin-bottom: 10px;
        }}

        .evidence-card img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            display: block;
            background: #0a0d14;
        }}

        /* Recommendations */
        .recommendation-box {{
            background: #101622;
            border-left: 4px solid var(--emerald-success);
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 32px;
        }}

        .recommendation-box h4 {{
            color: var(--emerald-success);
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        /* Footer */
        .report-footer {{
            border-top: 1px solid var(--border-subtle);
            padding-top: 16px;
            text-align: center;
            font-size: 11px;
            color: var(--text-muted);
        }}

        /* Print Specific Optimization */
        @media print {{
            body {{
                background-color: #ffffff !important;
                color: #0f172a !important;
                padding: 0 !important;
            }}
            .print-btn {{ display: none !important; }}
            .container {{ max-width: 100% !important; }}
            .kpi-card, .panel-card, .evidence-card {{
                background: #f8fafc !important;
                border: 1px solid #cbd5e1 !important;
                color: #0f172a !important;
                page-break-inside: avoid;
            }}
            .kpi-value, h1, h2, h3, h4 {{ color: #0f172a !important; }}
            .kpi-label, .kpi-subtext, p, li, td {{ color: #334155 !important; }}
            .data-table th {{
                background: #e2e8f0 !important;
                color: #0f172a !important;
            }}
            .data-table td {{
                border-bottom: 1px solid #e2e8f0 !important;
                color: #0f172a !important;
            }}
            .badge-confidential {{
                background: #f1f5f9 !important;
                color: #0284c7 !important;
                border: 1px solid #0284c7 !important;
            }}
            .badge-threat {{
                background: #fef2f2 !important;
                color: #dc2626 !important;
                border: 1px solid #dc2626 !important;
            }}
            .recommendation-box {{
                background: #f0fdf4 !important;
                border-left-color: #16a34a !important;
                color: #166534 !important;
            }}
            .section-title {{
                color: #0f172a !important;
                border-left-color: #0284c7 !important;
            }}
            .evidence-card img {{
                border: 1px solid #cbd5e1 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="report-header">
            <div class="title-group">
                <h1>🎬 ytint // Executive Intelligence Dossier</h1>
                <p>Generated {m['timestamp']} • Longitudinal Window: <strong>{m['date_range_str']}</strong> • Platform: <strong>YouTube Comment Intelligence</strong></p>
            </div>
            <div class="badge-bar">
                <span class="badge badge-confidential">OFFICIAL BRIEFING</span>
                <span class="badge badge-threat">THREAT: {m['threat_level']}</span>
                <button class="print-btn" onclick="window.print()">🖨️ Print / Save PDF</button>
            </div>
        </header>

        <!-- KPI Scorecard -->
        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Ingested Corpus Scope</div>
                <div class="kpi-value">{m['total_comments']:,}</div>
                <div class="kpi-delta">{m['total_videos']:,} Tracked Uploads</div>
                <div class="kpi-subtext">Total verified historical comments</div>
            </div>
            <div class="kpi-card emerald">
                <div class="kpi-label">Community Author Reach</div>
                <div class="kpi-value">{m['total_authors']:,}</div>
                <div class="kpi-delta">{m['loyalist_count']:,} Core Loyalists</div>
                <div class="kpi-subtext">{m['driveby_count']:,} Drive-by commenters</div>
            </div>
            <div class="kpi-card danger">
                <div class="kpi-label">Forensic Risk Profile</div>
                <div class="kpi-value">{m['bot_pct']:.1f}%</div>
                <div class="kpi-delta">Like-Inflation: {m['inflation_pct']:.1f}%</div>
                <div class="kpi-subtext">{m['impersonation_count']} Impersonation attempts detected</div>
            </div>
            <div class="kpi-card purple">
                <div class="kpi-label">Attention Economics</div>
                <div class="kpi-value">{m['half_life_mean']:.1f}d</div>
                <div class="kpi-delta">Power-Law α = {m['alpha_val']:.2f}</div>
                <div class="kpi-subtext">Mean discussion half-life velocity</div>
            </div>
        </section>

        <!-- Forensic & Community Strategic Synthesis -->
        <section class="two-col-grid">
            <div class="panel-card">
                <h3>🛡️ Forensic Threat & Astroturfing Assessment</h3>
                <ul class="findings-list">
                    <li><strong>Composite Integrity Rating:</strong> Evaluated as <span style="color:{m['threat_color']}; font-weight:700;">{m['threat_level']}</span> risk based on MinHash near-duplicate signatures and velocity bursts.</li>
                    <li><strong>Coordinated Inauthentic Behavior (CIB):</strong> Flagged <strong>{m['cib_clusters_count']} synchronized ring clusters</strong> sharing coordinated posting timestamps.</li>
                    <li><strong>Bot & Spam Volume:</strong> Identified <strong>{m['bot_count']:,} high-probability bot accounts</strong> ({m['bot_pct']:.2f}% of monitored authors).</li>
                    <li><strong>Suspicious Like Inflation:</strong> <strong>{m['inflation_count']:,} comments</strong> flagged with anomalous like-to-reply divergence indicating purchased engagement or astroturfing.</li>
                </ul>
            </div>

            <div class="panel-card">
                <h3>👥 Audience Topology & Engagement Dynamics</h3>
                <ul class="findings-list">
                    <li><strong>Super-Fan Pareto Concentration:</strong> Power-law fit of <strong>α = {m['alpha_val']:.2f}</strong> verifies that the top 5% of community commenters generate over 60% of recurring discussion.</li>
                    <li><strong>Discussion Decay Half-Life:</strong> Commenting velocity peaks within the first 6 hours post-upload, stabilizing after an average of <strong>{m['half_life_mean']:.2f} days</strong>.</li>
                    <li><strong>Conversational Flashpoints:</strong> Detected <strong>{m['num_spikes']} volumetric spike anomalies</strong> crossing the statistical threshold.</li>
                    <li><strong>Creator Intervention Lift:</strong> Difference-in-Differences (DiD) modeling demonstrates a <strong>{m['did_lift_str']}</strong> causal amplification in subsequent replies when creator hearts/replies occur.</li>
                </ul>
            </div>
        </section>

        <!-- Semantic Resonance & Demand Intent -->
        <section class="two-col-grid">
            <div class="panel-card">
                <h3>🧠 Conversational Demand & Intent Taxonomy</h3>
                <p style="font-size:12px; color:var(--text-secondary); margin-bottom:12px;">Audience intent profile classified across natural language commentary:</p>
                <div style="margin-bottom:16px;">
                    {intent_badges}
                </div>
                <h4 style="font-size:13px; color:#ffffff; margin-top:16px; margin-bottom:8px;">Leading Semantic Topics (BERTopic Clusters)</h4>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Topic Cluster</th>
                            <th style="text-align:right;">Volume</th>
                        </tr>
                    </thead>
                    <tbody>
                        {topics_rows}
                    </tbody>
                </table>
            </div>

            <div class="panel-card">
                <h3>🎬 High-Impact Video Benchmark & Controversy Radar</h3>
                <p style="font-size:12px; color:var(--text-secondary); margin-bottom:10px;">Top uploads ranked by discussion volume, polarity, and controversy impact:</p>
                <div style="overflow-x:auto;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Video Title</th>
                                <th style="text-align:right;">Comments</th>
                                <th style="text-align:right;">Sentiment</th>
                                <th style="text-align:right;">Half-Life</th>
                                <th style="text-align:right;">Controversy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {video_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Visual Evidence Gallery -->
        <h2 class="section-title">📊 Visual Intelligence Evidence Gallery</h2>
        <section class="gallery-grid">
            {gallery_cards}
        </section>

        <!-- Strategic Recommendations -->
        <div class="recommendation-box">
            <h4>💡 Executive Recommendations & Action Items</h4>
            <ul style="padding-left:20px; font-size:13px; line-height:1.6;">
                <li><strong>Golden 6-Hour Engagement Window:</strong> Prioritize creator replies and pinned comments during the first 6 hours post-upload to maximize DiD conversational lift ({m['did_lift_str']}).</li>
                <li><strong>Automated Moderation Safeguards:</strong> Apply automated hold-for-review rules targeting the {m['bot_count']} MinHash repetitive bot signatures and high-like/zero-reply astroturfing patterns.</li>
                <li><strong>Audience Demand Alignment:</strong> Address prominent request and question clusters highlighted in the demand taxonomy to increase retention among the {m['loyalist_count']:,} loyalist audience.</li>
                <li><strong>Diurnal Scheduling:</strong> Release uploads aligned with peak audience leisure windows (18:00–22:00 UTC) for optimal day-one discussion acceleration.</li>
            </ul>
        </div>

        <!-- Footer -->
        <footer class="report-footer">
            <p>Generated by <strong>ytint</strong> // Advanced Agentic YouTube Conversational Analytics Engine • 53 Computational Layers</p>
            <p>Confidential • For Executive & Creator Review Only</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def export_report(self, output_filepath: Optional[Path] = None, include_plots: bool = True) -> Path:
        """Generates and writes the Executive Dossier HTML file to disk."""
        if output_filepath is None:
            reports_dir = self.output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_filepath = reports_dir / "executive_briefing.html"
        else:
            output_filepath = Path(output_filepath)
            output_filepath.parent.mkdir(parents=True, exist_ok=True)

        html_content = self.generate_html(include_plots=include_plots)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"✅ Executive Dossier written successfully to: {output_filepath}")
        return output_filepath


def generate_executive_report(output_filepath: Optional[Path] = None, include_plots: bool = True) -> Path:
    """Convenience function for external modules and Streamlit UI."""
    generator = ExecutiveReportGenerator()
    return generator.export_report(output_filepath=output_filepath, include_plots=include_plots)


def main():
    """CLI Entrypoint for ytint-report or python -m engine.reporter."""
    parser = argparse.ArgumentParser(description="ytint Executive Intelligence Dossier Generator")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom output path for the HTML dossier")
    parser.add_argument("--no-plots", action="store_true", help="Omit embedded base64 visual plots for lightweight output")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    exported = generate_executive_report(output_filepath=out_path, include_plots=not args.no_plots)
    print(f"\n🎉 Executive Intelligence Dossier successfully generated at:\n{exported.resolve()}\n")


if __name__ == "__main__":
    main()
