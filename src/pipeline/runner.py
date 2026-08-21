import os
import sys
import logging
import argparse
import pathlib
import importlib

# Ensure src directory is on sys.path
_src_dir = str(pathlib.Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from engine.config_loader import load_config, get_paths

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ytint_orchestrator")

class PipelineRunner:
    def __init__(self):
        self.config = load_config()
        self.raw_db, self.interim, self.output = get_paths(self.config)
        self.root_dir = self.config["_root_dir"]
        
        # Ensure src/ is on sys.path for module imports
        src_dir = str(self.root_dir / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        # Ensure workspace directories exist
        self.interim.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        
        # Ordered Lineage Mapping: Stages execute in logical dependency order
        # Phase 1: Ingestion & Core NLP (s00 - s08)
        # Phase 2: Conversation Tree & Thread Dynamics (s09 - s15)
        # Phase 3: Author Profiling & Forensics (s16 - s27)
        # Phase 4: Video Dynamics, Cohorts & Longitudinal Topology (s28 - s37)
        # Phase 5: Predictive, Causal Modeling & Synthesis (s38 - s40)
        # Phase 6: Visualizations & Reporting (s99 - ALWAYS LAST)
        self.registry = {
            # === Phase 1: Ingestion & Foundational NLP/Semantics ===
            "s00": {
                "desc": "Raw SQLite Data Ingestion & Synthesis",
                "module": "pipeline.s00_ingest",
                "entry_func": "migrate_from_commentsuite",
                "inputs": [self.raw_db],
                "outputs": [self.interim / "comments_clean.parquet", self.interim / "videos_clean.parquet"]
            },
            "s01": {
                "desc": "Deep Sentiment & Linguistic Enrichment",
                "module": "pipeline.s01_enrich",
                "entry_func": "enrich_comments",
                "inputs": [self.interim / "comments_clean.parquet", self.interim / "videos_clean.parquet"],
                "outputs": [self.interim / ".s01_complete"]
            },
            "s02": {
                "desc": "Topic Modeling (BERTopic)",
                "module": "pipeline.s02_topics",
                "entry_func": "run_topic_modeling",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "topic_metadata.parquet"]
            },
            "s03": {
                "desc": "Named Entity Extraction (NER)",
                "module": "pipeline.s03_ner",
                "entry_func": "run_ner",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "named_entities.parquet"]
            },
            "s04": {
                "desc": "Word Co-occurrence Network",
                "module": "pipeline.s04_cooccurrence",
                "entry_func": "run_cooccurrence",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "word_cooccurrence_edges.parquet", self.output / "word_cooccurrence_nodes.parquet"]
            },
            "s05": {
                "desc": "Hashtag & Mention Networks",
                "module": "pipeline.s05_tags_mentions",
                "entry_func": "run_tag_networks",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "tag_network_edges.parquet", self.output / "tag_network_nodes.parquet"]
            },
            "s06": {
                "desc": "Audience Demand & Content Intent Mining",
                "module": "pipeline.s06_audience_intent",
                "entry_func": "run_audience_intent",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "audience_intent_summary.parquet", self.output / "audience_content_requests.parquet"]
            },
            "s07": {
                "desc": "Target-Specific Stance Detection & Polarization Drift",
                "module": "pipeline.s07_stance_drift",
                "entry_func": "run_stance_analysis",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "stance_summary.parquet", self.output / "stance_depth_drift.parquet", self.output / "polarized_threads.parquet"]
            },
            "s08": {
                "desc": "Code-Switching & Multilingual Detection",
                "module": "pipeline.s08_code_switching",
                "entry_func": "run_code_switching",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "code_switching_impact.parquet"]
            },

            # === Phase 2: Conversation Tree & Thread Dynamics ===
            "s09": {
                "desc": "Thread Width & Branching Factor",
                "module": "pipeline.s09_thread_width",
                "entry_func": "run_thread_width",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "thread_width_dist.parquet"]
            },
            "s10": {
                "desc": "Reply Latency Distribution",
                "module": "pipeline.s10_reply_latency",
                "entry_func": "run_reply_latency",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "reply_latency.parquet"]
            },
            "s11": {
                "desc": "Position Bias & Early Mover Advantage",
                "module": "pipeline.s11_position_bias",
                "entry_func": "run_position_bias",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "position_bias.parquet"]
            },
            "s12": {
                "desc": "Conversation Resolution Patterns",
                "module": "pipeline.s12_resolution_patterns",
                "entry_func": "run_resolution_patterns",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "resolution_patterns.parquet"]
            },
            "s13": {
                "desc": "Initiator/Responder Patterns",
                "module": "pipeline.s13_initiator_patterns",
                "entry_func": "run_initiator_patterns",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "initiator_patterns.parquet"]
            },
            "s14": {
                "desc": "Thread Polarization Dynamics",
                "module": "pipeline.s14_polarization",
                "entry_func": "run_polarization_dynamics",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "thread_polarization_corpus.parquet"]
            },
            "s15": {
                "desc": "Toxicity Contagion & Troll Catalyst Identification",
                "module": "pipeline.s15_toxicity_contagion",
                "entry_func": "run_toxicity_contagion",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "toxicity_contagion_summary.parquet", self.output / "troll_catalysts.parquet", self.output / "video_toxicity_contagion.parquet"]
            },

            # === Phase 3: Author Profiling & Forensics ===
            "s16": {
                "desc": "Author Network & Graph Construction",
                "module": "pipeline.s16_network",
                "entry_func": "build_networks",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.interim / "authors_network_metrics.parquet"]
            },
            "s17": {
                "desc": "Author-Video Bipartite Graph",
                "module": "pipeline.s17_bipartite",
                "entry_func": "run_bipartite",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "bipartite_edges.parquet", self.output / "bipartite_nodes.parquet"]
            },
            "s18": {
                "desc": "Co-commenting Graph",
                "module": "pipeline.s18_cocommenting",
                "entry_func": "run_cocommenting",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "cocomment_edges.parquet", self.output / "cocomment_nodes.parquet"]
            },
            "s19": {
                "desc": "Author & Video Aggregations (RFM)",
                "module": "pipeline.s19_aggregation",
                "entry_func": "aggregate_data",
                "inputs": [self.interim / "comments_clean.parquet", self.interim / "authors_network_metrics.parquet"],
                "outputs": [self.output / "authors_final.parquet", self.output / "videos_final.parquet"]
            },
            "s20": {
                "desc": "Commenting Frequency Tiers",
                "module": "pipeline.s20_frequency_tiers",
                "entry_func": "run_frequency_tiers",
                "inputs": [self.output / "authors_final.parquet", self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "frequency_tiers.parquet"]
            },
            "s21": {
                "desc": "Drive-by vs Loyalists Segmentation",
                "module": "pipeline.s21_driveby_loyalists",
                "entry_func": "run_driveby_loyalists",
                "inputs": [self.output / "authors_final.parquet", self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "driveby_loyalists.parquet"]
            },
            "s22": {
                "desc": "Author Stylometric Fingerprints",
                "module": "pipeline.s22_author_fingerprints",
                "entry_func": "run_author_fingerprints",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "author_fingerprints.parquet"]
            },
            "s23": {
                "desc": "Creator Impersonation Detection",
                "module": "pipeline.s23_impersonation_detection",
                "entry_func": "run_impersonation_detection",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "impersonation_detection.parquet"]
            },
            "s24": {
                "desc": "Bot & Spammer Heuristics Classifier",
                "module": "pipeline.s24_bot_heuristics",
                "entry_func": "run_bot_heuristics",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "authors_final.parquet"],
                "outputs": [self.output / "bot_classifications.parquet"]
            },
            "s25": {
                "desc": "Coordinated Inauthentic Behavior (CIB) Rings",
                "module": "pipeline.s25_cib_detection",
                "entry_func": "run_cib_detection",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "cib_rings.parquet", self.output / "cib_coordinated_comments.parquet"]
            },
            "s26": {
                "desc": "Integrity & Spam Flags",
                "module": "pipeline.s26_integrity",
                "entry_func": "run_integrity_detection",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.interim / "integrity_flags.parquet"]
            },
            "s27": {
                "desc": "Suspicious Like Inflation & Astroturfing",
                "module": "pipeline.s27_anomaly_inflation",
                "entry_func": "run_like_inflation",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "bot_classifications.parquet"],
                "outputs": [self.output / "like_inflation.parquet"]
            },

            # === Phase 4: Video Dynamics, Cohorts & Longitudinal Topology ===
            "s28": {
                "desc": "Narrative Timeline & Change-Points",
                "module": "pipeline.s28_narrative",
                "entry_func": "compile_narrative",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "historical_timeline.parquet", self.output / "viral_events.parquet"]
            },
            "s29": {
                "desc": "Cross-Modal Reaction Mapping",
                "module": "pipeline.s29_cross_modal",
                "entry_func": "run_cross_modal",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "video_reaction_map.parquet"]
            },
            "s30": {
                "desc": "Shelf Life of Likes & Video Decay",
                "module": "pipeline.s30_shelf_life",
                "entry_func": "run_shelf_life",
                "inputs": [self.interim / "comments_clean.parquet", self.interim / "videos_clean.parquet"],
                "outputs": [self.output / "shelf_life_likes.parquet"]
            },
            "s31": {
                "desc": "Topic Evolution over Time",
                "module": "pipeline.s31_topic_evolution",
                "entry_func": "run_topic_evolution",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"],
                "outputs": [self.output / "topic_evolution.parquet"]
            },
            "s32": {
                "desc": "Topic x Video Matrix",
                "module": "pipeline.s32_topic_matrix",
                "entry_func": "run_topic_matrix",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"],
                "outputs": [self.output / "topic_video_matrix.parquet"]
            },
            "s33": {
                "desc": "Acquisition Cohorts & Retention",
                "module": "pipeline.s33_cohorts",
                "entry_func": "run_cohorts",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "cohort_retention.parquet", self.output / "cohort_retention_pct.parquet"]
            },
            "s34": {
                "desc": "Cross-Video Audience Overlap",
                "module": "pipeline.s34_overlap",
                "entry_func": "run_overlap",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "video_overlap_matrix.parquet"]
            },
            "s35": {
                "desc": "Topic-Cohort Clustering",
                "module": "pipeline.s35_topic_cohorts",
                "entry_func": "run_topic_cohorts",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "topic_cohorts.parquet"]
            },
            "s36": {
                "desc": "Arrival Speed Dynamics",
                "module": "pipeline.s36_arrival_speed",
                "entry_func": "run_arrival_speed",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "arrival_speed.parquet"]
            },
            "s37": {
                "desc": "Comparative & Cross-Video Profiles",
                "module": "pipeline.s37_cross_video_comparisons",
                "entry_func": "run_cross_video",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "videos_final.parquet"],
                "outputs": [self.output / "video_profile_radar.parquet"]
            },

            # === Phase 5: Predictive, Causal Modeling & Synthesis ===
            "s38": {
                "desc": "Advanced Statistical & Predictive Modeling (XGBoost, SHAP, Kaplan-Meier)",
                "module": "pipeline.s38_modeling",
                "entry_func": "run_modeling",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "authors_final.parquet"],
                "outputs": [self.output / "xgboost_like_predictor.pkl", self.output / "kaplan_meier_survival.parquet"]
            },
            "s39": {
                "desc": "Creator Interaction Causal Uplift Analysis (DiD)",
                "module": "pipeline.s39_creator_uplift",
                "entry_func": "run_creator_uplift",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "creator_causal_uplift.parquet", self.output / "creator_intervention_threads.parquet"]
            },
            "s40": {
                "desc": "UI Metric Synthesis",
                "module": "pipeline.s40_synthesis",
                "entry_func": "compile_ui_metrics",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"],
                "outputs": []  # Modifies topic_metadata.parquet in-place
            },

            # === Phase 6: Final Visualizations & Reporting (ALWAYS LAST) ===
            "s99": {
                "desc": "Comprehensive Visualizations & Publication Plots",
                "module": "pipeline.s99_visualize",
                "entry_func": "run_visualizations",
                "inputs": [self.output / "authors_final.parquet", self.output / "kaplan_meier_survival.parquet"],
                "outputs": []  # Generates full plot gallery in output/plots
            }
        }

        # Ordered stage execution sequence
        self.ordered_stages = [
            f"s{i:02d}" for i in range(41)
        ] + ["s99"]

        # Named convenience aliases
        self.aliases = {
            "ingest": "s00",
            "enrich": "s01",
            "topics": "s02",
            "ner": "s03",
            "intent": "s06",
            "stance": "s07",
            "network": "s16",
            "aggregation": "s19",
            "cib": "s25",
            "integrity": "s26",
            "narrative": "s28",
            "modeling": "s38",
            "uplift": "s39",
            "synthesis": "s40",
            "s_visualize": "s99",
            "visualize": "s99",
            "plots": "s99"
        }

    def stage_requires_execution(self, stage_id: str) -> bool:
        canonical_id = self.resolve_stage_id(stage_id)
        if not canonical_id or canonical_id not in self.registry:
            return False
        meta = self.registry[canonical_id]
        
        # If no outputs defined (e.g. s40 in-place or s99 plots), check if execution needed
        if not meta["outputs"]:
            if canonical_id == "s99":
                plots_dir = self.output / "plots"
                return not (plots_dir.exists() and any(plots_dir.iterdir()))
            return True

        # Guard 1: If target artifacts are missing from disk, execution is mandatory
        for out_file in meta["outputs"]:
            if not out_file.exists():
                logger.debug(f"Stage [{canonical_id}] target missing: {out_file.name}")
                return True
                
        # Guard 2: Mtime checking logic for linear upstream tracking (primarily s00 validation)
        if canonical_id == "s00":
            if self.raw_db.exists():
                min_output_mtime = min(os.path.getmtime(f) for f in meta["outputs"])
                if os.path.getmtime(self.raw_db) > min_output_mtime:
                    logger.info(f"Upstream modification observed for Stage [{canonical_id}]: {self.raw_db.name}")
                    return True
                    
        return False

    def execute_stage(self, stage_id: str):
        canonical_id = self.resolve_stage_id(stage_id)
        if not canonical_id or canonical_id not in self.registry:
            logger.error(f"❌ Cannot execute unknown stage [{stage_id}]")
            sys.exit(1)
            
        meta = self.registry[canonical_id]
        logger.info(f"🚀 Running Stage [{canonical_id}] -> {meta['desc']}")
        
        try:
            # Dynamically import the target script module
            module = importlib.import_module(meta["module"])
            
            # Resolve and execute the targeted functional entry hook
            run_func = getattr(module, meta["entry_func"])
            
            # Run execution layer
            run_func()
            
            # Touch sentinel files if configured
            for out_file in meta["outputs"]:
                if out_file.name.startswith("."):
                    out_file.touch()
            
            logger.info(f"✨ Stage [{canonical_id}] execution completed successfully.\n")
            
        except AttributeError:
            logger.error(f"❌ Operational Hook Error: Module '{meta['module']}' missing function '{meta['entry_func']}'")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"❌ Fatal Runtime Crash inside Stage [{canonical_id}]: {e}")
            sys.exit(1)

    def resolve_stage_id(self, stage_input: str) -> str:
        if not stage_input:
            return None
        norm = str(stage_input).lower().strip()
        
        # 1. Direct registry hit
        if norm in self.registry:
            return norm
            
        # 2. Direct alias hit
        if norm in self.aliases:
            return self.aliases[norm]
            
        # 3. Numeric formatting (e.g. "6" -> "s06", "38" -> "s38")
        if norm.isdigit():
            formatted = f"s{int(norm):02d}"
            if formatted in self.registry:
                return formatted
            if formatted in self.aliases:
                return self.aliases[formatted]
                
        # 4. Strip leading 's' and reformat
        if norm.startswith("s") and norm[1:].isdigit():
            formatted = f"s{int(norm[1:]):02d}"
            if formatted in self.registry:
                return formatted
            if formatted in self.aliases:
                return self.aliases[formatted]
                
        # 5. Prefix matching
        matches = [k for k in self.registry if k.startswith(norm)]
        if matches:
            return matches[0]
            
        return stage_input

    def run(self, force_stage: str = None, run_from: str = None):
        logger.info(f"Initializing ytint Processing Engine Execution Grid. Root context: {self.root_dir}")
        
        if force_stage:
            resolved = self.resolve_stage_id(force_stage)
            if not resolved or resolved not in self.registry:
                logger.error(f"❌ Requested stage '{force_stage}' does not exist in pipeline footprint.")
                sys.exit(1)
            self.execute_stage(resolved)
            logger.info(f"🎉 Targeted Execution of Stage [{resolved}] Completed.")
            return

        stages_to_run = list(self.ordered_stages)
        if run_from:
            resolved_from = self.resolve_stage_id(run_from)
            if resolved_from and resolved_from in self.ordered_stages:
                start_idx = self.ordered_stages.index(resolved_from)
                stages_to_run = self.ordered_stages[start_idx:]

        cascade = False
        for s in stages_to_run:
            if run_from and s == self.resolve_stage_id(run_from):
                cascade = True
                
            if cascade or self.stage_requires_execution(s):
                self.execute_stage(s)
                # Lock cascade open: Once an upstream script transforms data, force update downstream layers
                cascade = True
            else:
                logger.info(f"✓ Skipping Stage [{s}] ({self.registry[s]['desc']}) - Artifacts Valid.")
        
        logger.info("🎉 Complete Pipeline Processing Sweep Executed Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ytint Analytical Engine Runner Workflow Grid")
    parser.add_argument("--stage", type=str, default=None, help="Force execute a single standalone module stage slot")
    parser.add_argument("--from-stage", type=str, default=None, help="Force sequential cascade execution from this stage index forward")
    args = parser.parse_args()

    orchestrator = PipelineRunner()
    orchestrator.run(force_stage=args.stage, run_from=args.from_stage)