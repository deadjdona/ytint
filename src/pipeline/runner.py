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
        
        # Lineage Mapping: Maps stages to description, specific file outputs, and function hooks
        # Stages are executed in sorted key order (s00, s01, s02a, s02b, s03, s04a, s04b, s05, s06)
        self.registry = {
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
                # Sentinel file marks that enrichment completed — avoids the
                # input==output skip-logic problem with comments_clean.parquet
                "outputs": [self.interim / ".s01_complete"]
            },
            "s02a": {
                "desc": "Network & Graph Construction",
                "module": "pipeline.s02_network",
                "entry_func": "build_networks",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.interim / "authors_network_metrics.parquet"]
            },
            "s02b": {
                "desc": "Topic Modeling (BERTopic)",
                "module": "pipeline.s02_topics",
                "entry_func": "run_topic_modeling",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "topic_metadata.parquet"]
            },
            "s03": {
                "desc": "Narrative Timeline & Change-Points",
                "module": "pipeline.s03_narrative",
                "entry_func": "compile_narrative",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "historical_timeline.parquet", self.output / "viral_events.parquet"]
            },
            "s04a": {
                "desc": "Aggregation & Cohort Modeling",
                "module": "pipeline.s04_aggregation",
                "entry_func": "aggregate_data",
                "inputs": [self.interim / "comments_clean.parquet", self.interim / "authors_network_metrics.parquet"],
                "outputs": [self.output / "authors_final.parquet", self.output / "videos_final.parquet"]
            },
            "s04b": {
                "desc": "UI Metric Synthesis",
                "module": "pipeline.s04_synthesis",
                "entry_func": "compile_ui_metrics",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"],
                "outputs": []  # Modifies topic_metadata.parquet in-place
            },
            "s05": {
                "desc": "Advanced Statistical & Predictive Modeling",
                "module": "pipeline.s05_modeling",
                "entry_func": "run_modeling",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "authors_final.parquet"],
                "outputs": [self.output / "xgboost_like_predictor.pkl", self.output / "kaplan_meier_survival.parquet"]
            },
            "s07": {
                "desc": "Cross-Modal Reaction Mapping",
                "module": "pipeline.s07_cross_modal",
                "entry_func": "run_cross_modal",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "video_reaction_map.parquet"]
            },
            "s08": {
                "desc": "Integrity & Bot Detection",
                "module": "pipeline.s08_integrity",
                "entry_func": "run_integrity_detection",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.interim / "integrity_flags.parquet"]
            },
            "s09": {
                "desc": "Thread Polarization Dynamics",
                "module": "pipeline.s09_polarization",
                "entry_func": "run_polarization_dynamics",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "thread_polarization_corpus.parquet"]
            },
            "s10": {
                "desc": "Topic Evolution over Time",
                "module": "pipeline.s10_topic_evolution",
                "entry_func": "run_topic_evolution",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"],
                "outputs": [self.output / "topic_evolution.parquet"]
            },
            "s11": {
                "desc": "Reply Latency Distribution",
                "module": "pipeline.s11_reply_latency",
                "entry_func": "run_reply_latency",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "reply_latency.parquet"]
            },
            "s12": {
                "desc": "Shelf Life of Likes",
                "module": "pipeline.s12_shelf_life",
                "entry_func": "run_shelf_life",
                "inputs": [self.interim / "comments_clean.parquet", self.interim / "videos_clean.parquet"],
                "outputs": [self.output / "shelf_life_likes.parquet"]
            },
            "s13": {
                "desc": "Topic x Video Matrix",
                "module": "pipeline.s13_topic_matrix",
                "entry_func": "run_topic_matrix",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"],
                "outputs": [self.output / "topic_video_matrix.parquet"]
            },
            "s14": {
                "desc": "Word Co-occurrence Network",
                "module": "pipeline.s14_cooccurrence",
                "entry_func": "run_cooccurrence",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "word_cooccurrence_edges.parquet", self.output / "word_cooccurrence_nodes.parquet"]
            },
            "s15": {
                "desc": "Named Entity Extraction",
                "module": "pipeline.s15_ner",
                "entry_func": "run_ner",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "named_entities.parquet"]
            },
            "s16": {
                "desc": "Hashtag & Mention Networks",
                "module": "pipeline.s16_tags_mentions",
                "entry_func": "run_tag_networks",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "tag_network_edges.parquet", self.output / "tag_network_nodes.parquet"]
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
                "desc": "Acquisition Cohorts",
                "module": "pipeline.s19_cohorts",
                "entry_func": "run_cohorts",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "cohort_retention.parquet", self.output / "cohort_retention_pct.parquet"]
            },
            "s20": {
                "desc": "Cross-Video Overlap",
                "module": "pipeline.s20_overlap",
                "entry_func": "run_overlap",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "video_overlap_matrix.parquet"]
            },
            "s21": {
                "desc": "Topic-Cohort Clustering",
                "module": "pipeline.s21_topic_cohorts",
                "entry_func": "run_topic_cohorts",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "topic_cohorts.parquet"]
            },
            "s22": {
                "desc": "Commenting Frequency Tiers",
                "module": "pipeline.s22_frequency_tiers",
                "entry_func": "run_frequency_tiers",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "frequency_tiers.parquet"]
            },
            "s23": {
                "desc": "Bot Heuristics Classifier",
                "module": "pipeline.s23_bot_heuristics",
                "entry_func": "run_bot_heuristics",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "bot_classifications.parquet"]
            },
            "s24": {
                "desc": "Drive-by vs Loyalists",
                "module": "pipeline.s24_driveby_loyalists",
                "entry_func": "run_driveby_loyalists",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "driveby_loyalists.parquet"]
            },
            "s25": {
                "desc": "Position Bias & Branching Factor",
                "module": "pipeline.s25_position_bias",
                "entry_func": "run_position_bias",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "position_bias.parquet"]
            },
            "s26": {
                "desc": "Arrival Speed Classification",
                "module": "pipeline.s26_arrival_speed",
                "entry_func": "run_arrival_speed",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "arrival_speed.parquet"]
            },
            "s27": {
                "desc": "Thread Width & Branching",
                "module": "pipeline.s27_thread_width",
                "entry_func": "run_thread_width",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "thread_width_dist.parquet"]
            },
            "s28": {
                "desc": "Conversation Resolution Patterns",
                "module": "pipeline.s28_resolution_patterns",
                "entry_func": "run_resolution_patterns",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "resolution_patterns.parquet"]
            },
            "s29": {
                "desc": "Initiator/Response Patterns",
                "module": "pipeline.s29_initiator_patterns",
                "entry_func": "run_initiator_patterns",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "initiator_patterns.parquet"]
            },
            "s30": {
                "desc": "Code-Switching Detection",
                "module": "pipeline.s30_code_switching",
                "entry_func": "run_code_switching",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "code_switching_impact.parquet"]
            },
            "s31": {
                "desc": "Comparative & Cross-Video",
                "module": "pipeline.s31_cross_video_comparisons",
                "entry_func": "run_cross_video",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "video_profile_radar.parquet"]
            },
            "s32": {
                "desc": "Suspicious Like Inflation",
                "module": "pipeline.s32_anomaly_inflation",
                "entry_func": "run_like_inflation",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "like_inflation.parquet"]
            },
            "s33": {
                "desc": "Author Fingerprints",
                "module": "pipeline.s33_author_fingerprints",
                "entry_func": "run_author_fingerprints",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "author_fingerprints.parquet"]
            },
            "s34": {
                "desc": "Impersonation Detection",
                "module": "pipeline.s34_impersonation_detection",
                "entry_func": "run_impersonation_detection",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "impersonation_detection.parquet"]
            },
            "s06": {
                "desc": "Visualizations & Reporting",
                "module": "pipeline.s06_visualize",
                "entry_func": "run_visualizations",
                "inputs": [self.output / "authors_final.parquet", self.output / "kaplan_meier_survival.parquet"],
                "outputs": []  # Creates images in output/plots
            },
            "s99": {
                "desc": "Visualizations & Reporting",
                "module": "pipeline.s06_visualize",
                "entry_func": "run_visualizations",
                "inputs": [self.output / "authors_final.parquet", self.output / "kaplan_meier_survival.parquet"],
                "outputs": []  # Creates images in output/plots
            }
        }

    def stage_requires_execution(self, stage_id: str) -> bool:
        meta = self.registry[stage_id]
        
        # Guard 1: If target artifacts are missing from disk, execution is mandatory
        for out_file in meta["outputs"]:
            if not out_file.exists():
                logger.debug(f"Stage [{stage_id}] target missing: {out_file.name}")
                return True
                
        # Guard 2: Mtime checking logic for linear upstream tracking (primarily s00 validation)
        if stage_id == "s00":
            if self.raw_db.exists():
                min_output_mtime = min(os.path.getmtime(f) for f in meta["outputs"])
                if os.path.getmtime(self.raw_db) > min_output_mtime:
                    logger.info(f"Upstream modification observed for Stage [{stage_id}]: {self.raw_db.name}")
                    return True
                    
        return False

    def execute_stage(self, stage_id: str):
        meta = self.registry[stage_id]
        logger.info(f"🚀 Running Stage [{stage_id}] -> {meta['desc']}")
        
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
            
            logger.info(f"✨ Stage [{stage_id}] execution completed successfully.\n")
            
        except AttributeError:
            logger.error(f"❌ Operational Hook Error: Module '{meta['module']}' missing function '{meta['entry_func']}'")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"❌ Fatal Runtime Crash inside Stage [{stage_id}]: {e}")
            sys.exit(1)

    def resolve_stage_id(self, stage_input: str) -> str:
        if not stage_input:
            return None
        if stage_input in self.registry:
            return stage_input
        norm = stage_input.lower().strip()
        if not norm.startswith("s"):
            norm = f"s{int(norm):02d}" if norm.isdigit() else f"s{norm}"
        if norm in self.registry:
            return norm
        matches = [k for k in self.registry if k.startswith(norm)]
        if matches:
            return matches[0]
        return stage_input

    def run(self, force_stage: str = None, run_from: str = None):
        logger.info(f"Initializing ytint Processing Engine Execution Grid. Root context: {self.root_dir}")
        stages = sorted(list(self.registry.keys()))
        
        force_stage = self.resolve_stage_id(force_stage)
        run_from = self.resolve_stage_id(run_from)

        if force_stage and force_stage not in self.registry:
            logger.error(f"❌ Requested stage '{force_stage}' does not exist in pipeline footprint.")
            sys.exit(1)
        if run_from and run_from not in self.registry:
            logger.error(f"❌ Sequential starter target '{run_from}' does not exist in pipeline footprint.")
            sys.exit(1)

        cascade = False
        for s in stages:
            # Trigger subsequent stages if a cascade flag is raised
            if run_from and s >= run_from:
                cascade = True
                
            if force_stage == s or cascade or self.stage_requires_execution(s):
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