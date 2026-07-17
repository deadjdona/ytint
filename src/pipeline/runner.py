import os
import sys
import logging
import argparse
import pathlib
import importlib
import yaml

# Establish path constraints dynamically relative to the project root
CURRENT_FILE = pathlib.Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent
while ROOT_DIR != ROOT_DIR.parent:
    if (ROOT_DIR / "config").is_dir():
        break
    ROOT_DIR = ROOT_DIR.parent

# Inject both the root directory and the src container into the system path
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ytint_orchestrator")

class PipelineRunner:
    def __init__(self):
        config_path = ROOT_DIR / "config" / "settings.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"❌ Could not find configuration file at: {config_path}")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Absolute path translations
        self.raw_db = ROOT_DIR / self.config["paths"]["raw_db"]
        self.interim = ROOT_DIR / self.config["paths"]["interim_dir"]
        self.output = ROOT_DIR / self.config["paths"]["output_dir"]
        
        # Ensure workspace directories exist
        self.interim.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        
        # Lineage Mapping: Maps stages to description, specific file outputs, and function hooks
        self.registry = {
            "s00": {
                "desc": "Raw SQLite Data Ingestion & Synthesis",
                "module": "pipeline.s00_ingest",
                "entry_func": "migrate_from_commentsuite",
                "inputs": [self.raw_db],
                "outputs": [self.interim / "comments_clean.parquet", self.interim / "videos_clean.parquet"]
            },
            "s01": {
                "desc": "Deep RoBERTa Sentiment Enrichment & In-Place Join",
                "module": "pipeline.s01_enrich",
                "entry_func": "enrich_comments",
                "inputs": [self.interim / "comments_clean.parquet", self.interim / "videos_clean.parquet"],
                "outputs": [self.interim / "comments_clean.parquet"] 
            },
            "s02": {
                "desc": "BERTopic Vector Clustering & Discovery Layer",
                "module": "pipeline.s02_topics",
                "entry_func": "run_topic_modeling",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"]
            },
            "s03": {
                "desc": "Pelt Change-Point Detection & Volumetric Anomalies",
                "module": "pipeline.s03_narrative",
                "entry_func": "compile_narrative",
                "inputs": [self.interim / "comments_clean.parquet"],
                "outputs": [self.output / "historical_timeline.parquet", self.output / "viral_events.parquet"]
            },
            "s04": {
                "desc": "Cross-Layer Aggregations & UI Metric Synthesis",
                "module": "pipeline.s04_synthesis",
                "entry_func": "compile_ui_metrics",
                "inputs": [self.interim / "comments_clean.parquet", self.output / "topic_metadata.parquet"],
                "outputs": [self.output / "topic_metadata.parquet"] 
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
            logger.info(f"✨ Stage [{stage_id}] execution completed successfully.\n")
            
        except AttributeError:
            logger.error(f"❌ Operational Hook Error: Module '{meta['module']}' missing function '{meta['entry_func']}'")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"❌ Fatal Runtime Crash inside Stage [{stage_id}]: {e}")
            sys.exit(1)

    def run(self, force_stage: str = None, run_from: str = None):
        logger.info(f"Initializing ytint Processing Engine Execution Grid. Root context: {ROOT_DIR}")
        stages = sorted(list(self.registry.keys()))
        
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