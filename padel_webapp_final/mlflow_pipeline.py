"""
=============================================================================
PADEL ANALYTICS — AUTOMATED MLFLOW PIPELINE
=============================================================================
This script runs the full training pipeline and automatically:
  1. Trains all 14 models
  2. Logs everything to MLflow (params, metrics, artifacts)
  3. Identifies the BEST model per task (classification, regression)
  4. Registers the best models in MLflow Model Registry
  5. Promotes them to "Production" stage automatically
  6. Logs results to a summary file

Usage:
    python mlflow_pipeline.py
=============================================================================
"""
import os
import sys
import time
import subprocess
import json
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# =============================================================================
# CONFIGURATION
# =============================================================================
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "Padel_Analytics"

# Quality thresholds per task (don't promote runs below these)
TASK_THRESHOLDS = {
    "classification": {"metric": "ROC_AUC", "min": 0.70, "registry_prefix": "padel-classifier"},
    "regression":     {"metric": "R2",      "min": 0.50, "registry_prefix": "padel-regressor"},
    "clustering":     {"metric": "silhouette", "min": 0.20, "registry_prefix": "padel-clustering"},
    "recommendation": {"metric": "type_hit_rate", "min": 0.30, "registry_prefix": "padel-recommendation"},
    "anomaly_detection": {"metric": "anomalies_detected", "min": 1, "registry_prefix": "padel-anomaly"},
    "time_series": {"metric": "MAE", "min": -1.0, "registry_prefix": "padel-timeseries", "lower_is_better": True},
}

# =============================================================================
# HELPERS
# =============================================================================
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icon = {"INFO": "[*]", "OK": "[OK]", "WARN": "[!!]", "ERR": "[ERROR]"}.get(level, "[*]")
    print(f"{ts} {icon} {msg}")


def run_training():
    """Run the actual ML pipeline as a subprocess."""
    log("Starting training pipeline...", "INFO")
    start = time.time()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["MLFLOW_TRACKING_URI"] = MLFLOW_URI

    result = subprocess.run(
        [sys.executable, "padel_ml_pipeline.py"],
        capture_output=True, text=True, env=env, encoding="utf-8",
        timeout=1800
    )
    duration = round(time.time() - start, 1)

    if result.returncode != 0:
        log(f"Pipeline FAILED after {duration}s", "ERR")
        log(f"stderr: {result.stderr[-500:]}", "ERR")
        return False, duration, result.stderr
    log(f"Pipeline finished in {duration}s", "OK")
    return True, duration, result.stdout


def find_all_runs(client, experiment_id):
    """Find latest run per (task, run_name) combination. Returns list of unique runs."""
    all_runs = []
    seen_run_names = set()  # Track (task, run_name) to deduplicate
    
    for task, cfg in TASK_THRESHOLDS.items():
        metric = cfg["metric"]
        min_score = cfg["min"]
        lower_is_better = cfg.get("lower_is_better", False)
        prefix = cfg["registry_prefix"]
        
        log(f"Searching all runs for task '{task}' (metric: {metric})...", "INFO")
        # Always order by start_time DESC so we get the latest first
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"params.task = '{task}'",
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )
        if not runs:
            log(f"No runs found for task {task}", "WARN")
            continue
        
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "unknown")
            
            # DEDUPLICATE: skip if we already have this run_name for this task
            key = (task, run_name)
            if key in seen_run_names:
                continue
            seen_run_names.add(key)
            
            score = run.data.metrics.get(metric, None)
            if score is None:
                continue
            
            # Quality check
            if lower_is_better:
                qualifies = (min_score < 0) or (score <= abs(min_score))
            else:
                qualifies = score >= min_score
            
            if not qualifies:
                continue
            
            # Extract model type from run name (e.g., "Classifier_Random_Forest" -> "random-forest")
            model_subname = run_name.split("_", 1)[-1].lower().replace("_", "-")
            registry_name = f"{prefix}-{model_subname}"
            
            all_runs.append({
                "registry_name": registry_name,
                "run_id": run.info.run_id,
                "run_name": run_name,
                "task": task,
                "metric": metric,
                "score": score
            })
            log(f"  Found: {run_name} | {metric}={score:.4f} | -> {registry_name}", "OK")
    
    return all_runs


def register_and_promote(client, all_runs):
    """Register every run and mark the best per task as Production."""
    promoted = []
    
    # Group runs by task to find the best per task
    by_task = {}
    for run in all_runs:
        task = run["task"]
        by_task.setdefault(task, []).append(run)
    
    # Find best run per task (will be tagged "production")
    best_run_ids = set()
    for task, runs in by_task.items():
        cfg = TASK_THRESHOLDS[task]
        lower_is_better = cfg.get("lower_is_better", False)
        if lower_is_better:
            best = min(runs, key=lambda r: r["score"])
        else:
            best = max(runs, key=lambda r: r["score"])
        best_run_ids.add(best["run_id"])
        log(f"Best for {task}: {best['run_name']} | {best['metric']}={best['score']:.4f}", "OK")
    
    # Register every run
    for run in all_runs:
        registry_name = run["registry_name"]
        run_id = run["run_id"]
        model_uri = f"runs:/{run_id}/model"
        is_best = run_id in best_run_ids
        
        # 1. Register
        try:
            registered = mlflow.register_model(model_uri, registry_name)
            version = registered.version
            log(f"Registered '{registry_name}' as version {version}", "OK")
        except Exception as e:
            log(f"Could not register {registry_name}: {str(e)[:100]}", "WARN")
            continue
        
        # 2. Tag as champion if it's the best for its task, otherwise as challenger
        try:
            time.sleep(1)
            alias = "champion" if is_best else "challenger"
            try:
                client.set_registered_model_alias(registry_name, alias, version)
                log(f"  -> {registry_name} v{version} marked as {alias.upper()}", "OK")
            except Exception:
                # Fall back to old stage API
                stage = "Production" if is_best else "Staging"
                client.transition_model_version_stage(registry_name, version, stage)
                log(f"  -> {registry_name} v{version} -> {stage}", "OK")
            
            promoted.append({
                "model": registry_name, "version": version,
                "metric": run["metric"], "score": round(run["score"], 4),
                "is_best": is_best
            })
        except Exception as e:
            log(f"Could not tag {registry_name}: {str(e)[:100]}", "WARN")
    
    return promoted


def save_summary(success, duration, promoted, errors=None):
    """Save a summary JSON for the API/dashboard to read."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "status": "SUCCESS" if success else "FAILED",
        "duration_seconds": duration,
        "promoted_models": promoted,
        "errors": errors,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved to logs/pipeline_summary.json", "OK")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    log("=" * 60, "INFO")
    log("PADEL ANALYTICS — AUTOMATED MLFLOW PIPELINE", "INFO")
    log("=" * 60, "INFO")
    log(f"MLflow URI: {MLFLOW_URI}", "INFO")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Get / create experiment
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(EXPERIMENT_NAME)
        log(f"Created experiment {EXPERIMENT_NAME} (id={exp_id})", "OK")
    else:
        exp_id = exp.experiment_id
        log(f"Using existing experiment {EXPERIMENT_NAME} (id={exp_id})", "OK")

    # 1. Train
    success, duration, output = run_training()
    if not success:
        save_summary(False, duration, [], errors=output)
        return 1

    # 2. Find ALL runs (not just best)
    log("\n" + "=" * 60, "INFO")
    log("FINDING ALL TRAINED MODELS", "INFO")
    log("=" * 60, "INFO")
    all_runs = find_all_runs(client, exp_id)

    if not all_runs:
        log("No models qualified for registration!", "WARN")
        save_summary(True, duration, [])
        return 0

    log(f"\nTotal models found: {len(all_runs)}", "OK")

    # 3. Register all & mark champions
    log("\n" + "=" * 60, "INFO")
    log("REGISTERING ALL MODELS + CHAMPION SELECTION", "INFO")
    log("=" * 60, "INFO")
    promoted = register_and_promote(client, all_runs)

    # 4. Save summary
    save_summary(True, duration, promoted)

    # 5. Final report
    log("\n" + "=" * 60, "INFO")
    log("PIPELINE COMPLETE", "OK")
    log("=" * 60, "INFO")
    log(f"Duration: {duration}s", "INFO")
    log(f"Models registered: {len(promoted)}", "INFO")
    champions = [p for p in promoted if p.get("is_best")]
    challengers = [p for p in promoted if not p.get("is_best")]
    log(f"  Champions (best per task): {len(champions)}", "OK")
    for p in champions:
        log(f"    -> {p['model']} v{p['version']} | {p['metric']}={p['score']}", "OK")
    log(f"  Challengers: {len(challengers)}", "INFO")
    for p in challengers:
        log(f"    -> {p['model']} v{p['version']} | {p['metric']}={p['score']}", "INFO")
    log(f"\nView in MLflow UI: {MLFLOW_URI}", "INFO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
