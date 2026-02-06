"""
Experiment Tracking Module
==========================

Lightweight experiment tracking for causal impact analyses.
Tracks parameters, metrics, and artifacts across runs without
external dependencies like MLflow.

Author: Causal Impact Analysis Project
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
import hashlib
import shutil


@dataclass
class Run:
    """Represents a single analysis run."""
    run_id: str
    experiment_name: str
    run_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    tags: Dict[str, str]
    started_at: str
    ended_at: Optional[str]
    status: str  # running, completed, failed


@dataclass
class Experiment:
    """Represents an experiment (collection of runs)."""
    name: str
    description: str
    created_at: str
    runs: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    Lightweight experiment tracking for causal impact analysis.
    
    Tracks analysis runs with their parameters, metrics, and artifacts.
    Data is stored in JSON files for simplicity and portability.
    
    Example:
        >>> tracker = ExperimentTracker("./experiments")
        >>> 
        >>> with tracker.start_run("campaign_analysis", "run_001") as run:
        ...     run.log_param("intervention_date", "2024-03-01")
        ...     run.log_param("segment", "email")
        ...     run.log_metric("cumulative_effect", 12500.50)
        ...     run.log_metric("roi", 150.0)
        ...     run.log_artifact("report.pdf", "./output/report.pdf")
        >>> 
        >>> tracker.compare_runs("campaign_analysis")
    """
    
    def __init__(self, tracking_dir: str = None):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_dir: Directory to store tracking data
        """
        if tracking_dir is None:
            tracking_dir = Path(__file__).parent.parent / 'experiments'
        
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self._experiments_file = self.tracking_dir / 'experiments.json'
        self._active_run = None
        
        self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments index."""
        if self._experiments_file.exists():
            with open(self._experiments_file, 'r') as f:
                self._experiments = json.load(f)
        else:
            self._experiments = {}
    
    def _save_experiments(self):
        """Save experiments index."""
        with open(self._experiments_file, 'w') as f:
            json.dump(self._experiments, f, indent=2)
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}-{os.getpid()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    
    def create_experiment(self, name: str, description: str = "") -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Unique experiment name
            description: Optional description
        
        Returns:
            Created Experiment object
        """
        if name in self._experiments:
            print(f"Experiment '{name}' already exists")
            return self.get_experiment(name)
        
        experiment = Experiment(
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            runs=[]
        )
        
        # Create experiment directory
        exp_dir = self.tracking_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        self._experiments[name] = asdict(experiment)
        self._save_experiments()
        
        print(f"✓ Created experiment: {name}")
        return experiment
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        if name not in self._experiments:
            return None
        
        data = self._experiments[name]
        return Experiment(**data)
    
    def list_experiments(self) -> List[str]:
        """List all experiments."""
        return list(self._experiments.keys())
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> 'RunContext':
        """
        Start a new run within an experiment.
        
        Args:
            experiment_name: Name of experiment
            run_name: Optional run name (auto-generated if not provided)
            tags: Optional tags for the run
        
        Returns:
            RunContext for logging
        """
        # Create experiment if needed
        if experiment_name not in self._experiments:
            self.create_experiment(experiment_name)
        
        run_id = self._generate_run_id()
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = Run(
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name,
            parameters={},
            metrics={},
            artifacts={},
            tags=tags or {},
            started_at=datetime.now().isoformat(),
            ended_at=None,
            status="running"
        )
        
        # Create run directory
        run_dir = self.tracking_dir / experiment_name / run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / 'artifacts').mkdir(exist_ok=True)
        
        self._active_run = run
        
        return RunContext(self, run)
    
    def _end_run(self, run: Run, status: str = "completed"):
        """End a run and save its data."""
        run.ended_at = datetime.now().isoformat()
        run.status = status
        
        # Save run data
        run_dir = self.tracking_dir / run.experiment_name / run.run_id
        with open(run_dir / 'run.json', 'w') as f:
            json.dump(asdict(run), f, indent=2)
        
        # Update experiment index
        if run.run_id not in self._experiments[run.experiment_name]['runs']:
            self._experiments[run.experiment_name]['runs'].append(run.run_id)
            self._save_experiments()
        
        self._active_run = None
        print(f"✓ Run {run.run_name} completed")
    
    def get_run(self, experiment_name: str, run_id: str) -> Optional[Run]:
        """Get a specific run."""
        run_file = self.tracking_dir / experiment_name / run_id / 'run.json'
        if not run_file.exists():
            return None
        
        with open(run_file, 'r') as f:
            data = json.load(f)
        
        return Run(**data)
    
    def list_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """List all runs for an experiment."""
        if experiment_name not in self._experiments:
            return []
        
        runs = []
        for run_id in self._experiments[experiment_name]['runs']:
            run = self.get_run(experiment_name, run_id)
            if run:
                runs.append({
                    'run_id': run.run_id,
                    'run_name': run.run_name,
                    'started_at': run.started_at,
                    'status': run.status,
                    'metrics': run.metrics
                })
        
        return runs
    
    def compare_runs(
        self,
        experiment_name: str,
        metric_keys: Optional[List[str]] = None
    ) -> Any:
        """
        Compare runs within an experiment.
        
        Args:
            experiment_name: Experiment to compare
            metric_keys: Specific metrics to compare (None = all)
        
        Returns:
            DataFrame with comparison
        """
        import pandas as pd
        
        runs = self.list_runs(experiment_name)
        if not runs:
            return pd.DataFrame()
        
        # Build comparison data
        data = []
        for run_info in runs:
            run = self.get_run(experiment_name, run_info['run_id'])
            if run:
                row = {
                    'run_name': run.run_name,
                    'started_at': run.started_at,
                    'status': run.status,
                    **run.parameters,
                    **run.metrics
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        if metric_keys:
            cols = ['run_name', 'started_at'] + [c for c in metric_keys if c in df.columns]
            df = df[cols]
        
        return df
    
    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        minimize: bool = False
    ) -> Optional[Run]:
        """
        Get the best run based on a metric.
        
        Args:
            experiment_name: Experiment to search
            metric: Metric to optimize
            minimize: If True, lower is better
        
        Returns:
            Best Run object
        """
        runs = self.list_runs(experiment_name)
        if not runs:
            return None
        
        best_run_id = None
        best_value = float('inf') if minimize else float('-inf')
        
        for run_info in runs:
            if metric in run_info['metrics']:
                value = run_info['metrics'][metric]
                if minimize and value < best_value:
                    best_value = value
                    best_run_id = run_info['run_id']
                elif not minimize and value > best_value:
                    best_value = value
                    best_run_id = run_info['run_id']
        
        if best_run_id:
            return self.get_run(experiment_name, best_run_id)
        return None
    
    def delete_run(self, experiment_name: str, run_id: str) -> bool:
        """Delete a run and its artifacts."""
        if experiment_name not in self._experiments:
            return False
        
        run_dir = self.tracking_dir / experiment_name / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        
        if run_id in self._experiments[experiment_name]['runs']:
            self._experiments[experiment_name]['runs'].remove(run_id)
            self._save_experiments()
        
        return True
    
    def delete_experiment(self, name: str) -> bool:
        """Delete an experiment and all its runs."""
        if name not in self._experiments:
            return False
        
        exp_dir = self.tracking_dir / name
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        
        del self._experiments[name]
        self._save_experiments()
        
        return True


class RunContext:
    """Context manager for a run, providing logging methods."""
    
    def __init__(self, tracker: ExperimentTracker, run: Run):
        self.tracker = tracker
        self.run = run
        self._run_dir = tracker.tracking_dir / run.experiment_name / run.run_id
    
    def __enter__(self) -> 'RunContext':
        print(f"Starting run: {self.run.run_name} ({self.run.run_id})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "completed"
        self.tracker._end_run(self.run, status)
        return False  # Don't suppress exceptions
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        self.run.parameters[key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        self.run.parameters.update(params)
    
    def log_metric(self, key: str, value: float):
        """Log a metric."""
        self.run.metrics[key] = value
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics."""
        self.run.metrics.update(metrics)
    
    def log_artifact(self, name: str, source_path: str):
        """
        Log an artifact (file).
        
        Args:
            name: Name for the artifact
            source_path: Path to source file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Artifact source not found: {source_path}")
        
        dest = self._run_dir / 'artifacts' / name
        shutil.copy2(source, dest)
        self.run.artifacts[name] = str(dest)
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        self.run.tags[key] = value


def main():
    """Demo experiment tracking."""
    print("=" * 60)
    print("EXPERIMENT TRACKING DEMO")
    print("=" * 60)
    
    tracker = ExperimentTracker()
    
    # Create experiment
    tracker.create_experiment(
        "campaign_impact_analysis",
        "Testing different intervention dates and segments"
    )
    
    # Run 1
    with tracker.start_run("campaign_impact_analysis", "baseline_run") as run:
        run.log_params({
            'intervention_date': '2024-03-01',
            'segment': 'aggregated',
            'model': 'bayesian_ridge'
        })
        run.log_metrics({
            'cumulative_effect': 12500.50,
            'p_value': 0.003,
            'roi_percentage': 150.0
        })
        run.set_tag('type', 'baseline')
    
    # Run 2
    with tracker.start_run("campaign_impact_analysis", "email_segment") as run:
        run.log_params({
            'intervention_date': '2024-03-01',
            'segment': 'channel:email',
            'model': 'bayesian_ridge'
        })
        run.log_metrics({
            'cumulative_effect': 8500.00,
            'p_value': 0.015,
            'roi_percentage': 120.0
        })
        run.set_tag('type', 'segment')
    
    # Run 3
    with tracker.start_run("campaign_impact_analysis", "social_segment") as run:
        run.log_params({
            'intervention_date': '2024-03-01',
            'segment': 'channel:social',
            'model': 'bayesian_ridge'
        })
        run.log_metrics({
            'cumulative_effect': 15000.00,
            'p_value': 0.001,
            'roi_percentage': 200.0
        })
        run.set_tag('type', 'segment')
    
    # Compare runs
    print("\nRun Comparison:")
    comparison = tracker.compare_runs("campaign_impact_analysis")
    print(comparison.to_string(index=False))
    
    # Get best run
    best_run = tracker.get_best_run("campaign_impact_analysis", "roi_percentage")
    if best_run:
        print(f"\nBest run by ROI: {best_run.run_name} ({best_run.metrics['roi_percentage']:.1f}%)")
    
    print("\n✓ Experiment tracking demo completed!")


if __name__ == '__main__':
    main()
