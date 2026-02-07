"""
Scheduled Analysis Jobs Module
==============================

Provides scheduling capabilities for automated analysis runs.
Supports cron-like scheduling with persistence and monitoring.

Author: Causal Impact Analysis Project
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import schedule
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""
    job_id: str
    name: str
    schedule_expr: str  # e.g., 'daily', 'hourly', 'every 5 minutes'
    task_type: str
    parameters: Dict[str, Any]
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    error_count: int = 0


@dataclass 
class JobResult:
    """Result of a job execution."""
    job_id: str
    started_at: str
    completed_at: str
    success: bool
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    duration_seconds: float


class AnalysisScheduler:
    """
    Scheduler for automated analysis jobs.
    
    Supports:
    - Cron-like scheduling expressions
    - Persistent job configuration
    - Job history and monitoring
    - Error handling and retries
    
    Example:
        >>> scheduler = AnalysisScheduler()
        >>> scheduler.add_job(
        ...     name='daily_analysis',
        ...     schedule='daily',
        ...     task_type='causal_impact',
        ...     parameters={'segment': 'email'}
        ... )
        >>> scheduler.start()
    """
    
    SCHEDULE_TYPES = {
        'every minute': lambda job: schedule.every().minute.do(job),
        'every 5 minutes': lambda job: schedule.every(5).minutes.do(job),
        'every 15 minutes': lambda job: schedule.every(15).minutes.do(job),
        'every 30 minutes': lambda job: schedule.every(30).minutes.do(job),
        'hourly': lambda job: schedule.every().hour.do(job),
        'every 2 hours': lambda job: schedule.every(2).hours.do(job),
        'every 6 hours': lambda job: schedule.every(6).hours.do(job),
        'daily': lambda job: schedule.every().day.at("00:00").do(job),
        'daily at 9am': lambda job: schedule.every().day.at("09:00").do(job),
        'weekly': lambda job: schedule.every().monday.at("00:00").do(job),
        'monthly': lambda job: schedule.every(30).days.do(job),
    }
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        max_history: int = 100
    ):
        """
        Initialize scheduler.
        
        Args:
            config_dir: Directory for job configs and history
            max_history: Maximum job results to keep in history
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / 'scheduling'
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_history = max_history
        self._jobs: Dict[str, ScheduledJob] = {}
        self._job_handlers: Dict[str, Callable] = {}
        self._history: List[JobResult] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Load existing jobs
        self._load_jobs()
        
        # Register default handlers
        self._register_default_handlers()
    
    def _load_jobs(self):
        """Load jobs from config file."""
        jobs_file = self.config_dir / 'jobs.json'
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                data = json.load(f)
                for job_data in data.get('jobs', []):
                    job = ScheduledJob(**job_data)
                    self._jobs[job.job_id] = job
    
    def _save_jobs(self):
        """Save jobs to config file."""
        jobs_file = self.config_dir / 'jobs.json'
        with open(jobs_file, 'w') as f:
            json.dump({
                'jobs': [asdict(job) for job in self._jobs.values()]
            }, f, indent=2)
    
    def _register_default_handlers(self):
        """Register default task handlers."""
        self.register_handler('causal_impact', self._run_causal_impact)
        self.register_handler('segment_analysis', self._run_segment_analysis)
        self.register_handler('anomaly_check', self._run_anomaly_check)
        self.register_handler('report_generation', self._run_report_generation)
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register a task handler."""
        self._job_handlers[task_type] = handler
    
    def add_job(
        self,
        name: str,
        schedule_expr: str,
        task_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> ScheduledJob:
        """
        Add a new scheduled job.
        
        Args:
            name: Job name
            schedule_expr: Schedule expression (e.g., 'daily', 'hourly')
            task_type: Type of task to run
            parameters: Task parameters
            job_id: Optional custom job ID
        
        Returns:
            Created job
        """
        if schedule_expr not in self.SCHEDULE_TYPES:
            raise ValueError(f"Unknown schedule: {schedule_expr}. "
                           f"Use one of: {list(self.SCHEDULE_TYPES.keys())}")
        
        if task_type not in self._job_handlers:
            raise ValueError(f"Unknown task type: {task_type}. "
                           f"Registered types: {list(self._job_handlers.keys())}")
        
        job_id = job_id or f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            schedule_expr=schedule_expr,
            task_type=task_type,
            parameters=parameters or {},
            enabled=True
        )
        
        self._jobs[job_id] = job
        self._save_jobs()
        
        # Schedule the job
        self._schedule_job(job)
        
        logger.info(f"Added job: {name} ({schedule_expr})")
        return job
    
    def _schedule_job(self, job: ScheduledJob):
        """Schedule a job for execution."""
        if not job.enabled:
            return
        
        def wrapped_handler():
            return self._execute_job(job)
        
        scheduler_fn = self.SCHEDULE_TYPES.get(job.schedule_expr)
        if scheduler_fn:
            scheduler_fn(wrapped_handler)
    
    def _execute_job(self, job: ScheduledJob) -> JobResult:
        """Execute a scheduled job."""
        started_at = datetime.now()
        
        logger.info(f"Executing job: {job.name}")
        
        try:
            handler = self._job_handlers.get(job.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {job.task_type}")
            
            result = handler(job.parameters)
            
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()
            
            job_result = JobResult(
                job_id=job.job_id,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                success=True,
                result=result,
                error=None,
                duration_seconds=duration
            )
            
            job.last_run = completed_at.isoformat()
            job.run_count += 1
            
            logger.info(f"Job {job.name} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()
            
            job_result = JobResult(
                job_id=job.job_id,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                success=False,
                result=None,
                error=str(e),
                duration_seconds=duration
            )
            
            job.error_count += 1
            
            logger.error(f"Job {job.name} failed: {e}")
        
        # Update history
        self._history.append(job_result)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        
        self._save_jobs()
        return job_result
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save_jobs()
            return True
        return False
    
    def enable_job(self, job_id: str):
        """Enable a job."""
        if job_id in self._jobs:
            self._jobs[job_id].enabled = True
            self._schedule_job(self._jobs[job_id])
            self._save_jobs()
    
    def disable_job(self, job_id: str):
        """Disable a job."""
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            self._save_jobs()
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs."""
        return [asdict(job) for job in self._jobs.values()]
    
    def get_history(self, job_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get job execution history."""
        history = self._history
        
        if job_id:
            history = [h for h in history if h.job_id == job_id]
        
        return [asdict(h) for h in history[-limit:]]
    
    def start(self, blocking: bool = False):
        """
        Start the scheduler.
        
        Args:
            blocking: If True, block the main thread
        """
        if self._running:
            return
        
        self._running = True
        
        # Schedule all enabled jobs
        for job in self._jobs.values():
            if job.enabled:
                self._schedule_job(job)
        
        logger.info("Scheduler started")
        
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
    
    def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """Stop the scheduler."""
        self._running = False
        schedule.clear()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def run_now(self, job_id: str) -> Optional[JobResult]:
        """Manually trigger a job."""
        job = self._jobs.get(job_id)
        if job:
            return self._execute_job(job)
        return None
    
    # Default task handlers
    
    def _run_causal_impact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run causal impact analysis."""
        # This would integrate with the actual analysis modules
        logger.info(f"Running causal impact analysis with params: {params}")
        
        # Simulated result
        return {
            'status': 'completed',
            'segment': params.get('segment', 'all'),
            'effect': 12500.50,
            'p_value': 0.003
        }
    
    def _run_segment_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run segment analysis."""
        logger.info(f"Running segment analysis with params: {params}")
        
        return {
            'status': 'completed',
            'segments_analyzed': params.get('segments', []),
            'results_count': 5
        }
    
    def _run_anomaly_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run anomaly detection."""
        logger.info(f"Running anomaly check with params: {params}")
        
        return {
            'status': 'completed',
            'anomalies_found': 0,
            'metrics_checked': params.get('metrics', ['roi', 'effect'])
        }
    
    def _run_report_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis report."""
        logger.info(f"Generating report with params: {params}")
        
        return {
            'status': 'completed',
            'report_path': f"/output/report_{datetime.now().strftime('%Y%m%d')}.pdf"
        }


def main():
    """Demo scheduled jobs."""
    print("=" * 60)
    print("SCHEDULED JOBS DEMO")
    print("=" * 60)
    
    scheduler = AnalysisScheduler()
    
    # Add some jobs
    print("\nAdding scheduled jobs...")
    
    scheduler.add_job(
        name='Daily Impact Analysis',
        schedule_expr='daily',
        task_type='causal_impact',
        parameters={'segment': 'all', 'metric': 'revenue_usd'}
    )
    
    scheduler.add_job(
        name='Hourly Anomaly Check',
        schedule_expr='hourly',
        task_type='anomaly_check',
        parameters={'metrics': ['roi', 'effect', 'p_value']}
    )
    
    scheduler.add_job(
        name='Weekly Report',
        schedule_expr='weekly',
        task_type='report_generation',
        parameters={'format': 'pdf', 'include_charts': True}
    )
    
    # List jobs
    print("\nScheduled Jobs:")
    for job in scheduler.list_jobs():
        print(f"  [{job['job_id']}] {job['name']} - {job['schedule_expr']}")
    
    # Run a job manually
    print("\nRunning job manually...")
    job_id = list(scheduler._jobs.keys())[0]
    result = scheduler.run_now(job_id)
    
    if result:
        print(f"  Status: {'Success' if result.success else 'Failed'}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        if result.result:
            print(f"  Result: {result.result}")
    
    # Show history
    print("\nJob History:")
    for h in scheduler.get_history(limit=5):
        status = '✓' if h['success'] else '✗'
        print(f"  {status} {h['job_id']}: {h['started_at'][:19]}")
    
    print("\n✓ Scheduled jobs demo completed!")


if __name__ == '__main__':
    main()
