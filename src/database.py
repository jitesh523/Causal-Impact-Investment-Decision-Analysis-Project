"""
Database Integration Module
============================

Provides database abstraction for storing and retrieving analysis data.
Supports SQLite for local development and PostgreSQL for production.

Author: Causal Impact Analysis Project
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import sqlite3
from contextlib import contextmanager


@dataclass
class AnalysisRecord:
    """Record for storing analysis results."""
    id: Optional[int]
    timestamp: str
    segment: str
    intervention_date: str
    campaign_cost: float
    cumulative_effect: float
    average_effect: float
    p_value: float
    roi_percentage: float
    net_profit: float
    parameters: str  # JSON string
    created_at: str


class DatabaseManager:
    """
    Database manager for storing causal impact analysis results.
    
    Supports SQLite for local/development use and PostgreSQL URLs
    for production deployment.
    
    Example:
        >>> db = DatabaseManager("sqlite:///analysis.db")
        >>> db.initialize()
        >>> db.save_analysis(results)
        >>> history = db.get_analysis_history(segment="channel:email")
    """
    
    def __init__(self, connection_string: str = None):
        """
        Initialize database manager.
        
        Args:
            connection_string: Database URL or path. Defaults to SQLite.
                - "sqlite:///path/to/db.sqlite" for SQLite
                - "postgresql://user:pass@host:port/db" for PostgreSQL
        """
        self.connection_string = connection_string or self._default_connection()
        self.db_type = self._detect_db_type()
        self._connection = None
        
    def _default_connection(self) -> str:
        """Get default SQLite connection string."""
        db_path = Path(__file__).parent.parent / 'data' / 'analysis_history.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    
    def _detect_db_type(self) -> str:
        """Detect database type from connection string."""
        if self.connection_string.startswith("postgresql"):
            return "postgresql"
        return "sqlite"
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        if self.db_type == "sqlite":
            db_path = self.connection_string.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()
        else:
            # PostgreSQL support (requires psycopg2)
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(self.connection_string)
                try:
                    yield conn
                    conn.commit()
                finally:
                    conn.close()
            except ImportError:
                raise ImportError("PostgreSQL requires psycopg2: pip install psycopg2-binary")
    
    def initialize(self):
        """Initialize database schema."""
        print("Initializing database...")
        
        create_analyses_table = """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            segment TEXT NOT NULL,
            intervention_date TEXT NOT NULL,
            campaign_cost REAL NOT NULL,
            cumulative_effect REAL NOT NULL,
            average_effect REAL NOT NULL,
            p_value REAL NOT NULL,
            roi_percentage REAL NOT NULL,
            net_profit REAL NOT NULL,
            parameters TEXT,
            created_at TEXT NOT NULL
        )
        """
        
        create_experiments_table = """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        )
        """
        
        create_runs_table = """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            run_name TEXT NOT NULL,
            parameters TEXT,
            metrics TEXT,
            artifacts TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            status TEXT DEFAULT 'running',
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
        """
        
        create_index = """
        CREATE INDEX IF NOT EXISTS idx_analyses_segment ON analyses(segment)
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_analyses_table)
            cursor.execute(create_experiments_table)
            cursor.execute(create_runs_table)
            cursor.execute(create_index)
        
        print("✓ Database initialized successfully")
    
    def save_analysis(
        self,
        segment: str,
        intervention_date: str,
        campaign_cost: float,
        impact_metrics: Dict[str, Any],
        financial_results: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save analysis results to database.
        
        Args:
            segment: Segment identifier
            intervention_date: Intervention date
            campaign_cost: Campaign cost
            impact_metrics: Dict with cumulative_effect, average_effect, p_value
            financial_results: Dict with roi_percentage, net_profit
            parameters: Optional dict of analysis parameters
        
        Returns:
            ID of inserted record
        """
        now = datetime.now().isoformat()
        
        insert_sql = """
        INSERT INTO analyses (
            timestamp, segment, intervention_date, campaign_cost,
            cumulative_effect, average_effect, p_value,
            roi_percentage, net_profit, parameters, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            now,
            segment,
            intervention_date,
            campaign_cost,
            impact_metrics.get('cumulative_effect', 0),
            impact_metrics.get('average_effect', 0),
            impact_metrics.get('p_value', 1),
            financial_results.get('roi_percentage', 0),
            financial_results.get('net_profit', 0),
            json.dumps(parameters) if parameters else None,
            now
        )
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(insert_sql, params)
            record_id = cursor.lastrowid
        
        print(f"✓ Analysis saved with ID: {record_id}")
        return record_id
    
    def get_analysis_history(
        self,
        segment: Optional[str] = None,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve analysis history.
        
        Args:
            segment: Optional segment filter
            limit: Maximum records to return
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            List of analysis records
        """
        query = "SELECT * FROM analyses WHERE 1=1"
        params = []
        
        if segment:
            query += " AND segment = ?"
            params.append(segment)
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific analysis by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
        
        return dict(row) if row else None
    
    def delete_analysis(self, analysis_id: int) -> bool:
        """Delete an analysis record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            deleted = cursor.rowcount > 0
        
        return deleted
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of stored analyses."""
        query = """
        SELECT 
            COUNT(*) as total_analyses,
            COUNT(DISTINCT segment) as unique_segments,
            AVG(cumulative_effect) as avg_effect,
            AVG(roi_percentage) as avg_roi,
            SUM(CASE WHEN p_value < 0.05 THEN 1 ELSE 0 END) as significant_count,
            MIN(created_at) as first_analysis,
            MAX(created_at) as last_analysis
        FROM analyses
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
        
        return dict(row) if row else {}
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all analyses to pandas DataFrame."""
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM analyses ORDER BY created_at DESC", conn)
        return df
    
    def import_from_dataframe(self, df: pd.DataFrame):
        """Import analyses from pandas DataFrame."""
        required_cols = ['segment', 'intervention_date', 'cumulative_effect', 'p_value']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for _, row in df.iterrows():
            self.save_analysis(
                segment=row['segment'],
                intervention_date=row['intervention_date'],
                campaign_cost=row.get('campaign_cost', 0),
                impact_metrics={
                    'cumulative_effect': row['cumulative_effect'],
                    'average_effect': row.get('average_effect', 0),
                    'p_value': row['p_value']
                },
                financial_results={
                    'roi_percentage': row.get('roi_percentage', 0),
                    'net_profit': row.get('net_profit', 0)
                }
            )


def main():
    """Demo database functionality."""
    print("=" * 60)
    print("DATABASE INTEGRATION DEMO")
    print("=" * 60)
    
    # Initialize database
    db = DatabaseManager()
    db.initialize()
    
    # Save sample analysis
    analysis_id = db.save_analysis(
        segment="channel:email",
        intervention_date="2024-03-01",
        campaign_cost=5000,
        impact_metrics={
            'cumulative_effect': 12500.50,
            'average_effect': 250.01,
            'p_value': 0.003
        },
        financial_results={
            'roi_percentage': 150.0,
            'net_profit': 7500.50
        },
        parameters={'model': 'bayesian_ridge', 'alpha': 0.05}
    )
    
    # Retrieve history
    print("\nRecent analyses:")
    history = db.get_analysis_history(limit=5)
    for record in history:
        print(f"  [{record['id']}] {record['segment']}: ${record['cumulative_effect']:,.2f}")
    
    # Get stats
    print("\nSummary Statistics:")
    stats = db.get_summary_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Database demo completed successfully!")


if __name__ == '__main__':
    main()
