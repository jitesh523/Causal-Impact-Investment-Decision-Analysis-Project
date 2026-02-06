"""
Pytest configuration and fixtures for testing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def mock_dataset_file(tmp_path_factory):
    """
    Create a temporary Excel file with mock data matching the expected schema.
    This allows tests to run without the real Dataset.xlsx file.
    """
    # Create temporary directory for the session
    tmp_dir = tmp_path_factory.mktemp("data")
    dataset_path = tmp_dir / "Dataset.xlsx"
    
    # Generate mock data matching the expected schema
    np.random.seed(42)  # For reproducibility
    
    n_samples = 1000
    
    # Create mock dataset
    data = {
        'user_id': [f'user_{i:05d}' for i in range(n_samples)],
        'treatment_exposed': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
        'impressions': np.random.randint(100, 10000, size=n_samples),
        'clicks': np.random.randint(10, 1000, size=n_samples),
        'spend_usd': np.random.uniform(10, 500, size=n_samples).round(2),
        'conversion': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'revenue_usd': np.random.uniform(50, 2000, size=n_samples).round(2),
        'roi': np.random.uniform(-0.5, 3.0, size=n_samples).round(2),
        'channel': np.random.choice(['Google', 'Facebook', 'Instagram', 'Twitter'], size=n_samples),
        'country': np.random.choice(['US', 'UK', 'Canada', 'Germany'], size=n_samples),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save to Excel
    df.to_excel(dataset_path, index=False)
    
    yield dataset_path
    
    # Cleanup (optional, tmp_path_factory handles this)
    # shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_config_file(tmp_path_factory, mock_dataset_file):
    """
    Create a temporary config.yaml that points to the mock dataset.
    """
    tmp_dir = tmp_path_factory.mktemp("config")
    config_path = tmp_dir / "config.yaml"
    
    # Create config content pointing to mock dataset
    config_content = f"""# Causal Impact Project Configuration

data:
  raw_path: "{mock_dataset_file}"
  processed_dir: "data/processed"
  output_dir: "reports"
  figures_dir: "reports/figures"

dates:
  start_date: "2024-01-01"
  intervention_date: "2024-03-15"
  total_days: 150 
  post_period_days: 76

campaign:
  cost: 5000.0
  currency: "USD"

model:
  alpha: 0.05
  n_iter: 300

segments:
  - "channel"
  - "country"
  - "device"
"""
    
    config_path.write_text(config_content)
    
    yield config_path
