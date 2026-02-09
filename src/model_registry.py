"""
Model Registry Module
=====================

Track, version, and manage trained models with metadata.
Provides model lifecycle management for production deployments.

Author: Causal Impact Analysis Project
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil


@dataclass
class ModelVersion:
    """Represents a model version."""
    model_id: str
    version: int
    name: str
    description: str
    created_at: str
    created_by: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: List[str]
    stage: str  # 'development', 'staging', 'production', 'archived'
    file_path: str
    file_hash: str


@dataclass
class ModelMetadata:
    """Model metadata for registry."""
    name: str
    description: str
    created_at: str
    updated_at: str
    versions: List[int]
    current_production_version: Optional[int]
    current_staging_version: Optional[int]


class ModelRegistry:
    """
    Model Registry for versioning and lifecycle management.
    
    Features:
    - Model versioning with automatic incrementing
    - Metadata and metrics tracking
    - Stage management (dev/staging/production)
    - Model comparison
    
    Example:
        >>> registry = ModelRegistry("./models")
        >>> version = registry.register(
        ...     model=trained_model,
        ...     name="causal_forest_v1",
        ...     metrics={'r2': 0.85, 'mae': 0.12}
        ... )
        >>> registry.transition_stage(version.model_id, version.version, 'production')
        >>> model = registry.load_production("causal_forest_v1")
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Directory to store models and metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self._metadata_file = self.registry_path / "registry.json"
        self._registry: Dict[str, ModelMetadata] = {}
        self._versions: Dict[str, Dict[int, ModelVersion]] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                data = json.load(f)
                for name, meta in data.get('models', {}).items():
                    self._registry[name] = ModelMetadata(**meta)
                    self._versions[name] = {}
                    
                    # Load version metadata
                    model_dir = self.registry_path / name
                    if model_dir.exists():
                        for version_file in model_dir.glob("v*/metadata.json"):
                            with open(version_file, 'r') as vf:
                                version_data = json.load(vf)
                                mv = ModelVersion(**version_data)
                                self._versions[name][mv.version] = mv
    
    def _save_registry(self):
        """Save registry to disk."""
        data = {
            'models': {name: asdict(meta) for name, meta in self._registry.items()},
            'updated_at': datetime.now().isoformat()
        }
        with open(self._metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_hash(self, model: Any) -> str:
        """Compute hash of serialized model."""
        serialized = pickle.dumps(model)
        return hashlib.md5(serialized).hexdigest()[:12]
    
    def register(
        self,
        model: Any,
        name: str,
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model: The model object to register
            name: Model name
            description: Model description
            metrics: Performance metrics
            parameters: Model parameters/hyperparameters
            tags: Tags for categorization
            created_by: User who created this version
        
        Returns:
            ModelVersion with version info
        """
        now = datetime.now().isoformat()
        
        # Get or create model metadata
        if name not in self._registry:
            self._registry[name] = ModelMetadata(
                name=name,
                description=description,
                created_at=now,
                updated_at=now,
                versions=[],
                current_production_version=None,
                current_staging_version=None
            )
            self._versions[name] = {}
        
        # Determine version number
        existing_versions = self._registry[name].versions
        version = max(existing_versions) + 1 if existing_versions else 1
        
        # Create version directory
        model_dir = self.registry_path / name / f"v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        file_hash = self._compute_hash(model)
        
        # Generate model ID
        model_id = f"{name}:{version}"
        
        # Create version record
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            created_at=now,
            created_by=created_by,
            metrics=metrics or {},
            parameters=parameters or {},
            tags=tags or [],
            stage='development',
            file_path=str(model_file),
            file_hash=file_hash
        )
        
        # Save version metadata
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(asdict(model_version), f, indent=2)
        
        # Update registry
        self._versions[name][version] = model_version
        self._registry[name].versions.append(version)
        self._registry[name].updated_at = now
        self._save_registry()
        
        print(f"✓ Registered {model_id} (stage: development)")
        return model_version
    
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: str
    ) -> ModelVersion:
        """
        Transition model to new stage.
        
        Args:
            name: Model name
            version: Version number
            stage: Target stage ('staging', 'production', 'archived')
        
        Returns:
            Updated ModelVersion
        """
        if stage not in ['development', 'staging', 'production', 'archived']:
            raise ValueError(f"Invalid stage: {stage}")
        
        if name not in self._versions or version not in self._versions[name]:
            raise ValueError(f"Model version not found: {name}:{version}")
        
        model_version = self._versions[name][version]
        old_stage = model_version.stage
        model_version.stage = stage
        
        # Update current pointers
        if stage == 'production':
            # Archive old production version
            old_prod = self._registry[name].current_production_version
            if old_prod and old_prod in self._versions[name]:
                self._versions[name][old_prod].stage = 'archived'
            self._registry[name].current_production_version = version
            
        elif stage == 'staging':
            self._registry[name].current_staging_version = version
        
        # Save metadata
        model_dir = self.registry_path / name / f"v{version}"
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(asdict(model_version), f, indent=2)
        
        self._save_registry()
        
        print(f"✓ Transitioned {name}:{version} from {old_stage} to {stage}")
        return model_version
    
    def load(self, name: str, version: Optional[int] = None) -> Any:
        """
        Load a model by name and version.
        
        Args:
            name: Model name
            version: Specific version (None for latest)
        
        Returns:
            Loaded model object
        """
        if name not in self._versions:
            raise ValueError(f"Model not found: {name}")
        
        if version is None:
            version = max(self._versions[name].keys())
        
        if version not in self._versions[name]:
            raise ValueError(f"Version not found: {name}:{version}")
        
        model_version = self._versions[name][version]
        
        with open(model_version.file_path, 'rb') as f:
            return pickle.load(f)
    
    def load_production(self, name: str) -> Any:
        """Load current production model."""
        if name not in self._registry:
            raise ValueError(f"Model not found: {name}")
        
        prod_version = self._registry[name].current_production_version
        if not prod_version:
            raise ValueError(f"No production version for: {name}")
        
        return self.load(name, prod_version)
    
    def load_staging(self, name: str) -> Any:
        """Load current staging model."""
        if name not in self._registry:
            raise ValueError(f"Model not found: {name}")
        
        staging_version = self._registry[name].current_staging_version
        if not staging_version:
            raise ValueError(f"No staging version for: {name}")
        
        return self.load(name, staging_version)
    
    def get_version(self, name: str, version: int) -> ModelVersion:
        """Get version metadata."""
        if name not in self._versions or version not in self._versions[name]:
            raise ValueError(f"Version not found: {name}:{version}")
        return self._versions[name][version]
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        result = []
        for name, meta in self._registry.items():
            result.append({
                'name': name,
                'description': meta.description,
                'versions': len(meta.versions),
                'latest_version': max(meta.versions) if meta.versions else None,
                'production_version': meta.current_production_version,
                'staging_version': meta.current_staging_version
            })
        return result
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        if name not in self._versions:
            return []
        
        return [
            {
                'version': v.version,
                'stage': v.stage,
                'created_at': v.created_at,
                'metrics': v.metrics,
                'tags': v.tags
            }
            for v in sorted(self._versions[name].values(), key=lambda x: x.version)
        ]
    
    def compare_versions(
        self,
        name: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        v1 = self.get_version(name, version1)
        v2 = self.get_version(name, version2)
        
        metrics_comparison = {}
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        
        for metric in all_metrics:
            val1 = v1.metrics.get(metric)
            val2 = v2.metrics.get(metric)
            if val1 is not None and val2 is not None:
                metrics_comparison[metric] = {
                    f'v{version1}': val1,
                    f'v{version2}': val2,
                    'diff': val2 - val1,
                    'pct_change': (val2 - val1) / val1 if val1 != 0 else None
                }
        
        return {
            'version1': version1,
            'version2': version2,
            'metrics_comparison': metrics_comparison,
            'parameters_changed': v1.parameters != v2.parameters,
            'created_at_1': v1.created_at,
            'created_at_2': v2.created_at
        }
    
    def delete_version(self, name: str, version: int) -> bool:
        """Delete a model version."""
        if name not in self._versions or version not in self._versions[name]:
            return False
        
        mv = self._versions[name][version]
        if mv.stage == 'production':
            raise ValueError("Cannot delete production model")
        
        # Remove files
        model_dir = self.registry_path / name / f"v{version}"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Update registry
        del self._versions[name][version]
        self._registry[name].versions.remove(version)
        self._save_registry()
        
        return True


def main():
    """Demo model registry."""
    print("=" * 60)
    print("MODEL REGISTRY DEMO")
    print("=" * 60)
    
    import tempfile
    from sklearn.linear_model import LinearRegression
    
    # Create temporary registry
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(tmpdir)
        
        # Register models
        print("\n1. Registering models...")
        
        model1 = LinearRegression()
        v1 = registry.register(
            model=model1,
            name="revenue_model",
            description="Linear regression for revenue prediction",
            metrics={'r2': 0.75, 'mae': 100.5},
            parameters={'fit_intercept': True},
            tags=['baseline', 'linear']
        )
        
        model2 = LinearRegression()
        v2 = registry.register(
            model=model2,
            name="revenue_model",
            description="Improved version",
            metrics={'r2': 0.82, 'mae': 85.2},
            parameters={'fit_intercept': True},
            tags=['improved']
        )
        
        # List models
        print("\n2. Registered models:")
        for m in registry.list_models():
            print(f"   {m['name']}: {m['versions']} version(s)")
        
        # Transition to production
        print("\n3. Stage transitions:")
        registry.transition_stage("revenue_model", 1, "staging")
        registry.transition_stage("revenue_model", 2, "production")
        
        # List versions
        print("\n4. Model versions:")
        for v in registry.list_versions("revenue_model"):
            print(f"   v{v['version']}: {v['stage']} (R²={v['metrics'].get('r2', 'N/A')})")
        
        # Compare versions
        print("\n5. Version comparison:")
        comparison = registry.compare_versions("revenue_model", 1, 2)
        for metric, vals in comparison['metrics_comparison'].items():
            print(f"   {metric}: v1={vals['v1']:.3f} → v2={vals['v2']:.3f} ({vals['diff']:+.3f})")
        
        # Load production model
        print("\n6. Loading production model...")
        prod_model = registry.load_production("revenue_model")
        print(f"   Loaded: {type(prod_model).__name__}")
    
    print("\n✓ Model registry demo completed!")


if __name__ == '__main__':
    main()
