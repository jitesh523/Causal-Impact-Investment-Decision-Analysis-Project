"""
Data Upload Interface Module
============================

Provides a web interface for uploading datasets directly to the system.
Supports CSV, Parquet, Excel, and JSON formats with validation.

Author: Causal Impact Analysis Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib


@dataclass
class UploadResult:
    """Result of a data upload."""
    success: bool
    filename: str
    file_path: Optional[str]
    rows: int
    columns: int
    size_bytes: int
    schema: Dict[str, str]
    validation_errors: List[str]
    warnings: List[str]
    upload_time: str


@dataclass
class DataValidationRule:
    """Rule for validating uploaded data."""
    column: str
    rule_type: str  # 'required', 'type', 'range', 'regex', 'unique'
    params: Dict[str, Any]
    error_message: str


class DataUploader:
    """
    Handles data file uploads with validation and storage.
    
    Supports:
    - Multiple file formats (CSV, Parquet, Excel, JSON)
    - Schema validation
    - Automatic type inference
    - Data quality checks
    
    Example:
        >>> uploader = DataUploader(upload_dir='./uploads')
        >>> result = uploader.upload(file_content, 'dataset.csv')
        >>> if result.success:
        ...     df = uploader.load(result.file_path)
    """
    
    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json'
    }
    
    def __init__(
        self,
        upload_dir: str = './uploads',
        max_file_size_mb: int = 100,
        allowed_formats: Optional[List[str]] = None
    ):
        """
        Initialize uploader.
        
        Args:
            upload_dir: Directory to store uploaded files
            max_file_size_mb: Maximum file size in MB
            allowed_formats: List of allowed file extensions
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.allowed_formats = allowed_formats or list(self.SUPPORTED_FORMATS.keys())
        
        self._validation_rules: List[DataValidationRule] = []
        self._metadata_file = self.upload_dir / 'metadata.json'
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load upload metadata."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        return {'uploads': {}}
    
    def _save_metadata(self):
        """Save upload metadata."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute file hash for deduplication."""
        return hashlib.md5(content).hexdigest()[:12]
    
    def add_validation_rule(self, rule: DataValidationRule):
        """Add a validation rule."""
        self._validation_rules.append(rule)
    
    def set_required_columns(self, columns: List[str]):
        """Set required columns."""
        for col in columns:
            self._validation_rules.append(DataValidationRule(
                column=col,
                rule_type='required',
                params={},
                error_message=f"Required column '{col}' is missing"
            ))
    
    def _validate_file(self, content: bytes, filename: str) -> Tuple[bool, List[str]]:
        """Validate file before processing."""
        errors = []
        
        # Check file size
        if len(content) > self.max_file_size:
            errors.append(f"File size exceeds limit of {self.max_file_size // (1024*1024)}MB")
        
        # Check format
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_formats:
            errors.append(f"Unsupported format: {ext}. Allowed: {self.allowed_formats}")
        
        return len(errors) == 0, errors
    
    def _validate_data(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate data against rules."""
        errors = []
        warnings = []
        
        for rule in self._validation_rules:
            if rule.rule_type == 'required':
                if rule.column not in df.columns:
                    errors.append(rule.error_message)
            
            elif rule.rule_type == 'type' and rule.column in df.columns:
                expected_type = rule.params.get('dtype')
                if expected_type:
                    try:
                        df[rule.column].astype(expected_type)
                    except (ValueError, TypeError):
                        errors.append(rule.error_message)
            
            elif rule.rule_type == 'range' and rule.column in df.columns:
                min_val = rule.params.get('min')
                max_val = rule.params.get('max')
                if min_val is not None and df[rule.column].min() < min_val:
                    errors.append(rule.error_message)
                if max_val is not None and df[rule.column].max() > max_val:
                    errors.append(rule.error_message)
            
            elif rule.rule_type == 'unique' and rule.column in df.columns:
                if df[rule.column].duplicated().any():
                    warnings.append(f"Column '{rule.column}' has duplicate values")
        
        # Standard quality checks
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            warnings.append(f"Columns with null values: {null_cols}")
        
        return errors, warnings
    
    def _read_file(self, content: bytes, filename: str) -> pd.DataFrame:
        """Read file content into DataFrame."""
        import io
        ext = Path(filename).suffix.lower()
        format_type = self.SUPPORTED_FORMATS.get(ext)
        
        if format_type == 'csv':
            return pd.read_csv(io.BytesIO(content))
        elif format_type == 'parquet':
            return pd.read_parquet(io.BytesIO(content))
        elif format_type == 'excel':
            return pd.read_excel(io.BytesIO(content))
        elif format_type == 'json':
            return pd.read_json(io.BytesIO(content))
        else:
            raise ValueError(f"Unknown format: {ext}")
    
    def upload(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict] = None
    ) -> UploadResult:
        """
        Upload a data file.
        
        Args:
            content: File content as bytes
            filename: Original filename
            metadata: Optional metadata to store
        
        Returns:
            UploadResult with status and details
        """
        timestamp = datetime.now()
        
        # Validate file
        valid, file_errors = self._validate_file(content, filename)
        if not valid:
            return UploadResult(
                success=False,
                filename=filename,
                file_path=None,
                rows=0,
                columns=0,
                size_bytes=len(content),
                schema={},
                validation_errors=file_errors,
                warnings=[],
                upload_time=timestamp.isoformat()
            )
        
        try:
            # Parse file
            df = self._read_file(content, filename)
            
            # Validate data
            data_errors, warnings = self._validate_data(df)
            
            if data_errors:
                return UploadResult(
                    success=False,
                    filename=filename,
                    file_path=None,
                    rows=len(df),
                    columns=len(df.columns),
                    size_bytes=len(content),
                    schema={col: str(dtype) for col, dtype in df.dtypes.items()},
                    validation_errors=data_errors,
                    warnings=warnings,
                    upload_time=timestamp.isoformat()
                )
            
            # Generate unique filename
            file_hash = self._compute_hash(content)
            base_name = Path(filename).stem
            ext = Path(filename).suffix
            unique_name = f"{base_name}_{file_hash}{ext}"
            file_path = self.upload_dir / unique_name
            
            # Save file
            if ext.lower() == '.csv':
                df.to_csv(file_path, index=False)
            elif ext.lower() == '.parquet':
                df.to_parquet(file_path, index=False)
            else:
                with open(file_path, 'wb') as f:
                    f.write(content)
            
            # Update metadata
            self._metadata['uploads'][unique_name] = {
                'original_name': filename,
                'upload_time': timestamp.isoformat(),
                'rows': len(df),
                'columns': len(df.columns),
                'size_bytes': len(content),
                'schema': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'custom_metadata': metadata or {}
            }
            self._save_metadata()
            
            return UploadResult(
                success=True,
                filename=filename,
                file_path=str(file_path),
                rows=len(df),
                columns=len(df.columns),
                size_bytes=len(content),
                schema={col: str(dtype) for col, dtype in df.dtypes.items()},
                validation_errors=[],
                warnings=warnings,
                upload_time=timestamp.isoformat()
            )
            
        except Exception as e:
            return UploadResult(
                success=False,
                filename=filename,
                file_path=None,
                rows=0,
                columns=0,
                size_bytes=len(content),
                schema={},
                validation_errors=[f"Error processing file: {str(e)}"],
                warnings=[],
                upload_time=timestamp.isoformat()
            )
    
    def load(self, file_path: str) -> pd.DataFrame:
        """Load an uploaded file."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(path)
        elif ext == '.parquet':
            return pd.read_parquet(path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif ext == '.json':
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def list_uploads(self) -> List[Dict]:
        """List all uploaded files."""
        uploads = []
        for name, info in self._metadata.get('uploads', {}).items():
            uploads.append({
                'name': name,
                'original_name': info['original_name'],
                'upload_time': info['upload_time'],
                'rows': info['rows'],
                'columns': info['columns'],
                'size_bytes': info['size_bytes']
            })
        return sorted(uploads, key=lambda x: x['upload_time'], reverse=True)
    
    def delete(self, filename: str) -> bool:
        """Delete an uploaded file."""
        file_path = self.upload_dir / filename
        if file_path.exists():
            file_path.unlink()
            if filename in self._metadata.get('uploads', {}):
                del self._metadata['uploads'][filename]
                self._save_metadata()
            return True
        return False
    
    def get_preview(self, filename: str, n_rows: int = 5) -> pd.DataFrame:
        """Get preview of uploaded file."""
        file_path = self.upload_dir / filename
        df = self.load(str(file_path))
        return df.head(n_rows)


def create_streamlit_upload_interface():
    """
    Create Streamlit upload interface code.
    
    Returns code snippet for Streamlit data upload page.
    """
    return '''
# Streamlit Data Upload Page
# Add to your dashboard.py or create pages/upload.py

import streamlit as st
from src.data_upload import DataUploader

def render_upload_page():
    st.title("📤 Data Upload")
    
    uploader = DataUploader(upload_dir='./uploads')
    uploader.set_required_columns(['date', 'user_id', 'revenue_usd'])
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'parquet', 'xlsx', 'json'],
        help="Supported formats: CSV, Parquet, Excel, JSON"
    )
    
    if uploaded_file:
        # Show file info
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        # Upload button
        if st.button("Upload and Validate"):
            with st.spinner("Processing..."):
                result = uploader.upload(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
            
            if result.success:
                st.success(f"✓ Uploaded successfully!")
                st.write(f"Rows: {result.rows}, Columns: {result.columns}")
                
                # Show preview
                df = uploader.load(result.file_path)
                st.dataframe(df.head(10))
                
                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            else:
                st.error("Upload failed!")
                for e in result.validation_errors:
                    st.error(e)
    
    # Show existing uploads
    st.subheader("Previous Uploads")
    uploads = uploader.list_uploads()
    
    if uploads:
        for u in uploads:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(u['original_name'])
            col2.write(f"{u['rows']} rows")
            if col3.button("Delete", key=u['name']):
                uploader.delete(u['name'])
                st.experimental_rerun()
    else:
        st.info("No uploads yet")

if __name__ == "__main__":
    render_upload_page()
'''


def main():
    """Demo data upload."""
    print("=" * 60)
    print("DATA UPLOAD INTERFACE DEMO")
    print("=" * 60)
    
    # Create uploader
    uploader = DataUploader(upload_dir='./demo_uploads')
    
    # Set validation rules
    uploader.set_required_columns(['date', 'value'])
    
    # Create sample data
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'value': np.random.randn(100) * 100 + 500,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Convert to CSV bytes
    csv_content = df.to_csv(index=False).encode()
    
    # Upload
    print("\n1. Uploading sample CSV...")
    result = uploader.upload(csv_content, 'sample_data.csv')
    
    print(f"   Success: {result.success}")
    print(f"   Rows: {result.rows}, Columns: {result.columns}")
    print(f"   File: {result.file_path}")
    
    if result.warnings:
        print(f"   Warnings: {result.warnings}")
    
    # List uploads
    print("\n2. Listing uploads...")
    for u in uploader.list_uploads():
        print(f"   - {u['original_name']} ({u['rows']} rows)")
    
    # Preview
    print("\n3. Preview:")
    preview = uploader.get_preview('sample_data.csv' if result.file_path else '', n_rows=3)
    print(preview.to_string())
    
    # Test invalid upload
    print("\n4. Testing validation...")
    bad_df = pd.DataFrame({'x': [1, 2, 3]})  # Missing required columns
    bad_content = bad_df.to_csv(index=False).encode()
    bad_result = uploader.upload(bad_content, 'bad_data.csv')
    
    print(f"   Success: {bad_result.success}")
    print(f"   Errors: {bad_result.validation_errors}")
    
    # Cleanup
    import shutil
    shutil.rmtree('./demo_uploads', ignore_errors=True)
    
    print("\n✓ Data upload demo completed!")


if __name__ == '__main__':
    main()
