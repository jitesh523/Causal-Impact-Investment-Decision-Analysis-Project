"""
Cloud Data Connectors Module
============================

Provides connectors for cloud storage and data warehouses.
Supports AWS S3, Google Cloud Storage, and BigQuery.

Author: Causal Impact Analysis Project
"""

import os
import json
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pandas as pd
import io


@dataclass
class CloudStorageConfig:
    """Configuration for cloud storage."""
    provider: str  # 's3', 'gcs', 'azure'
    bucket: str
    prefix: str = ''
    credentials_path: Optional[str] = None
    region: Optional[str] = None


class S3Connector:
    """
    AWS S3 connector for data loading and saving.
    
    Example:
        >>> connector = S3Connector(bucket='my-bucket', prefix='data/')
        >>> df = connector.read_csv('dataset.csv')
        >>> connector.write_parquet(df, 'output/results.parquet')
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = '',
        region: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize S3 connector.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all operations
            region: AWS region
            credentials_path: Path to credentials file
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        
        self._client = None
        self._init_client(credentials_path)
    
    def _init_client(self, credentials_path: Optional[str]):
        """Initialize S3 client."""
        try:
            import boto3
            
            if credentials_path:
                # Load credentials from file
                with open(credentials_path, 'r') as f:
                    creds = json.load(f)
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=creds.get('access_key'),
                    aws_secret_access_key=creds.get('secret_key'),
                    region_name=self.region
                )
            else:
                # Use default credentials (env vars, IAM role, etc.)
                self._client = boto3.client('s3', region_name=self.region)
                
        except ImportError:
            print("Warning: boto3 not installed. Install with: pip install boto3")
            self._client = None
    
    def _get_key(self, path: str) -> str:
        """Get full S3 key with prefix."""
        return self.prefix + path.lstrip('/')
    
    def list_objects(self, path: str = '', suffix: Optional[str] = None) -> List[str]:
        """List objects in path."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        prefix = self._get_key(path)
        
        response = self._client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )
        
        keys = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if suffix is None or key.endswith(suffix):
                keys.append(key[len(self.prefix):])
        
        return keys
    
    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Read CSV from S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(path)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        
        return pd.read_csv(io.BytesIO(response['Body'].read()), **kwargs)
    
    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet from S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(path)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        
        return pd.read_parquet(io.BytesIO(response['Body'].read()), **kwargs)
    
    def read_json(self, path: str) -> Dict:
        """Read JSON from S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(path)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        
        return json.loads(response['Body'].read().decode('utf-8'))
    
    def write_csv(self, df: pd.DataFrame, path: str, **kwargs):
        """Write DataFrame as CSV to S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(path)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, **kwargs)
        buffer.seek(0)
        
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue()
        )
    
    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs):
        """Write DataFrame as Parquet to S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(path)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, **kwargs)
        buffer.seek(0)
        
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue()
        )
    
    def write_json(self, data: Dict, path: str):
        """Write JSON to S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(path)
        
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
    
    def download_file(self, remote_path: str, local_path: str):
        """Download file from S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._client.download_file(self.bucket, key, local_path)
    
    def upload_file(self, local_path: str, remote_path: str):
        """Upload file to S3."""
        if not self._client:
            raise RuntimeError("S3 client not initialized")
        
        key = self._get_key(remote_path)
        
        self._client.upload_file(local_path, self.bucket, key)


class GCSConnector:
    """
    Google Cloud Storage connector.
    
    Example:
        >>> connector = GCSConnector(bucket='my-bucket')
        >>> df = connector.read_csv('data/input.csv')
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = '',
        credentials_path: Optional[str] = None
    ):
        """
        Initialize GCS connector.
        
        Args:
            bucket: GCS bucket name
            prefix: Blob prefix for all operations
            credentials_path: Path to service account JSON
        """
        self.bucket_name = bucket
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        
        self._client = None
        self._bucket = None
        self._init_client(credentials_path)
    
    def _init_client(self, credentials_path: Optional[str]):
        """Initialize GCS client."""
        try:
            from google.cloud import storage
            
            if credentials_path:
                self._client = storage.Client.from_service_account_json(credentials_path)
            else:
                self._client = storage.Client()
            
            self._bucket = self._client.bucket(self.bucket_name)
            
        except ImportError:
            print("Warning: google-cloud-storage not installed. "
                  "Install with: pip install google-cloud-storage")
            self._client = None
    
    def _get_blob_name(self, path: str) -> str:
        """Get full blob name with prefix."""
        return self.prefix + path.lstrip('/')
    
    def list_blobs(self, path: str = '', suffix: Optional[str] = None) -> List[str]:
        """List blobs in path."""
        if not self._client:
            raise RuntimeError("GCS client not initialized")
        
        prefix = self._get_blob_name(path)
        blobs = self._client.list_blobs(self.bucket_name, prefix=prefix)
        
        names = []
        for blob in blobs:
            if suffix is None or blob.name.endswith(suffix):
                names.append(blob.name[len(self.prefix):])
        
        return names
    
    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Read CSV from GCS."""
        if not self._bucket:
            raise RuntimeError("GCS client not initialized")
        
        blob_name = self._get_blob_name(path)
        blob = self._bucket.blob(blob_name)
        content = blob.download_as_bytes()
        
        return pd.read_csv(io.BytesIO(content), **kwargs)
    
    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet from GCS."""
        if not self._bucket:
            raise RuntimeError("GCS client not initialized")
        
        blob_name = self._get_blob_name(path)
        blob = self._bucket.blob(blob_name)
        content = blob.download_as_bytes()
        
        return pd.read_parquet(io.BytesIO(content), **kwargs)
    
    def write_csv(self, df: pd.DataFrame, path: str, **kwargs):
        """Write DataFrame as CSV to GCS."""
        if not self._bucket:
            raise RuntimeError("GCS client not initialized")
        
        blob_name = self._get_blob_name(path)
        blob = self._bucket.blob(blob_name)
        
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, **kwargs)
        blob.upload_from_string(buffer.getvalue(), content_type='text/csv')
    
    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs):
        """Write DataFrame as Parquet to GCS."""
        if not self._bucket:
            raise RuntimeError("GCS client not initialized")
        
        blob_name = self._get_blob_name(path)
        blob = self._bucket.blob(blob_name)
        
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, **kwargs)
        blob.upload_from_string(buffer.getvalue(), content_type='application/octet-stream')


class BigQueryConnector:
    """
    BigQuery connector for data warehouse access.
    
    Example:
        >>> bq = BigQueryConnector(project='my-project')
        >>> df = bq.query("SELECT * FROM dataset.table LIMIT 1000")
    """
    
    def __init__(
        self,
        project: str,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize BigQuery connector.
        
        Args:
            project: GCP project ID
            credentials_path: Path to service account JSON
        """
        self.project = project
        self._client = None
        self._init_client(credentials_path)
    
    def _init_client(self, credentials_path: Optional[str]):
        """Initialize BigQuery client."""
        try:
            from google.cloud import bigquery
            
            if credentials_path:
                self._client = bigquery.Client.from_service_account_json(
                    credentials_path,
                    project=self.project
                )
            else:
                self._client = bigquery.Client(project=self.project)
                
        except ImportError:
            print("Warning: google-cloud-bigquery not installed. "
                  "Install with: pip install google-cloud-bigquery")
            self._client = None
    
    def query(self, sql: str, **kwargs) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        if not self._client:
            raise RuntimeError("BigQuery client not initialized")
        
        return self._client.query(sql).to_dataframe(**kwargs)
    
    def query_to_table(
        self,
        sql: str,
        destination: str,
        write_disposition: str = 'WRITE_TRUNCATE'
    ):
        """Execute query and write results to table."""
        if not self._client:
            raise RuntimeError("BigQuery client not initialized")
        
        from google.cloud import bigquery
        
        job_config = bigquery.QueryJobConfig(
            destination=destination,
            write_disposition=write_disposition
        )
        
        job = self._client.query(sql, job_config=job_config)
        job.result()  # Wait for completion
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        destination: str,
        write_disposition: str = 'WRITE_TRUNCATE'
    ):
        """Load DataFrame to BigQuery table."""
        if not self._client:
            raise RuntimeError("BigQuery client not initialized")
        
        from google.cloud import bigquery
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition
        )
        
        job = self._client.load_table_from_dataframe(
            df, destination, job_config=job_config
        )
        job.result()
    
    def list_tables(self, dataset: str) -> List[str]:
        """List tables in dataset."""
        if not self._client:
            raise RuntimeError("BigQuery client not initialized")
        
        tables = self._client.list_tables(dataset)
        return [table.table_id for table in tables]


class DataConnectorFactory:
    """Factory for creating data connectors."""
    
    @staticmethod
    def create(
        provider: str,
        **kwargs
    ) -> Union[S3Connector, GCSConnector, BigQueryConnector]:
        """
        Create a data connector.
        
        Args:
            provider: 's3', 'gcs', or 'bigquery'
            **kwargs: Provider-specific arguments
        
        Returns:
            Configured connector
        """
        if provider == 's3':
            return S3Connector(**kwargs)
        elif provider == 'gcs':
            return GCSConnector(**kwargs)
        elif provider == 'bigquery':
            return BigQueryConnector(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def from_config(config: CloudStorageConfig) -> Union[S3Connector, GCSConnector]:
        """Create connector from config."""
        if config.provider == 's3':
            return S3Connector(
                bucket=config.bucket,
                prefix=config.prefix,
                region=config.region,
                credentials_path=config.credentials_path
            )
        elif config.provider == 'gcs':
            return GCSConnector(
                bucket=config.bucket,
                prefix=config.prefix,
                credentials_path=config.credentials_path
            )
        else:
            raise ValueError(f"Unknown provider: {config.provider}")


def main():
    """Demo cloud connectors."""
    print("=" * 60)
    print("CLOUD DATA CONNECTORS DEMO")
    print("=" * 60)
    
    # Demo S3 connector (mock)
    print("\n1. S3 Connector:")
    print("-" * 40)
    print("   S3Connector(bucket='my-bucket', prefix='data/')")
    print("   Methods: read_csv, read_parquet, write_csv, write_parquet")
    print("   Note: Requires boto3 and AWS credentials")
    
    # Demo GCS connector (mock)
    print("\n2. GCS Connector:")
    print("-" * 40)
    print("   GCSConnector(bucket='my-bucket')")
    print("   Methods: read_csv, read_parquet, list_blobs")
    print("   Note: Requires google-cloud-storage")
    
    # Demo BigQuery connector (mock)
    print("\n3. BigQuery Connector:")
    print("-" * 40)
    print("   BigQueryConnector(project='my-project')")
    print("   Methods: query, load_dataframe, list_tables")
    print("   Note: Requires google-cloud-bigquery")
    
    # Demo factory
    print("\n4. Connector Factory:")
    print("-" * 40)
    print("   DataConnectorFactory.create('s3', bucket='my-bucket')")
    print("   DataConnectorFactory.from_config(config)")
    
    print("\n✓ Cloud connectors demo completed!")
    print("\nInstall dependencies:")
    print("  pip install boto3                    # For S3")
    print("  pip install google-cloud-storage     # For GCS")
    print("  pip install google-cloud-bigquery    # For BigQuery")


if __name__ == '__main__':
    main()
