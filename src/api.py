"""
REST API Module (FastAPI)
=========================

Provides RESTful API endpoints for programmatic access to causal impact
analysis. Includes endpoints for running analysis, getting results,
and generating reports.

Author: Causal Impact Analysis Project
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import date, datetime
from enum import Enum
import yaml
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import DataPipeline
from causal_analysis import CausalAnalyzer
from financial_analysis import FinancialAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Causal Impact Analysis API",
    description="RESTful API for causal impact analysis and investment decision support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============== Pydantic Models ==============

class SegmentType(str, Enum):
    aggregated = "aggregated"
    channel = "channel"
    region = "region"
    customer_segment = "customer_segment"


class AnalysisRequest(BaseModel):
    """Request model for running causal analysis."""
    intervention_date: str = Field(
        ..., 
        description="Intervention date in YYYY-MM-DD format",
        example="2024-03-01"
    )
    segment_type: Optional[str] = Field(
        None,
        description="Segment column to analyze (None for aggregated)"
    )
    segment_value: Optional[str] = Field(
        None,
        description="Specific segment value to analyze"
    )
    campaign_cost: float = Field(
        5000.0,
        description="Campaign cost for ROI calculation",
        ge=0
    )


class ImpactMetrics(BaseModel):
    """Response model for impact metrics."""
    segment: str
    cumulative_effect: float
    cumulative_effect_pct: float
    average_effect: float
    p_value: float
    significant: bool
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class FinancialMetrics(BaseModel):
    """Response model for financial metrics."""
    roi_percentage: float
    roi_ratio: float
    net_profit: float
    cost_per_incremental: float
    payback_period_days: Optional[float] = None


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    status: str
    timestamp: str
    segment: str
    impact_metrics: ImpactMetrics
    financial_metrics: FinancialMetrics
    summary: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    config_loaded: bool


class SegmentListResponse(BaseModel):
    """List of available segments."""
    segment_type: str
    values: List[str]


# ============== Global State ==============

_pipeline = None
_config = None


def get_config():
    """Load and cache configuration."""
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                _config = yaml.safe_load(f)
        else:
            _config = {
                'dates': {'intervention_date': '2024-03-01'},
                'campaign': {'cost': 5000},
                'segments': ['channel', 'region']
            }
    return _config


def get_pipeline():
    """Get or create data pipeline."""
    global _pipeline
    if _pipeline is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            _pipeline = DataPipeline(str(config_path))
            _pipeline.load_data().clean_data()
    return _pipeline


# ============== API Endpoints ==============

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Causal Impact Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns current status and configuration state.
    """
    config = get_config()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        config_loaded=config is not None
    )


@app.get("/segments", response_model=List[SegmentListResponse], tags=["Data"])
async def list_segments():
    """
    List available segment types and their values.
    
    Returns all segment columns and their unique values from the dataset.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    config = get_config()
    segment_cols = config.get('segments', ['channel', 'region'])
    
    result = []
    for seg_col in segment_cols:
        if seg_col in pipeline.cleaned_data.columns:
            values = pipeline.cleaned_data[seg_col].unique().tolist()
            result.append(SegmentListResponse(
                segment_type=seg_col,
                values=sorted(str(v) for v in values)
            ))
    
    return result


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def run_analysis(request: AnalysisRequest):
    """
    Run causal impact analysis.
    
    Performs Bayesian structural time-series analysis to estimate
    the causal impact of an intervention on the outcome variable.
    
    **Parameters:**
    - `intervention_date`: Date when intervention started (YYYY-MM-DD)
    - `segment_type`: Optional segment column for filtering
    - `segment_value`: Optional specific segment value
    - `campaign_cost`: Cost of campaign for ROI calculation
    
    **Returns:**
    - Impact metrics (cumulative effect, p-value, etc.)
    - Financial metrics (ROI, net profit, etc.)
    - Text summary
    """
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    config = get_config()
    
    try:
        # Determine segment
        segment = None
        segment_name = "aggregated"
        
        if request.segment_type and request.segment_value:
            segment = (request.segment_type, request.segment_value)
            segment_name = f"{request.segment_type}:{request.segment_value}"
            
            # Create filtered time series
            pipeline.create_time_series(
                intervention_date=request.intervention_date,
                segment_col=request.segment_type,
                segment_val=request.segment_value
            )
        else:
            pipeline.create_time_series(intervention_date=request.intervention_date)
        
        # Run analysis
        analysis_data = pipeline.get_analysis_series(metric='revenue_usd')
        analyzer = CausalAnalyzer(analysis_data, config=config, segment=segment)
        analyzer.run_causal_impact()
        
        metrics = analyzer.get_impact_metrics()
        
        # Financial analysis
        fin_analyzer = FinancialAnalyzer(metrics, campaign_cost=request.campaign_cost)
        fin_analyzer.calculate_roi()
        fin_results = fin_analyzer.financial_results
        
        # Build response
        impact = ImpactMetrics(
            segment=segment_name,
            cumulative_effect=metrics['cumulative_effect'],
            cumulative_effect_pct=metrics['cumulative_effect_pct'],
            average_effect=metrics['average_effect'],
            p_value=metrics['p_value'],
            significant=metrics['p_value'] < 0.05,
            ci_lower=metrics.get('ci_lower'),
            ci_upper=metrics.get('ci_upper')
        )
        
        financial = FinancialMetrics(
            roi_percentage=fin_results['roi_percentage'],
            roi_ratio=fin_results['roi_ratio'],
            net_profit=fin_results['net_profit'],
            cost_per_incremental=fin_results.get('cost_per_incremental_dollar', 0),
            payback_period_days=fin_results.get('payback_period_days')
        )
        
        # Generate summary
        summary = f"Analysis of {segment_name}: "
        if metrics['p_value'] < 0.05:
            direction = "positive" if metrics['cumulative_effect'] > 0 else "negative"
            summary += f"Found statistically significant {direction} effect of ${metrics['cumulative_effect']:,.2f} "
            summary += f"(p={metrics['p_value']:.4f}). ROI: {fin_results['roi_percentage']:.1f}%."
        else:
            summary += f"No statistically significant effect detected (p={metrics['p_value']:.4f})."
        
        return AnalysisResponse(
            status="success",
            timestamp=datetime.now().isoformat(),
            segment=segment_name,
            impact_metrics=impact,
            financial_metrics=financial,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/analyze/segments/{segment_type}", tags=["Analysis"])
async def analyze_all_segments(
    segment_type: str,
    intervention_date: str = Query(..., description="Intervention date (YYYY-MM-DD)"),
    campaign_cost: float = Query(5000.0, description="Campaign cost", ge=0)
):
    """
    Run analysis for all values of a segment type.
    
    Performs causal impact analysis for each unique value in the specified
    segment column and returns comparative results.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    if segment_type not in pipeline.cleaned_data.columns:
        raise HTTPException(status_code=400, detail=f"Unknown segment type: {segment_type}")
    
    config = get_config()
    results = []
    
    for seg_val in pipeline.cleaned_data[segment_type].unique():
        try:
            segment = (segment_type, seg_val)
            
            pipeline.create_time_series(
                intervention_date=intervention_date,
                segment_col=segment_type,
                segment_val=seg_val
            )
            
            analysis_data = pipeline.get_analysis_series(metric='revenue_usd')
            analyzer = CausalAnalyzer(analysis_data, config=config, segment=segment)
            analyzer.run_causal_impact()
            
            metrics = analyzer.get_impact_metrics()
            
            fin_analyzer = FinancialAnalyzer(metrics, campaign_cost=campaign_cost)
            fin_analyzer.calculate_roi()
            
            results.append({
                "segment": str(seg_val),
                "cumulative_effect": metrics['cumulative_effect'],
                "p_value": metrics['p_value'],
                "significant": metrics['p_value'] < 0.05,
                "roi_percentage": fin_analyzer.financial_results['roi_percentage']
            })
            
        except Exception as e:
            results.append({
                "segment": str(seg_val),
                "error": str(e)
            })
    
    return {
        "segment_type": segment_type,
        "intervention_date": intervention_date,
        "campaign_cost": campaign_cost,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }


@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """
    Get current configuration.
    
    Returns the loaded configuration including dates, segments, and campaign settings.
    """
    config = get_config()
    return {
        "config": config,
        "timestamp": datetime.now().isoformat()
    }


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("=" * 60)
    print("Causal Impact Analysis API Starting...")
    print("=" * 60)
    
    try:
        get_config()
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"⚠ Configuration error: {e}")
    
    try:
        get_pipeline()
        print("✓ Data pipeline initialized")
    except Exception as e:
        print(f"⚠ Pipeline error: {e}")
    
    print("=" * 60)
    print("API Ready! Documentation at /docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _pipeline, _config
    _pipeline = None
    _config = None
    print("API shutdown complete")


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
