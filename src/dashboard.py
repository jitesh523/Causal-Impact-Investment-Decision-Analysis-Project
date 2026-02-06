"""
Causal Impact Dashboard - Enhanced Version
Interactive Streamlit application with:
- Segment Comparison View
- Export functionality  
- Dynamic Date Picker
- Dark Mode support
- Segment Heatmap Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys
import io

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from data_pipeline import DataPipeline
from causal_analysis import CausalAnalyzer
from financial_analysis import FinancialAnalyzer

# Page config with theme support
st.set_page_config(
    page_title="Causal Impact Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme handling
def apply_theme(dark_mode: bool):
    """Apply dark or light theme"""
    if dark_mode:
        st.markdown("""
        <style>
            .stApp { background-color: #1a1a2e; color: #eee; }
            .stMetric { background-color: #16213e; border-radius: 10px; padding: 15px; }
            .stTabs [data-baseweb="tab-list"] { background-color: #16213e; }
            .css-1d391kg { background-color: #0f3460; }
        </style>
        """, unsafe_allow_html=True)

# Load config
@st.cache_data
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Caching the data load
@st.cache_data
def get_data():
    pipeline = DataPipeline('config.yaml')
    pipeline.load_data().clean_data()
    return pipeline

def run_analysis(pipeline, intervention_date, segment=None, cost=5000):
    """Run causal analysis with given parameters"""
    if segment:
        pipeline.create_time_series(
            intervention_date=intervention_date,
            segment_col=segment[0],
            segment_val=segment[1]
        )
    else:
        pipeline.create_time_series(intervention_date=intervention_date)
    
    analysis_data = pipeline.get_analysis_series(metric='revenue_usd')
    analyzer = CausalAnalyzer(analysis_data, config=config, segment=segment)
    analyzer.run_causal_impact()
    metrics = analyzer.get_impact_metrics()
    
    fin_analyzer = FinancialAnalyzer(metrics, campaign_cost=cost)
    fin_analyzer.calculate_roi()
    
    return analyzer, metrics, fin_analyzer.financial_results

def create_comparison_chart(results_dict):
    """Create comparison bar chart for multiple segments"""
    segments = list(results_dict.keys())
    effects = [r['metrics']['cumulative_effect'] for r in results_dict.values()]
    rois = [r['financial']['roi_percentage'] for r in results_dict.values()]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Cumulative Effect ($)', 'ROI (%)'))
    
    # Effect bars
    colors = ['#06A77D' if e > 0 else '#E63946' for e in effects]
    fig.add_trace(
        go.Bar(x=segments, y=effects, marker_color=colors, name='Effect'),
        row=1, col=1
    )
    
    # ROI bars
    roi_colors = ['#2E86AB' if r > 0 else '#E63946' for r in rois]
    fig.add_trace(
        go.Bar(x=segments, y=rois, marker_color=roi_colors, name='ROI'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Segment Comparison"
    )
    
    return fig


def create_impact_heatmap(pipeline, results_by_segment, segment_col1, segment_col2=None, dark_mode=False):
    """
    Create heatmap visualization of impact across segments.
    
    Args:
        pipeline: DataPipeline instance
        results_by_segment: Dict of segment -> results
        segment_col1: Primary segment column
        segment_col2: Optional secondary segment for 2D heatmap
        dark_mode: Whether to use dark theme
    
    Returns:
        Plotly figure
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    
    if segment_col2 is None:
        # Single dimension heatmap (horizontal bar-style)
        segments = list(results_by_segment.keys())
        effects = [results_by_segment[s]['metrics']['cumulative_effect'] for s in segments]
        rois = [results_by_segment[s]['financial']['roi_percentage'] for s in segments]
        p_values = [results_by_segment[s]['metrics']['p_value'] for s in segments]
        
        # Create grid for heatmap
        metrics = ['Impact ($)', 'ROI (%)', 'Significance']
        z_data = [
            effects,
            rois,
            [1 if p < 0.05 else 0 for p in p_values]
        ]
        
        # Normalize for color scale
        z_normalized = []
        for row in z_data:
            max_val = max(abs(v) for v in row) if row else 1
            z_normalized.append([v / max_val if max_val > 0 else 0 for v in row])
        
        # Create hover text
        hover_text = []
        for i, metric in enumerate(metrics):
            row_text = []
            for j, seg in enumerate(segments):
                if metric == 'Impact ($)':
                    row_text.append(f"{seg}<br>{metric}: ${effects[j]:,.0f}")
                elif metric == 'ROI (%)':
                    row_text.append(f"{seg}<br>{metric}: {rois[j]:.1f}%")
                else:
                    row_text.append(f"{seg}<br>p-value: {p_values[j]:.4f}")
            hover_text.append(row_text)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_normalized,
            x=segments,
            y=metrics,
            colorscale='RdYlGn',
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showscale=True,
            colorbar=dict(title='Relative<br>Performance')
        ))
        
        # Add annotations for values
        for i, metric in enumerate(metrics):
            for j, seg in enumerate(segments):
                if metric == 'Impact ($)':
                    text = f"${effects[j]:,.0f}"
                elif metric == 'ROI (%)':
                    text = f"{rois[j]:.0f}%"
                else:
                    text = "✓" if p_values[j] < 0.05 else "✗"
                
                fig.add_annotation(
                    x=seg, y=metric,
                    text=text,
                    showarrow=False,
                    font=dict(color='white' if abs(z_normalized[i][j]) > 0.5 else 'black', size=12)
                )
        
        fig.update_layout(
            title="Segment Performance Heatmap",
            xaxis_title=segment_col1,
            yaxis_title="Metric",
            template=template,
            height=300
        )
        
    else:
        # Two-dimensional heatmap (matrix)
        seg1_vals = sorted(set(s[0] for s in results_by_segment.keys() if isinstance(s, tuple)))
        seg2_vals = sorted(set(s[1] for s in results_by_segment.keys() if isinstance(s, tuple)))
        
        z_matrix = []
        hover_matrix = []
        
        for s1 in seg1_vals:
            row = []
            hover_row = []
            for s2 in seg2_vals:
                key = (s1, s2)
                if key in results_by_segment:
                    effect = results_by_segment[key]['metrics']['cumulative_effect']
                    p_val = results_by_segment[key]['metrics']['p_value']
                    row.append(effect)
                    hover_row.append(f"{s1} x {s2}<br>Impact: ${effect:,.0f}<br>p-value: {p_val:.4f}")
                else:
                    row.append(None)
                    hover_row.append("No data")
            z_matrix.append(row)
            hover_matrix.append(hover_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=seg2_vals,
            y=seg1_vals,
            colorscale='RdYlGn',
            text=hover_matrix,
            hovertemplate='%{text}<extra></extra>',
            colorbar=dict(title='Impact ($)')
        ))
        
        fig.update_layout(
            title=f"Impact Heatmap: {segment_col1} × {segment_col2}",
            xaxis_title=segment_col2,
            yaxis_title=segment_col1,
            template=template,
            height=400
        )
    
    return fig

def main():
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Theme toggle
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
    apply_theme(dark_mode)
    
    # Campaign Cost
    cost = st.sidebar.number_input(
        "💰 Campaign Cost ($)", 
        value=float(config['campaign']['cost']),
        step=500.0,
        format="%.2f"
    )
    
    # Date Picker for What-If Analysis
    st.sidebar.subheader("📅 Intervention Date")
    default_date = pd.to_datetime(config['dates']['intervention_date'])
    intervention_date = st.sidebar.date_input(
        "Select intervention date",
        value=default_date,
        min_value=pd.to_datetime("2024-01-15"),
        max_value=pd.to_datetime("2024-04-30")
    )
    intervention_date_str = intervention_date.strftime('%Y-%m-%d')
    
    # Analysis Mode
    st.sidebar.subheader("📊 Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Choose mode",
        ["Single Segment", "Compare Segments"],
        index=0
    )
    
    pipeline = get_data()
    
    # Title
    st.title("📈 Causal Impact & Investment Analysis")
    st.markdown("*Analyze marketing campaign effectiveness with interactive controls*")
    
    if analysis_mode == "Single Segment":
        # Single segment analysis
        segment_type = st.sidebar.selectbox(
            "Analyze by",
            ["Aggregated"] + config.get('segments', [])
        )
        
        selected_segment = None
        if segment_type != "Aggregated":
            unique_vals = pipeline.cleaned_data[segment_type].unique()
            segment_val = st.sidebar.selectbox(f"Select {segment_type}", unique_vals)
            selected_segment = (segment_type, segment_val)
        
        try:
            with st.spinner("Running Causal Impact Analysis..."):
                analyzer, metrics, fin_results = run_analysis(
                    pipeline, intervention_date_str, selected_segment, cost
                )
            
            # Top Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Revenue Lift", 
                    f"${metrics['cumulative_effect']:,.2f}",
                    delta=f"{metrics['cumulative_effect_pct']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "ROI Ratio", 
                    f"{fin_results['roi_ratio']:.2f}x",
                    delta=f"{fin_results['roi_percentage']:.1f}% ROI"
                )
                
            with col3:
                st.metric(
                    "Net Profit", 
                    f"${fin_results['net_profit']:,.2f}"
                )
                
            with col4:
                is_sig = metrics['p_value'] < 0.05
                st.metric(
                    "Statistical Significance", 
                    f"p = {metrics['p_value']:.4f}",
                    delta="Significant" if is_sig else "Not Significant",
                    delta_color="normal" if is_sig else "inverse"
                )
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Impact Charts", "💰 Financial Report", "📋 Raw Data", "📥 Export"])
            
            with tab1:
                # Interactive Plotly Chart
                dates = analyzer.dates
                actual = analyzer.predictions['actual']
                predicted = analyzer.predictions['predicted']
                lower = analyzer.predictions['predicted_lower']
                upper = analyzer.predictions['predicted_upper']
                
                fig = go.Figure()
                
                # Confidence Interval
                fig.add_trace(go.Scatter(
                    x=np.concatenate([dates, dates[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(162, 59, 114, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    name='95% Confidence Interval'
                ))
                
                # Actual
                fig.add_trace(go.Scatter(
                    x=dates, y=actual,
                    mode='lines+markers',
                    name='Observed Data',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                # Predicted
                fig.add_trace(go.Scatter(
                    x=dates, y=predicted,
                    mode='lines',
                    name='Counterfactual',
                    line=dict(color='#A23B72', width=2, dash='dash')
                ))
                
                # Intervention Line
                int_date = dates[analyzer.post_period[0]]
                fig.add_vline(x=int_date, line_width=2, line_dash="dot", line_color="red")
                
                template = "plotly_dark" if dark_mode else "plotly_white"
                fig.update_layout(
                    title="Observed vs. Counterfactual Revenue",
                    xaxis_title="Date",
                    yaxis_title="Revenue ($)",
                    template=template,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cumulative Effect Chart
                cum_effect = np.cumsum(analyzer.predictions['point_effect'])
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=dates, y=cum_effect,
                    mode='lines',
                    name='Cumulative Effect',
                    fill='tozeroy',
                    line=dict(color='#06A77D', width=2)
                ))
                fig2.add_vline(x=int_date, line_width=2, line_dash="dot", line_color="red")
                
                fig2.update_layout(
                    title="Cumulative Causal Effect (Net Revenue Gain)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Revenue ($)",
                    template=template,
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

            with tab2:
                st.markdown("### Business Narrative")
                fin_analyzer = FinancialAnalyzer(metrics, campaign_cost=cost)
                fin_analyzer.calculate_roi()
                narrative = fin_analyzer.generate_business_narrative()
                st.info(narrative)
                
                st.markdown("### Detailed Metrics")
                res_df = pd.DataFrame([fin_results]).T
                res_df.columns = ["Value"]
                st.dataframe(res_df, use_container_width=True)
                
            with tab3:
                st.dataframe(pipeline.time_series_data, use_container_width=True)
            
            with tab4:
                st.markdown("### 📥 Export Data & Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export metrics as CSV
                    metrics_df = pd.DataFrame([metrics])
                    csv_buffer = io.StringIO()
                    metrics_df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📊 Download Impact Metrics (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="impact_metrics.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export time series data
                    ts_buffer = io.StringIO()
                    pipeline.time_series_data.to_csv(ts_buffer, index=False)
                    
                    st.download_button(
                        label="📋 Download Time Series Data (CSV)",
                        data=ts_buffer.getvalue(),
                        file_name="time_series_data.csv",
                        mime="text/csv"
                    )
                
                # Export financial results
                fin_df = pd.DataFrame([fin_results])
                fin_buffer = io.StringIO()
                fin_df.to_csv(fin_buffer, index=False)
                
                st.download_button(
                    label="💰 Download Financial Results (CSV)",
                    data=fin_buffer.getvalue(),
                    file_name="financial_results.csv",
                    mime="text/csv"
                )
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.code(str(e))
    
    else:
        # Comparison Mode
        st.sidebar.subheader("Compare Segments")
        segment_col = st.sidebar.selectbox(
            "Segment Type",
            config.get('segments', ['channel'])
        )
        
        unique_vals = list(pipeline.cleaned_data[segment_col].unique())
        selected_segments = st.sidebar.multiselect(
            f"Select {segment_col}s to compare",
            unique_vals,
            default=unique_vals[:3] if len(unique_vals) >= 3 else unique_vals
        )
        
        if len(selected_segments) < 2:
            st.warning("Please select at least 2 segments to compare.")
        else:
            results = {}
            
            with st.spinner("Running comparison analysis..."):
                for seg_val in selected_segments:
                    try:
                        analyzer, metrics, fin_results = run_analysis(
                            pipeline, intervention_date_str, (segment_col, seg_val), cost
                        )
                        results[seg_val] = {
                            'metrics': metrics,
                            'financial': fin_results
                        }
                    except Exception as e:
                        st.warning(f"Could not analyze {seg_val}: {str(e)}")
            
            if results:
                # Comparison chart
                st.subheader("📊 Segment Comparison")
                comp_fig = create_comparison_chart(results)
                st.plotly_chart(comp_fig, use_container_width=True)
                
                # Heatmap visualization
                st.subheader("🗺️ Performance Heatmap")
                heatmap_fig = create_impact_heatmap(pipeline, results, segment_col, dark_mode=dark_mode)
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Comparison table
                st.subheader("📋 Detailed Comparison")
                comp_data = []
                for seg, data in results.items():
                    comp_data.append({
                        'Segment': seg,
                        'Revenue Impact ($)': f"${data['metrics']['cumulative_effect']:,.2f}",
                        'ROI (%)': f"{data['financial']['roi_percentage']:.1f}%",
                        'Net Profit ($)': f"${data['financial']['net_profit']:,.2f}",
                        'P-Value': f"{data['metrics']['p_value']:.4f}",
                        'Significant': '✅' if data['metrics']['p_value'] < 0.05 else '⚠️'
                    })
                
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Export comparison
                csv_buffer = io.StringIO()
                comp_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Comparison (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="segment_comparison.csv",
                    mime="text/csv"
                )

if __name__ == '__main__':
    main()
