
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml
from pathlib import Path
import sys

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from data_pipeline import DataPipeline
from causal_analysis import CausalAnalyzer
from financial_analysis import FinancialAnalyzer

st.set_page_config(
    page_title="Causal Impact Dashboard",
    page_icon="📈",
    layout="wide"
)

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

def main():
    st.title("📈 Causal Impact & Investment Analysis")
    st.markdown("Analyze marketing campaign effectiveness and financial ROI.")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Campaign Cost
    cost = st.sidebar.number_input(
        "Campaign Cost ($)", 
        value=config['campaign']['cost'],
        step=500.0,
        format="%.2f"
    )
    
    # Segmentation
    st.sidebar.subheader("Segmentation")
    segment_type = st.sidebar.selectbox(
        "Analyze by",
        ["Aggregated"] + config.get('segments', [])
    )
    
    pipeline = get_data()
    
    selected_segment = None
    if segment_type != "Aggregated":
        unique_vals = pipeline.cleaned_data[segment_type].unique()
        segment_val = st.sidebar.selectbox(f"Select {segment_type}", unique_vals)
        selected_segment = (segment_type, segment_val)
    
    # Run Analysis
    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("Run Analysis", type="primary")
    
    if run_btn or True: # Auto-run on load
        try:
            with st.spinner("Running Causal Impact Analysis..."):
                # Create time series
                if selected_segment:
                    pipeline.create_time_series(
                        segment_col=selected_segment[0],
                        segment_val=selected_segment[1]
                    )
                else:
                    pipeline.create_time_series()
                
                analysis_data = pipeline.get_analysis_series(metric='revenue_usd')
                
                # Causal Analysis
                analyzer = CausalAnalyzer(analysis_data, config=config, segment=selected_segment)
                analyzer.run_causal_impact(alpha=0.05) # Fixed alpha for UI simplicity or add to sidebar
                metrics = analyzer.get_impact_metrics()
                
                # Financial Analysis
                fin_analyzer = FinancialAnalyzer(metrics, campaign_cost=cost)
                fin_analyzer.calculate_roi()
                fin_results = fin_analyzer.financial_results
                
                # --- Dashboard Layout ---
                
                # Top Metrics
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
                tab1, tab2, tab3 = st.tabs(["📊 Impact Charts", "💰 Financial Report", "📋 Raw Data"])
                
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
                    intervention_date = dates[analyzer.post_period[0]]
                    fig.add_vline(x=intervention_date, line_width=2, line_dash="dot", line_color="red")
                    
                    fig.update_layout(
                        title="Observed vs. Counterfactual Revenue",
                        xaxis_title="Date",
                        yaxis_title="Revenue ($)",
                        template="plotly_white",
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
                    fig2.add_vline(x=intervention_date, line_width=2, line_dash="dot", line_color="red")
                    
                    fig2.update_layout(
                        title="Cumulative Causal Effect (Net Revenue Gain)",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Revenue ($)",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                with tab2:
                    st.markdown("### Business Narrative")
                    narrative = fin_analyzer.generate_business_narrative()
                    # Clean up the box drawing characters for web display or keep them
                    st.info(narrative)
                    
                    st.markdown("### Detailed Metrics")
                    res_df = pd.DataFrame([fin_results]).T
                    res_df.columns = ["Value"]
                    st.dataframe(res_df, use_container_width=True)
                    
                with tab3:
                    st.dataframe(pipeline.time_series_data)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.code(str(e))

if __name__ == '__main__':
    main()
