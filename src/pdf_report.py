"""
PDF Report Generation Module
============================

Generates executive-ready PDF reports with causal impact analysis results,
charts, tables, and key findings. Uses ReportLab for PDF generation.

Author: Causal Impact Analysis Project
"""

import io
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class PDFReportGenerator:
    """
    Generate professional PDF reports for causal impact analysis.
    
    Creates executive-ready reports with:
    - Executive summary
    - Key metrics and KPIs
    - Visualizations
    - Segment-level breakdowns
    - Statistical details
    
    Example:
        >>> generator = PDFReportGenerator(
        ...     analysis_results=results,
        ...     output_path='report.pdf'
        ... )
        >>> generator.generate()
    """
    
    def __init__(
        self,
        analysis_results: Dict[str, Any],
        output_path: str = 'causal_impact_report.pdf',
        title: str = 'Causal Impact Analysis Report',
        author: str = 'Analytics Team',
        company: str = '',
        page_size: str = 'letter'
    ):
        """
        Initialize PDF report generator.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Path for output PDF file
            title: Report title
            author: Report author
            company: Company name (optional)
            page_size: Page size ('letter' or 'a4')
        """
        self.results = analysis_results
        self.output_path = output_path
        self.title = title
        self.author = author
        self.company = company
        self.page_size = letter if page_size == 'letter' else A4
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        self.elements = []
        self.temp_images = []
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=28,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2c5282'),
            spaceBefore=20,
            spaceAfter=12,
            borderColor=colors.HexColor('#2c5282'),
            borderWidth=0,
            borderPadding=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#4a5568'),
            spaceBefore=15,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#2d3748'),
            spaceBefore=6,
            spaceAfter=6,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=24,
            textColor=colors.HexColor('#2c5282'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#718096'),
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#a0aec0'),
            alignment=TA_CENTER
        ))
    
    def _add_title_page(self):
        """Add title page to report."""
        # Title
        self.elements.append(Spacer(1, 2 * inch))
        self.elements.append(Paragraph(self.title, self.styles['CustomTitle']))
        
        # Subtitle with date
        date_str = datetime.now().strftime('%B %d, %Y')
        self.elements.append(Paragraph(
            f"Generated: {date_str}",
            self.styles['CustomBody']
        ))
        
        if self.company:
            self.elements.append(Spacer(1, 0.5 * inch))
            self.elements.append(Paragraph(
                self.company,
                self.styles['SubSection']
            ))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        self.elements.append(Paragraph(
            f"Prepared by: {self.author}",
            self.styles['CustomBody']
        ))
        
        # Decorative line
        self.elements.append(Spacer(1, inch))
        self.elements.append(HRFlowable(
            width="80%",
            thickness=2,
            color=colors.HexColor('#2c5282'),
            spaceBefore=0,
            spaceAfter=0
        ))
        
        self.elements.append(PageBreak())
    
    def _add_executive_summary(self):
        """Add executive summary section."""
        self.elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key findings based on results
        summary_text = self._generate_summary_text()
        self.elements.append(Paragraph(summary_text, self.styles['CustomBody']))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # Key metrics cards
        self._add_metrics_cards()
        
        self.elements.append(Spacer(1, 0.3 * inch))
    
    def _generate_summary_text(self) -> str:
        """Generate summary text from results."""
        if not self.results:
            return "No analysis results available."
        
        summary_parts = []
        
        # Overall finding
        if 'cumulative_effect' in self.results:
            effect = self.results['cumulative_effect']
            direction = "positive" if effect > 0 else "negative"
            summary_parts.append(
                f"The causal impact analysis reveals a <b>{direction}</b> effect "
                f"with a cumulative impact of <b>{effect:,.2f}</b>."
            )
        
        # Statistical significance
        if 'p_value' in self.results:
            p_val = self.results['p_value']
            if p_val < 0.01:
                sig_text = "highly statistically significant (p < 0.01)"
            elif p_val < 0.05:
                sig_text = "statistically significant (p < 0.05)"
            else:
                sig_text = "not statistically significant at conventional levels"
            summary_parts.append(f"This effect is {sig_text}.")
        
        # ROI
        if 'roi_percentage' in self.results:
            roi = self.results['roi_percentage']
            summary_parts.append(f"The estimated return on investment is <b>{roi:.1f}%</b>.")
        
        # Confidence interval
        if 'ci_lower' in self.results and 'ci_upper' in self.results:
            summary_parts.append(
                f"The 95% confidence interval ranges from {self.results['ci_lower']:,.2f} "
                f"to {self.results['ci_upper']:,.2f}."
            )
        
        return " ".join(summary_parts) if summary_parts else "Analysis completed successfully."
    
    def _add_metrics_cards(self):
        """Add key metrics as visual cards."""
        metrics = []
        
        if 'cumulative_effect' in self.results:
            metrics.append(('Cumulative Effect', f"{self.results['cumulative_effect']:,.0f}"))
        if 'average_effect' in self.results:
            metrics.append(('Avg. Daily Effect', f"{self.results['average_effect']:,.2f}"))
        if 'roi_percentage' in self.results:
            metrics.append(('ROI', f"{self.results['roi_percentage']:.1f}%"))
        if 'p_value' in self.results:
            metrics.append(('P-Value', f"{self.results['p_value']:.4f}"))
        
        if not metrics:
            return
        
        # Create table for metrics
        metric_data = []
        labels = []
        values = []
        
        for label, value in metrics[:4]:  # Max 4 metrics
            labels.append(Paragraph(label, self.styles['MetricLabel']))
            values.append(Paragraph(str(value), self.styles['MetricValue']))
        
        metric_table = Table(
            [values, labels],
            colWidths=[1.5 * inch] * len(metrics)
        )
        
        metric_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        self.elements.append(metric_table)
    
    def _add_methodology_section(self):
        """Add methodology description."""
        self.elements.append(Paragraph("Methodology", self.styles['SectionHeader']))
        
        methodology_text = """
        This analysis employs Bayesian structural time-series modeling to estimate 
        the causal impact of the intervention. The methodology constructs a 
        counterfactual prediction of what would have happened in the absence of 
        the intervention, then compares this prediction to the observed outcomes.
        
        <br/><br/>
        Key steps in the analysis:
        <br/>• Data preparation and validation
        <br/>• Pre-intervention model training
        <br/>• Counterfactual prediction generation
        <br/>• Impact estimation with uncertainty quantification
        <br/>• Statistical significance testing
        """
        
        self.elements.append(Paragraph(methodology_text, self.styles['CustomBody']))
        self.elements.append(Spacer(1, 0.3 * inch))
    
    def _add_results_table(self, data: Dict[str, Any], title: str = "Results"):
        """Add a formatted results table."""
        self.elements.append(Paragraph(title, self.styles['SubSection']))
        
        table_data = [['Metric', 'Value']]
        
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                formatted_value = f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
            else:
                formatted_value = str(value)
            table_data.append([formatted_key, formatted_value])
        
        table = Table(table_data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f7fafc'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.3 * inch))
    
    def _add_chart(self, fig, caption: str = "", width: float = 6):
        """Add a matplotlib figure to the report."""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        
        # Calculate aspect ratio
        fig_width, fig_height = fig.get_size_inches()
        aspect = fig_height / fig_width
        
        img = Image(img_buffer, width=width * inch, height=width * aspect * inch)
        self.elements.append(img)
        
        if caption:
            self.elements.append(Paragraph(
                f"<i>{caption}</i>",
                self.styles['Footer']
            ))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        plt.close(fig)
    
    def _add_segment_analysis(self, segments: List[Dict[str, Any]]):
        """Add segment-level analysis section."""
        self.elements.append(Paragraph("Segment Analysis", self.styles['SectionHeader']))
        
        self.elements.append(Paragraph(
            "The following table shows the impact breakdown by segment:",
            self.styles['CustomBody']
        ))
        
        # Build table
        headers = ['Segment', 'Effect', 'P-Value', 'Significant']
        table_data = [headers]
        
        for seg in segments:
            row = [
                seg.get('segment', 'Unknown'),
                f"{seg.get('cumulative_effect', 0):,.2f}",
                f"{seg.get('p_value', 1):.4f}",
                '✓' if seg.get('p_value', 1) < 0.05 else '✗'
            ]
            table_data.append(row)
        
        table = Table(table_data, colWidths=[2 * inch, 1.5 * inch, 1.2 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f7fafc'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.3 * inch))
    
    def _add_recommendations(self):
        """Add recommendations section."""
        self.elements.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        
        # Generate recommendations based on results
        recommendations = []
        
        if 'p_value' in self.results:
            if self.results['p_value'] < 0.05:
                if self.results.get('cumulative_effect', 0) > 0:
                    recommendations.append(
                        "<b>Continue the intervention:</b> The analysis shows a statistically "
                        "significant positive effect. Consider scaling up the investment."
                    )
                else:
                    recommendations.append(
                        "<b>Reconsider the approach:</b> The analysis shows a statistically "
                        "significant negative effect. Review the intervention strategy."
                    )
            else:
                recommendations.append(
                    "<b>Gather more data:</b> The effect is not statistically significant. "
                    "Consider extending the analysis period or increasing sample size."
                )
        
        if 'roi_percentage' in self.results:
            roi = self.results['roi_percentage']
            if roi > 100:
                recommendations.append(
                    f"<b>Strong ROI ({roi:.0f}%):</b> The intervention is highly profitable. "
                    "Explore opportunities to replicate in other segments."
                )
            elif roi > 0:
                recommendations.append(
                    f"<b>Positive ROI ({roi:.0f}%):</b> The intervention is profitable but "
                    "consider optimization to improve returns."
                )
        
        recommendations.append(
            "<b>Monitor long-term effects:</b> Continue tracking to understand "
            "persistence and potential decay of the intervention effect."
        )
        
        for rec in recommendations:
            self.elements.append(Paragraph(f"• {rec}", self.styles['CustomBody']))
            self.elements.append(Spacer(1, 0.1 * inch))
    
    def _add_footer(self, canvas, doc):
        """Add footer to each page."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#a0aec0'))
        
        page_num = canvas.getPageNumber()
        footer_text = f"Page {page_num} | {self.title} | Generated {datetime.now().strftime('%Y-%m-%d')}"
        
        canvas.drawCentredString(
            self.page_size[0] / 2,
            0.5 * inch,
            footer_text
        )
        canvas.restoreState()
    
    def generate(
        self,
        include_methodology: bool = True,
        include_recommendations: bool = True,
        charts: Optional[List[Tuple[plt.Figure, str]]] = None,
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate the complete PDF report.
        
        Args:
            include_methodology: Include methodology section
            include_recommendations: Include recommendations section
            charts: Optional list of (figure, caption) tuples
            segments: Optional list of segment results
        
        Returns:
            Path to generated PDF
        """
        print(f"\n{'=' * 60}")
        print("GENERATING PDF REPORT")
        print(f"{'=' * 60}")
        
        # Create document
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=self.page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )
        
        # Build content
        self._add_title_page()
        self._add_executive_summary()
        
        if include_methodology:
            self._add_methodology_section()
        
        # Main results
        self.elements.append(Paragraph("Detailed Results", self.styles['SectionHeader']))
        self._add_results_table(self.results, "Impact Metrics")
        
        # Add charts if provided
        if charts:
            self.elements.append(Paragraph("Visualizations", self.styles['SectionHeader']))
            for fig, caption in charts:
                self._add_chart(fig, caption)
        
        # Segment analysis
        if segments:
            self._add_segment_analysis(segments)
        
        # Recommendations
        if include_recommendations:
            self.elements.append(PageBreak())
            self._add_recommendations()
        
        # Build PDF
        print("Building PDF document...")
        doc.build(self.elements, onFirstPage=self._add_footer, onLaterPages=self._add_footer)
        
        print(f"✓ Report generated: {self.output_path}")
        return self.output_path


def create_sample_chart() -> plt.Figure:
    """Create a sample impact chart for demonstration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Simulated data
    observed = 100 + np.cumsum(np.random.randn(100) * 2) + np.linspace(0, 20, 100)
    predicted = 100 + np.cumsum(np.random.randn(100) * 1.5) + np.linspace(0, 5, 100)
    
    intervention_idx = 60
    
    ax.plot(dates, observed, label='Observed', color='#2E86AB', linewidth=2)
    ax.plot(dates, predicted, label='Counterfactual', color='#F18F01', 
            linewidth=2, linestyle='--')
    ax.axvline(dates[intervention_idx], color='red', linestyle=':', 
               linewidth=2, label='Intervention')
    ax.fill_between(dates[intervention_idx:], observed[intervention_idx:], 
                   predicted[intervention_idx:], alpha=0.3, color='green',
                   label='Causal Effect')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Causal Impact Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Demo PDF report generation."""
    print("=" * 60)
    print("PDF REPORT GENERATION DEMO")
    print("=" * 60)
    
    # Sample results
    results = {
        'cumulative_effect': 125000.50,
        'average_effect': 1250.00,
        'cumulative_effect_pct': 15.5,
        'p_value': 0.003,
        'ci_lower': 95000.00,
        'ci_upper': 155000.00,
        'roi_percentage': 285.5,
        'net_profit': 100000.00
    }
    
    # Sample segments
    segments = [
        {'segment': 'Region A', 'cumulative_effect': 45000, 'p_value': 0.01},
        {'segment': 'Region B', 'cumulative_effect': 35000, 'p_value': 0.02},
        {'segment': 'Region C', 'cumulative_effect': 25000, 'p_value': 0.15},
        {'segment': 'Region D', 'cumulative_effect': 20000, 'p_value': 0.03},
    ]
    
    # Create chart
    chart = create_sample_chart()
    
    # Generate report
    generator = PDFReportGenerator(
        analysis_results=results,
        output_path='output/causal_impact_report.pdf',
        title='Causal Impact Analysis Report',
        author='Analytics Team',
        company='Acme Corporation'
    )
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    output_path = generator.generate(
        include_methodology=True,
        include_recommendations=True,
        charts=[(chart, 'Figure 1: Observed vs Counterfactual')],
        segments=segments
    )
    
    print(f"\n✓ PDF report generated successfully!")
    print(f"  Location: {output_path}")


if __name__ == '__main__':
    main()
