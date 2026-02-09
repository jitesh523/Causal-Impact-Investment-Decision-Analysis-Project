"""
Custom Report Builder Module
============================

Build and generate customizable analysis reports.
Supports multiple output formats: PDF, HTML, Markdown, JSON.

Author: Causal Impact Analysis Project
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
import io


@dataclass
class ReportSection:
    """A section in a report."""
    title: str
    content: str
    section_type: str  # 'text', 'table', 'chart', 'metrics', 'code'
    data: Optional[Any] = None
    order: int = 0


@dataclass
class ReportConfig:
    """Report configuration."""
    title: str
    subtitle: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    logo_path: Optional[str] = None
    theme: str = 'default'
    include_toc: bool = True
    include_summary: bool = True


class ReportBuilder:
    """
    Custom Report Builder for Analysis Results.
    
    Build modular reports with:
    - Text sections
    - Data tables
    - Charts/visualizations
    - Key metrics summaries
    - Code snippets
    
    Example:
        >>> builder = ReportBuilder("Campaign Analysis Q1 2024")
        >>> builder.add_section("Executive Summary", summary_text)
        >>> builder.add_metrics("Key Results", {'ROI': '743%', 'Lift': '35%'})
        >>> builder.add_table("Segment Results", results_df)
        >>> builder.export_html("report.html")
    """
    
    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        author: Optional[str] = None
    ):
        """
        Initialize report builder.
        
        Args:
            title: Report title
            subtitle: Optional subtitle
            author: Report author
        """
        self.config = ReportConfig(
            title=title,
            subtitle=subtitle,
            author=author,
            date=datetime.now().strftime("%Y-%m-%d")
        )
        
        self._sections: List[ReportSection] = []
        self._order_counter = 0
    
    def add_section(
        self,
        title: str,
        content: str,
        section_type: str = 'text'
    ) -> 'ReportBuilder':
        """
        Add a text section.
        
        Args:
            title: Section title
            content: Section content (text/markdown)
            section_type: Type of section
        
        Returns:
            self for chaining
        """
        self._order_counter += 1
        self._sections.append(ReportSection(
            title=title,
            content=content,
            section_type=section_type,
            order=self._order_counter
        ))
        return self
    
    def add_metrics(
        self,
        title: str,
        metrics: Dict[str, Any],
        description: Optional[str] = None
    ) -> 'ReportBuilder':
        """
        Add a metrics section with key-value pairs.
        
        Args:
            title: Section title
            metrics: Dictionary of metric names and values
            description: Optional description
        
        Returns:
            self for chaining
        """
        self._order_counter += 1
        self._sections.append(ReportSection(
            title=title,
            content=description or "",
            section_type='metrics',
            data=metrics,
            order=self._order_counter
        ))
        return self
    
    def add_table(
        self,
        title: str,
        data: Any,
        description: Optional[str] = None
    ) -> 'ReportBuilder':
        """
        Add a data table section.
        
        Args:
            title: Section title
            data: DataFrame or list of dicts
            description: Optional description
        
        Returns:
            self for chaining
        """
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            table_data = data.to_dict('records')
            columns = data.columns.tolist()
        elif isinstance(data, list):
            table_data = data
            columns = list(data[0].keys()) if data else []
        else:
            raise ValueError("Data must be DataFrame or list of dicts")
        
        self._order_counter += 1
        self._sections.append(ReportSection(
            title=title,
            content=description or "",
            section_type='table',
            data={'records': table_data, 'columns': columns},
            order=self._order_counter
        ))
        return self
    
    def add_chart(
        self,
        title: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        description: Optional[str] = None
    ) -> 'ReportBuilder':
        """
        Add a chart/image section.
        
        Args:
            title: Section title
            image_path: Path to image file
            image_base64: Base64 encoded image
            description: Optional description
        
        Returns:
            self for chaining
        """
        if image_path:
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode()
        
        self._order_counter += 1
        self._sections.append(ReportSection(
            title=title,
            content=description or "",
            section_type='chart',
            data={'image': image_base64},
            order=self._order_counter
        ))
        return self
    
    def add_code(
        self,
        title: str,
        code: str,
        language: str = 'python',
        description: Optional[str] = None
    ) -> 'ReportBuilder':
        """
        Add a code snippet section.
        
        Args:
            title: Section title
            code: Code content
            language: Programming language
            description: Optional description
        
        Returns:
            self for chaining
        """
        self._order_counter += 1
        self._sections.append(ReportSection(
            title=title,
            content=description or "",
            section_type='code',
            data={'code': code, 'language': language},
            order=self._order_counter
        ))
        return self
    
    def export_html(self, output_path: str) -> str:
        """
        Export report as HTML.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to generated file
        """
        html = self._generate_html()
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def export_markdown(self, output_path: str) -> str:
        """
        Export report as Markdown.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to generated file
        """
        md = self._generate_markdown()
        
        with open(output_path, 'w') as f:
            f.write(md)
        
        return output_path
    
    def export_json(self, output_path: str) -> str:
        """
        Export report as JSON.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to generated file
        """
        data = {
            'config': asdict(self.config),
            'sections': [asdict(s) for s in self._sections]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        css = """
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }
            .subtitle { color: #7f8c8d; font-size: 1.2em; }
            .meta { color: #95a5a6; font-size: 0.9em; margin-bottom: 30px; }
            .metrics { display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }
            .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; min-width: 150px; text-align: center; }
            .metric-value { font-size: 1.8em; font-weight: bold; }
            .metric-label { font-size: 0.9em; opacity: 0.9; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th { background: #3498db; color: white; padding: 12px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #eee; }
            tr:hover { background: #f9f9f9; }
            .code-block { background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: 'Fira Code', monospace; }
            .toc { background: #ecf0f1; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
            .toc ul { list-style: none; padding-left: 0; }
            .toc li { margin: 8px 0; }
            .toc a { color: #3498db; text-decoration: none; }
            img { max-width: 100%; border-radius: 5px; }
        </style>
        """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.config.title}</title>
    {css}
</head>
<body>
<div class="container">
    <h1>{self.config.title}</h1>
    {'<p class="subtitle">' + self.config.subtitle + '</p>' if self.config.subtitle else ''}
    <p class="meta">
        {'Author: ' + self.config.author + ' | ' if self.config.author else ''}
        Date: {self.config.date}
    </p>
"""
        
        # Table of contents
        if self.config.include_toc and len(self._sections) > 2:
            html += '<div class="toc"><h3>Contents</h3><ul>'
            for section in sorted(self._sections, key=lambda x: x.order):
                anchor = section.title.lower().replace(' ', '-')
                html += f'<li><a href="#{anchor}">{section.title}</a></li>'
            html += '</ul></div>'
        
        # Sections
        for section in sorted(self._sections, key=lambda x: x.order):
            anchor = section.title.lower().replace(' ', '-')
            html += f'<h2 id="{anchor}">{section.title}</h2>'
            
            if section.content:
                html += f'<p>{section.content}</p>'
            
            if section.section_type == 'metrics' and section.data:
                html += '<div class="metrics">'
                for label, value in section.data.items():
                    html += f'''
                    <div class="metric-card">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    '''
                html += '</div>'
            
            elif section.section_type == 'table' and section.data:
                html += '<table>'
                html += '<tr>' + ''.join(f'<th>{col}</th>' for col in section.data['columns']) + '</tr>'
                for row in section.data['records']:
                    html += '<tr>' + ''.join(f'<td>{row.get(col, "")}</td>' for col in section.data['columns']) + '</tr>'
                html += '</table>'
            
            elif section.section_type == 'chart' and section.data:
                if section.data.get('image'):
                    html += f'<img src="data:image/png;base64,{section.data["image"]}" alt="{section.title}">'
            
            elif section.section_type == 'code' and section.data:
                html += f'<pre class="code-block"><code>{section.data["code"]}</code></pre>'
        
        html += '</div></body></html>'
        return html
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        md = f"# {self.config.title}\n\n"
        
        if self.config.subtitle:
            md += f"*{self.config.subtitle}*\n\n"
        
        md += f"**Date:** {self.config.date}"
        if self.config.author:
            md += f" | **Author:** {self.config.author}"
        md += "\n\n---\n\n"
        
        # Table of contents
        if self.config.include_toc and len(self._sections) > 2:
            md += "## Contents\n\n"
            for section in sorted(self._sections, key=lambda x: x.order):
                anchor = section.title.lower().replace(' ', '-')
                md += f"- [{section.title}](#{anchor})\n"
            md += "\n---\n\n"
        
        # Sections
        for section in sorted(self._sections, key=lambda x: x.order):
            md += f"## {section.title}\n\n"
            
            if section.content:
                md += f"{section.content}\n\n"
            
            if section.section_type == 'metrics' and section.data:
                md += "| Metric | Value |\n|--------|-------|\n"
                for label, value in section.data.items():
                    md += f"| {label} | {value} |\n"
                md += "\n"
            
            elif section.section_type == 'table' and section.data:
                cols = section.data['columns']
                md += "| " + " | ".join(cols) + " |\n"
                md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
                for row in section.data['records']:
                    md += "| " + " | ".join(str(row.get(col, "")) for col in cols) + " |\n"
                md += "\n"
            
            elif section.section_type == 'code' and section.data:
                lang = section.data.get('language', '')
                md += f"```{lang}\n{section.data['code']}\n```\n\n"
        
        return md


def main():
    """Demo report builder."""
    print("=" * 60)
    print("CUSTOM REPORT BUILDER DEMO")
    print("=" * 60)
    
    import pandas as pd
    
    # Create report
    builder = ReportBuilder(
        title="Campaign Impact Analysis",
        subtitle="Q1 2024 Marketing Attribution Report",
        author="Analytics Team"
    )
    
    # Add executive summary
    builder.add_section(
        "Executive Summary",
        "This report analyzes the causal impact of our Q1 marketing campaign. "
        "Using Bayesian Structural Time Series modeling, we found a statistically "
        "significant positive effect on revenue."
    )
    
    # Add key metrics
    builder.add_metrics(
        "Key Results",
        {
            'Total Lift': '$42,137',
            'ROI': '743%',
            'Relative Effect': '+35%',
            'Significance': 'p < 0.001'
        },
        "Summary of main findings from the causal analysis."
    )
    
    # Add segment results table
    segment_data = pd.DataFrame({
        'Segment': ['Email', 'Social', 'Search', 'Display'],
        'Effect': ['$15,420', '$12,350', '$9,540', '$4,827'],
        'ROI': ['892%', '645%', '412%', '287%'],
        'Significant': ['Yes', 'Yes', 'Yes', 'No']
    })
    
    builder.add_table(
        "Segment Performance",
        segment_data,
        "Campaign performance broken down by marketing channel."
    )
    
    # Add methodology
    builder.add_section(
        "Methodology",
        "We applied **Bayesian Structural Time Series (BSTS)** modeling to estimate "
        "the causal impact. The pre-intervention period (Jan 1 - Feb 28) was used to "
        "build a counterfactual forecast, which was then compared against actual "
        "post-intervention outcomes (Mar 1 - Mar 31)."
    )
    
    # Add code example
    builder.add_code(
        "Analysis Code",
        '''from src.causal_analysis import CausalImpactAnalysis

analyzer = CausalImpactAnalysis()
results = analyzer.run(
    data=df,
    intervention_date='2024-03-01'
)
print(f"Effect: ${results.cumulative_effect:,.2f}")''',
        "python",
        "Code used to run the analysis."
    )
    
    # Export
    print("\nGenerating reports...")
    
    builder.export_html("demo_report.html")
    print("✓ Exported: demo_report.html")
    
    builder.export_markdown("demo_report.md")
    print("✓ Exported: demo_report.md")
    
    builder.export_json("demo_report.json")
    print("✓ Exported: demo_report.json")
    
    # Cleanup
    import os
    for f in ['demo_report.html', 'demo_report.md', 'demo_report.json']:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n✓ Report builder demo completed!")


if __name__ == '__main__':
    main()
