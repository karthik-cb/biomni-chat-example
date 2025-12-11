#!/usr/bin/env python3
"""
Gradio Interface for Multi-Agent Research Velocity Demo
Interactive web interface for Memorial Sloan Kettering presentations
"""

import os
import sys
import time
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple
import concurrent.futures
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# Add Biomni to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Biomni'))

from biomni.config import BiomniConfig
from biomni.llm import get_llm

# Import our cbioportal tools
from cbioportal_tools import (
    query_mutations_by_gene,
    analyze_mutation_frequency,
    get_molecular_profiles,
    get_clinical_data_by_study,
    get_available_studies
)

# Import demo classes
from final_demo import ResearchVelocityDemo, Hypothesis

import gradio as gr

class GradioResearchDemo:
    """
    Gradio interface for the multi-agent research velocity demo
    """
    
    def __init__(self):
        """Initialize the Gradio demo"""
        self.demo = ResearchVelocityDemo(dry_run_mode=False)
        self.results_history = []
        
    def run_demo_with_progress(self, time_budget: float, use_dry_run: bool, progress=gr.Progress()) -> Tuple[str, str, go.Figure, str, str]:
        """
        Run the demo with progress tracking for Gradio
        """
        try:
            progress(0.1, desc="Initializing demo...")
            
            if use_dry_run:
                # Use dry-run mode for reliable demo
                progress(0.3, desc="Running dry-run demo...")
                from final_demo import run_dry_run_demo
                results = run_dry_run_demo(self.demo, time_budget)
            else:
                progress(0.3, desc="Running live demo...")
                results = self.demo.run_research_velocity_demo()
            
            progress(0.7, desc="Processing results...")
            
            # Store results for visualization
            self.results_history.append(results)
            
            # Generate formatted output
            summary = self._format_summary(results)
            details = self._format_detailed_results(results)
            chart = self._generate_performance_chart(results)
            
            # Generate downloadable results
            results_json = json.dumps(results, indent=2)
            
            # Create downloadable file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(results_json)
                temp_file = f.name
            
            download_info = "✅ Results ready for download!"
            
            progress(1.0, desc="Demo completed!")
            
            return summary, details, chart, temp_file, download_info
            
        except Exception as e:
            error_msg = f"❌ Demo failed: {str(e)}"
            return error_msg, "", None, "", "Run failed - no results to download."
    
    def _format_summary(self, results: Dict[str, Any]) -> str:
        """Format results summary for display"""
        comparison = results.get('comparison', {})
        traditional = comparison.get('traditional_gpt', {})
        accelerated = comparison.get('accelerated_cerebras', {})
        metrics = comparison.get('velocity_metrics', {})
        
        summary = f"""
## 🏥 MSK Research Velocity Results

### 📊 Performance Comparison
| Approach | Hypotheses | Time (s) | Speed |
|----------|------------|----------|-------|
| Traditional (GPT) | {traditional.get('hypotheses_generated', 0)} | {traditional.get('total_time', 0):.1f}s | Baseline |
| Accelerated (Cerebras) | {accelerated.get('hypotheses_generated', 0)} | {accelerated.get('total_time', 0):.1f}s | {metrics.get('research_velocity_gain', 0):.1f}x faster |

### 🚀 Key Metrics
- **Hypothesis Speedup**: {metrics.get('hypothesis_speedup', 0):.1f}x more hypotheses
- **Time Savings**: {metrics.get('time_saved_percentage', 0):.1f}% faster
- **Research Velocity**: {metrics.get('research_velocity_gain', 0):.1f}x acceleration

### 🎯 Clinical Impact
✓ More hypothesis cycles per research session
✓ Faster identification of actionable biomarkers  
✓ Accelerated precision oncology discovery pipeline
✓ Increased research productivity for clinical translation
"""
        return summary
    
    def _format_detailed_results(self, results: Dict[str, Any]) -> str:
        """Format detailed results for display"""
        comparison = results.get('comparison', {})
        traditional = comparison.get('traditional_gpt', {})
        accelerated = comparison.get('accelerated_cerebras', {})
        
        details = f"""
## 🔬 Detailed Analysis

### Traditional Approach (GPT)
- **Hypotheses Generated**: {traditional.get('hypotheses_generated', 0)}
- **Generation Time**: {traditional.get('generation_time', 0):.1f}s
- **Testing Time**: {traditional.get('testing_time', 0):.1f}s  
- **Analysis Time**: {traditional.get('analysis_time', 0):.1f}s
- **Clinical Insights**: {traditional.get('clinical_insights', 'N/A')[:200]}...

### Accelerated Approach (Cerebras)
- **Hypotheses Generated**: {accelerated.get('hypotheses_generated', 0)}
- **Generation Time**: {accelerated.get('generation_time', 0):.1f}s
- **Testing Time**: {accelerated.get('testing_time', 0):.1f}s
- **Analysis Time**: {accelerated.get('analysis_time', 0):.1f}s  
- **Clinical Insights**: {accelerated.get('clinical_insights', 'N/A')[:200]}...

### Technical Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   GPT       │    │  Cerebras   │    │ cBioPortal  │
│  (Planner)  │───▶│ (Executor)  │───▶│  API Tools  │
│             │    │             │    │             │
│ • Strategy  │    │ • Rapid API │    │ • Mutation  │
│ • Quality   │    │   calls     │    │   data      │
│ • Clinical  │    │ • Speed     │    │ • Frequency │
│   insights  │    │             │    │   analysis  │
└─────────────┘    └─────────────┘    └─────────────┘
```
"""
        return details
    
    def _generate_performance_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Generate performance comparison chart"""
        comparison = results.get('comparison', {})
        traditional = comparison.get('traditional_gpt', {})
        accelerated = comparison.get('accelerated_cerebras', {})
        
        # Create bar chart
        fig = go.Figure()
        
        models = ['Traditional (GPT)', 'Accelerated (Cerebras)']
        hypotheses = [traditional.get('hypotheses_generated', 0), 
                     accelerated.get('hypotheses_generated', 0)]
        times = [traditional.get('total_time', 0), 
                accelerated.get('total_time', 0)]
        
        # Add hypothesis bars
        fig.add_trace(go.Bar(
            x=models,
            y=hypotheses,
            name='Hypotheses Generated',
            marker_color='#2E86AB',
            yaxis='y'
        ))
        
        # Add time line
        fig.add_trace(go.Scatter(
            x=models,
            y=times,
            name='Execution Time (s)',
            mode='lines+markers',
            marker_color='#A23B72',
            yaxis='y2'
        ))
        
        # Create dual y-axis layout
        fig.update_layout(
            title='Multi-Agent Research Velocity Performance',
            xaxis_title='Approach',
            yaxis=dict(
                title='Hypotheses Generated',
                titlefont=dict(color='#2E86AB'),
                tickfont=dict(color='#2E86AB')
            ),
            yaxis2=dict(
                title='Execution Time (s)',
                titlefont=dict(color='#A23B72'),
                tickfont=dict(color='#A23B72'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.1, y=1.1, orientation='h'),
            height=500
        )
        
        return fig
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="MSK Multi-Agent Research Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🏥 Memorial Sloan Kettering Multi-Agent Research Demo
            ### Accelerating Biomedical Discovery with Cerebras Fast Inference
            
            This interactive demo showcases how frontier models (planners) and fast inference models (executors) 
            can work together to accelerate precision oncology research workflows.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Demo Configuration")
                    
                    time_budget = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Time Budget (seconds)",
                        info="Research time budget for hypothesis generation"
                    )
                    
                    use_dry_run = gr.Checkbox(
                        value=True,
                        label="Use Dry-Run Mode",
                        info="Reliable demo with cached responses (recommended for presentations)"
                    )
                    
                    run_btn = gr.Button(
                        "🚀 Run Demo",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Demo Results")
                    
                    with gr.Tab("Summary"):
                        summary_output = gr.Markdown("")
                    
                    with gr.Tab("Detailed Results"):
                        details_output = gr.Markdown("")
                    
                    with gr.Tab("Performance Chart"):
                        chart_output = gr.Plot(label="Research Velocity Performance")
                    
                    with gr.Tab("Download Results"):
                        gr.Markdown("### 📥 Download Demo Results")
                        download_info = gr.Markdown("Run the demo to generate downloadable results.")
                        download_file = gr.File(label="Download JSON Results", visible=False)
            
            # Event handlers
            run_btn.click(
                fn=self.run_demo_with_progress,
                inputs=[time_budget, use_dry_run],
                outputs=[summary_output, details_output, chart_output, download_file, download_info]
            )
            
            gr.Markdown("""
            ---
            ### 🎯 Clinical Value Proposition
            - **Faster biomarker discovery** for precision treatment
            - **More therapeutic options explored** per research session  
            - **Accelerated research-to-clinic pipeline**
            - **Real clinical insights** generated more rapidly
            
            ### 🔬 Technical Innovation
            - **Multi-agent architecture**: Planner (GPT-4) + Executor (Cerebras)
            - **cBioPortal integration**: Real cancer mutation data from MSK-IMPACT studies
            - **Performance tracking**: Measures actual research velocity improvements
            - **Clinical focus**: Precision oncology biomarker discovery
            """)
        
        return interface

def main():
    """Launch the Gradio demo"""
    
    print("🌐 Starting Gradio Interface for MSK Multi-Agent Demo")
    
    # Check environment
    if not os.getenv("CEREBRAS_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("⚠️  API keys not found. Dry-run mode will be used.")
        print("   Set CEREBRAS_API_KEY and OPENAI_API_KEY in .env file for live demo.")
    
    try:
        demo_interface = GradioResearchDemo()
        interface = demo_interface.create_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to launch Gradio interface: {e}")
        print("   Install gradio with: pip install gradio")

if __name__ == "__main__":
    main()
