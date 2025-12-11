#!/usr/bin/env python3
"""
Multi-Agent Demo for Memorial Sloan Kettering
Actionable Biomarker Discovery in Precision Oncology

This demo showcases how frontier models (planners) and fast inference models (executors)
can work together to accelerate biomedical research workflows.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   You can still use environment variables without .env file support.")

# Add Biomni to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Biomni'))

from biomni.config import BiomniConfig
from biomni.llm import get_llm

# Import our cbioportal tools
from cbioportal_tools import (
    query_mutations_by_gene,
    analyze_mutation_frequency,
    get_molecular_profiles,
    comprehensive_mutation_analysis
)


class MultiAgentDemo:
    """
    Simplified multi-agent demo that separates planning and execution.
    Demonstrates the concept without complex A1 integration.
    """
    
    def __init__(self):
        """Initialize the multi-agent demo with planner and executor LLMs."""
        
        # Check for required API keys
        if not os.getenv("CEREBRAS_API_KEY"):
            raise ValueError("CEREBRAS_API_KEY environment variable required")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        # Initialize planner (GPT-4 for strategic reasoning)
        self.planner_config = BiomniConfig(
            llm="gpt-4",
            source="OpenAI",
            temperature=0.1
        )
        
        self.planner_llm = get_llm(
            model="gpt-4",
            source="OpenAI",
            temperature=0.1,
            config=self.planner_config
        )
        
        # Initialize executor (Cerebras Qwen for fast execution)
        self.executor_config = BiomniConfig(
            llm="qwen2.5-32b",
            source="Custom",
            base_url="https://api.cerebras.ai/v1",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            temperature=0.1
        )
        
        self.executor_llm = get_llm(
            model="qwen2.5-32b",
            source="Custom",
            base_url="https://api.cerebras.ai/v1",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            temperature=0.1,
            config=self.executor_config
        )
        
        # Performance tracking
        self.metrics = {
            "planner_calls": 0,
            "executor_calls": 0,
            "planner_latency": 0.0,
            "executor_latency": 0.0
        }
        
        print("🧠 Multi-Agent Demo Initialized")
        print(f"   Planner: GPT-4 (strategic reasoning)")
        print(f"   Executor: Qwen2.5-32b on Cerebras (fast execution)")
    
    def _call_planner(self, prompt: str) -> str:
        """Call planner LLM with performance tracking."""
        start_time = time.time()
        response = self.planner_llm.invoke(prompt)
        latency = time.time() - start_time
        
        self.metrics["planner_calls"] += 1
        self.metrics["planner_latency"] += latency
        
        print(f"🧠 Planner response ({latency:.2f}s)")
        return response.content
    
    def _call_executor(self, prompt: str) -> str:
        """Call executor LLM with performance tracking."""
        start_time = time.time()
        response = self.executor_llm.invoke(prompt)
        latency = time.time() - start_time
        
        self.metrics["executor_calls"] += 1
        self.metrics["executor_latency"] += latency
        
        print(f"⚡ Executor response ({latency:.2f}s)")
        return response.content
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        avg_planner = self.metrics["planner_latency"] / max(self.metrics["planner_calls"], 1)
        avg_executor = self.metrics["executor_latency"] / max(self.metrics["executor_calls"], 1)
        
        return {
            "planner_calls": self.metrics["planner_calls"],
            "executor_calls": self.metrics["executor_calls"],
            "avg_planner_latency": avg_planner,
            "avg_executor_latency": avg_executor,
            "speedup_factor": avg_planner / avg_executor if avg_executor > 0 else 0,
            "total_time_saved": (avg_planner - avg_executor) * self.metrics["executor_calls"]
        }
    
    def run_actionable_biomarker_demo(self) -> Dict[str, Any]:
        """
        Run the actionable biomarker discovery demo.
        
        Clinical Scenario: Identify rare but targetable mutations across
        multiple cancer types that could inform precision oncology treatment.
        """
        
        print("\n" + "="*60)
        print("🏥 ACTIONABLE BIOMARKER DISCOVERY DEMO")
        print("Memorial Sloan Kettering Precision Oncology Use Case")
        print("="*60)
        
        demo_results = {
            "scenario": "Actionable Biomarker Discovery for Precision Oncology",
            "steps": [],
            "findings": {},
            "performance": {}
        }
        
        # Step 1: Strategic Planning (Frontier Model)
        print("\n📋 STEP 1: Strategic Planning with GPT-4")
        print("-" * 40)
        
        planning_prompt = """
        You are a precision oncology researcher at Memorial Sloan Kettering.
        Your task is to identify actionable biomarkers for targeted therapy.
        
        Based on current clinical practice, which gene mutations are most important
        to analyze across cancer types for targeted therapy decisions?
        
        Please provide:
        1. A prioritized list of 5-7 key genes for actionable mutation analysis
        2. Rationale for each gene selection
        3. Suggested cancer types to focus on
        
        Format your response as a structured plan that will guide data analysis.
        """
        
        strategic_plan = self._call_planner(planning_prompt)
        demo_results["steps"].append({
            "step": "Strategic Planning",
            "model": "GPT-4",
            "output": strategic_plan[:500] + "..." if len(strategic_plan) > 500 else strategic_plan
        })
        
        # Step 2: Rapid Data Execution (Cerebras)
        print("\n🔬 STEP 2: Rapid Data Execution with Cerebras")
        print("-" * 40)
        
        # Extract genes from planner response (simplified for demo)
        key_genes = ["TP53", "EGFR", "ALK", "BRAF", "KRAS", "PIK3CA", "BRCA1"]
        
        execution_prompt = f"""
        Execute the following biomedical data analysis tasks rapidly:
        
        Genes to analyze: {key_genes}
        Task: Query mutation frequencies across major cancer studies
        
        For each gene, use the cbioportal tools to:
        1. Query mutation data
        2. Analyze frequency across studies
        3. Identify rare but clinically relevant alterations
        
        Execute these tasks efficiently and summarize findings.
        """
        
        execution_plan = self._call_executor(execution_prompt)
        demo_results["steps"].append({
            "step": "Data Execution Planning",
            "model": "Qwen2.5-32b (Cerebras)",
            "output": execution_plan[:500] + "..." if len(execution_plan) > 500 else execution_plan
        })
        
        # Step 3: Execute cbioportal queries
        print("\n📊 STEP 3: Executing cbioportal Data Queries")
        print("-" * 40)
        
        query_results = {}
        total_query_time = 0
        
        for gene in key_genes[:3]:  # Limit to 3 genes for demo speed
            print(f"   Querying {gene}...")
            start_time = time.time()
            
            # Use our cbioportal tools
            mutation_data = query_mutations_by_gene(gene)
            frequency_data = analyze_mutation_frequency([gene])
            
            query_time = time.time() - start_time
            total_query_time += query_time
            
            query_results[gene] = {
                "mutation_data": json.loads(mutation_data),
                "frequency_data": json.loads(frequency_data),
                "query_time": query_time
            }
            
            print(f"   ✓ {gene} completed ({query_time:.2f}s)")
        
        demo_results["findings"]["query_results"] = query_results
        demo_results["findings"]["total_query_time"] = total_query_time
        
        # Step 4: Clinical Interpretation (Frontier Model)
        print("\n🏥 STEP 4: Clinical Interpretation with GPT-4")
        print("-" * 40)
        
        # Prepare clinical summary
        clinical_summary = "CLINICAL FINDINGS SUMMARY:\n\n"
        for gene, data in query_results.items():
            freq_data = data["frequency_data"]
            if "mutation_frequencies" in freq_data:
                clinical_summary += f"{gene}: "
                for study_info in freq_data["mutation_frequencies"].get(gene, []):
                    clinical_summary += f"{study_info['frequency_percent']}% in {study_info['study_id']}; "
                clinical_summary += "\n"
        
        interpretation_prompt = f"""
        As a Memorial Sloan Kettering oncologist, interpret these mutation frequency findings:
        
        {clinical_summary}
        
        Provide clinical insights on:
        1. Which mutations are most clinically actionable
        2. Potential targeted therapy options
        3. Recommendations for precision treatment
        4. Rare mutations that may warrant further investigation
        
        Focus on actionable insights for patient care.
        """
        
        clinical_insights = self._call_planner(interpretation_prompt)
        demo_results["steps"].append({
            "step": "Clinical Interpretation",
            "model": "GPT-4",
            "output": clinical_insights[:500] + "..." if len(clinical_insights) > 500 else clinical_insights
        })
        
        # Performance Summary
        print("\n📈 PERFORMANCE SUMMARY")
        print("-" * 40)
        
        performance = self.get_performance_summary()
        demo_results["performance"] = performance
        
        print(f"   Planner calls: {performance['planner_calls']}")
        print(f"   Executor calls: {performance['executor_calls']}")
        print(f"   Avg planner latency: {performance['avg_planner_latency']:.2f}s")
        print(f"   Avg executor latency: {performance['avg_executor_latency']:.2f}s")
        print(f"   Speedup factor: {performance['speedup_factor']:.2f}x")
        print(f"   Total time saved: {performance['total_time_saved']:.2f}s")
        
        print("\n🎯 DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return demo_results


def main():
    """Run the multi-agent demo."""
    
    try:
        demo = MultiAgentDemo()
        results = demo.run_actionable_biomarker_demo()
        
        # Save results
        with open("demo_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to demo_results.json")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("\nPlease ensure you have set the required environment variables:")
        print("   export CEREBRAS_API_KEY=your_key_here")
        print("   export OPENAI_API_KEY=your_key_here")


if __name__ == "__main__":
    main()
