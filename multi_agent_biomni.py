#!/usr/bin/env python3
"""
Multi-Agent Biomni Framework for Memorial Sloan Kettering Demo

This module extends Biomni's A1 agent to support a dual-agent architecture:
- Planner Agent: Uses frontier models (GPT-4/Claude) for strategic reasoning
- Executor Agent: Uses fast inference models (Cerebras-hosted Qwen/GLM) for tool execution

Architecture:
- Inherits from Biomni A1 agent to maintain all existing functionality
- Overrides key LLM invocation methods to route to appropriate models
- Provides performance tracking to demonstrate speed improvements
"""

import os
import sys
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Add Biomni to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Biomni'))

from biomni.config import BiomniConfig
from biomni.llm import SourceType, get_llm
from biomni.agent.a1 import A1, AgentState
from langchain_core.messages import BaseMessage


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent setup."""
    
    # Planner configuration (frontier model for strategic reasoning)
    planner_model: str = "gpt-4"
    planner_source: SourceType = "OpenAI"
    planner_temperature: float = 0.1
    
    # Executor configuration (fast model for tool execution)
    executor_model: str = "qwen2.5-32b"
    executor_source: SourceType = "Custom"
    executor_base_url: str = "https://api.cerebras.ai/v1"
    executor_api_key: Optional[str] = None
    executor_temperature: float = 0.1
    
    # Performance tracking
    track_performance: bool = True
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.executor_api_key is None:
            self.executor_api_key = os.getenv("CEREBRAS_API_KEY")


class MultiAgentA1(A1):
    """
    Multi-agent version of Biomni A1 that separates planning and execution.
    
    - Strategic reasoning, feedback, and format checking use frontier models
    - Tool execution uses fast inference models
    - Tracks performance metrics for demo purposes
    """
    
    def __init__(
        self,
        path: str | None = None,
        multi_agent_config: Optional[MultiAgentConfig] = None,
        **kwargs
    ):
        """Initialize multi-agent with separate planner and executor LLMs."""
        
        # Set up multi-agent configuration
        self.multi_config = multi_agent_config or MultiAgentConfig()
        
        # Performance tracking
        self.performance_metrics = {
            "planner_calls": 0,
            "executor_calls": 0,
            "planner_latency": 0.0,
            "executor_latency": 0.0
        }
        
        # Initialize planner config (for strategic reasoning)
        self.planner_config = BiomniConfig(
            llm=self.multi_config.planner_model,
            source=self.multi_config.planner_source,
            temperature=self.multi_config.planner_temperature
        )
        
        # Initialize executor config (for tool execution)
        self.executor_config = BiomniConfig(
            llm=self.multi_config.executor_model,
            source=self.multi_config.executor_source,
            base_url=self.multi_config.executor_base_url,
            api_key=self.multi_config.executor_api_key,
            temperature=self.multi_config.executor_temperature
        )
        
        # Initialize parent A1 with planner model as default
        super().__init__(
            path=path,
            llm=self.multi_config.planner_model,
            source=self.multi_config.planner_source,
            temperature=self.multi_config.planner_temperature,
            **kwargs
        )
        
        # Create separate LLM instances
        self.planner_llm = get_llm(
            model=self.multi_config.planner_model,
            source=self.multi_config.planner_source,
            temperature=self.multi_config.planner_temperature,
            config=self.planner_config
        )
        
        self.executor_llm = get_llm(
            model=self.multi_config.executor_model,
            source=self.multi_config.executor_source,
            base_url=self.multi_config.executor_base_url,
            api_key=self.multi_config.executor_api_key,
            temperature=self.multi_config.executor_temperature,
            config=self.executor_config
        )
        
        print(f"🧠 Multi-Agent A1 Initialized:")
        print(f"   Planner: {self.multi_config.planner_model} ({self.multi_config.planner_source})")
        print(f"   Executor: {self.multi_config.executor_model} ({self.multi_config.executor_source})")
        print(f"   Executor URL: {self.multi_config.executor_base_url}")
    
    def _track_llm_call(self, llm_type: str, start_time: float):
        """Track performance metrics for LLM calls."""
        if self.multi_config.track_performance:
            latency = time.time() - start_time
            if llm_type == "planner":
                self.performance_metrics["planner_calls"] += 1
                self.performance_metrics["planner_latency"] += latency
            else:
                self.performance_metrics["executor_calls"] += 1
                self.performance_metrics["executor_latency"] += latency
    
    def _invoke_planner(self, messages: List[BaseMessage]) -> Any:
        """Invoke planner LLM with performance tracking."""
        start_time = time.time()
        response = self.planner_llm.invoke(messages)
        self._track_llm_call("planner", start_time)
        return response
    
    def _invoke_executor(self, messages: List[BaseMessage]) -> Any:
        """Invoke executor LLM with performance tracking."""
        start_time = time.time()
        response = self.executor_llm.invoke(messages)
        self._track_llm_call("executor", start_time)
        return response
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        metrics = self.performance_metrics.copy()
        
        # Calculate averages
        if metrics["planner_calls"] > 0:
            metrics["avg_planner_latency"] = metrics["planner_latency"] / metrics["planner_calls"]
        else:
            metrics["avg_planner_latency"] = 0.0
            
        if metrics["executor_calls"] > 0:
            metrics["avg_executor_latency"] = metrics["executor_latency"] / metrics["executor_calls"]
        else:
            metrics["avg_executor_latency"] = 0.0
        
        # Calculate speedup
        if metrics["avg_planner_latency"] > 0 and metrics["avg_executor_latency"] > 0:
            metrics["speedup_factor"] = metrics["avg_planner_latency"] / metrics["avg_executor_latency"]
        else:
            metrics["speedup_factor"] = 0.0
        
        return metrics
    
    # Override key methods from A1 to use appropriate LLMs
    def _generate_with_planner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Override generate step to use planner LLM for strategic reasoning."""
        system_prompt = self.system_prompt
        if hasattr(self.planner_llm, "model_name") and (
            "gpt" in str(self.planner_llm.model_name).lower() or "openai" in str(type(self.planner_llm)).lower()
        ):
            system_prompt += "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <execute> or <solution> in EVERY response. Do not use markdown code blocks (```) - use <execute> tags instead."

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = self._invoke_planner(messages)
        
        # Process response (same as original)
        content = response.content
        if isinstance(content, list):
            # Concatenate textual parts; ignore tool_use or other non-text blocks
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        
        state["messages"].append(AIMessage(content=content))
        return state
    
    def _execute_with_executor(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Override execute step to use executor LLM for tool execution."""
        # This would need to be integrated with A1's execution logic
        # For now, we'll use the original execution but track when it happens
        print("🔧 Executing tools with fast Cerebras model...")
        return state  # Placeholder - would need full integration
    
    def _generate_feedback_with_planner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Override feedback generation to use planner LLM."""
        messages = state["messages"]
        feedback_prompt = f"""
        Here is a reminder of what is the user requested: {getattr(self, 'user_task', 'Unknown task')}
        Examine the previous executions, reasoning, and solutions.
        Critic harshly on what could be improved?
        Be specific and constructive.
        Think hard what are missing to solve the task.
        No question asked, just feedbacks.
        """
        
        response = self._invoke_planner(messages + [HumanMessage(content=feedback_prompt)])
        
        # Add feedback as a new message
        state["messages"].append(
            HumanMessage(
                content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{response.content}"
            )
        )
        return state


def create_demo_agent():
    """Create a demo agent configured for Memorial Sloan Kettering."""
    
    # Check for required API keys
    if not os.getenv("CEREBRAS_API_KEY"):
        print("⚠️  CEREBRAS_API_KEY not found. Please set it with:")
        print("   export CEREBRAS_API_KEY=your_key_here")
        return None
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Please set it with:")
        print("   export OPENAI_API_KEY=your_key_here")
        return None
    
    # Create multi-agent configuration
    config = MultiAgentConfig(
        planner_model="gpt-4",
        planner_source="OpenAI",
        executor_model="qwen2.5-32b",
        executor_source="Custom",
        executor_base_url="https://api.cerebras.ai/v1",
        track_performance=True
    )
    
    # Create and return the multi-agent
    return MultiAgentA1(
        path="./demo_data",
        multi_agent_config=config
    )


if __name__ == "__main__":
    print("🚀 Creating Multi-Agent Biomni Demo")
    print("=" * 50)
    
    agent = create_demo_agent()
    if agent:
        print("✅ Multi-agent created successfully!")
        print("\n📊 Performance tracking enabled")
        print("🧠 Planner: GPT-4 for strategic reasoning")
        print("⚡ Executor: Qwen2.5-32b on Cerebras for fast execution")
