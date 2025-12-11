# Multi-Agent Biomedical Research Demo
## Memorial Sloan Kettering Precision Oncology Use Case

This demo showcases how fast inference with Cerebras complements frontier models like GPT-4 to accelerate biomedical research workflows, specifically for actionable biomarker discovery in precision oncology.

### 🎯 Clinical Value Proposition

**Traditional Approach**: GPT-4 plans and executes everything → Limited hypothesis cycles due to slower iteration
**Multi-Agent Approach**: GPT-4 provides strategic reasoning + Cerebras enables rapid execution → **3-5x more research hypotheses tested in same time**

### 🏥 Demo Scenario: Actionable Biomarker Discovery

The demo simulates a Memorial Sloan Kettering precision oncology research workflow:
1. **Strategic Planning** (GPT-4): Identify key cancer genes and analysis strategy
2. **Rapid Execution** (Cerebras): Generate and test multiple research hypotheses quickly
3. **Clinical Interpretation**: Translate findings into actionable treatment insights

### 🚀 Key Features

- **Cerebras Integration**: Custom provider route through Biomni framework
- **cBioPortal Tools**: Real cancer mutation data from MSK-IMPACT and TCGA studies
- **Multi-Agent Architecture**: Planner (GPT-4) + Executor (Cerebras) separation
- **Performance Tracking**: Measures actual research velocity improvements
- **Clinical Focus**: Precision oncology biomarker discovery

### 📋 Prerequisites

#### API Keys Required

**Option 1: Using .env file (Recommended)**
```bash
# Copy the template and fill in your keys
cp .env.example .env
# Edit .env file with your actual API keys:
# CEREBRAS_API_KEY=your_cerebras_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
```

**Option 2: Environment Variables**
```bash
# Set up environment variables
export CEREBRAS_API_KEY=your_cerebras_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
```

#### Python Environment
```bash
# Clone the repository (includes Biomni framework)
git clone <repository_url>
cd biomni-chat-example

# Install dependencies
pip install -r requirements.txt
```

### 🛠️ Installation

1. **Clone and Setup**
```bash
git clone <repository_url>
cd biomni-chat-example
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API Keys**
```bash
export CEREBRAS_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

4. **Verify Installation**
```bash
python test_environment.py
```

### 🎮 Run the Demo

#### **Primary Demo (CLI)**
```bash
# For Live Presentation (Reliable)
python3 final_demo.py --dry-run

# For Testing with Real APIs  
python3 final_demo.py
python enhanced_demo.py
```

### 📊 Expected Output

The demo generates comparative results showing:

```
🔬 RESEARCH VELOCITY RESULTS
==================================================
Traditional (GPT-4): 8 hypotheses, 45.2s total
Accelerated (Cerebras): 24 hypotheses, 28.7s total
🚀 Hypothesis Speedup: 3.0x more hypotheses
⏱️  Time Savings: 16.5s (36.5% faster)
📈 Velocity Gain: 1.6x research acceleration
```

### 📈 Performance Metrics

The demo tracks:
- **Hypothesis Generation Rate**: Number of research hypotheses per minute
- **Testing Cycles**: How many hypotheses can be experimentally validated
- **Clinical Insights**: Speed of generating actionable treatment recommendations
- **Overall Research Velocity**: Complete workflow acceleration

### 🏥 Clinical Impact for MSK

- **Faster Biomarker Discovery**: More therapeutic options explored per research session
- **Real-time Decision Support**: Rapid analysis for clinical trial matching
- **Accelerated Research Pipeline**: From hypothesis to clinical insight in minutes vs hours
- **Increased Productivity**: More publications and discoveries per research time

### 🔧 Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GPT-4         │    │   Cerebras       │    │   cBioPortal    │
│   (Planner)     │───▶│   (Executor)     │───▶│   API Tools     │
│                 │    │                  │    │                 │
│ • Strategy      │    │ • Rapid API      │    │ • Mutation      │
│ • Clinical      │    │   calls          │    │   data          │
│   insights      │    │ • Hypothesis     │    │ • Frequency     │
│ • Quality       │    │   generation     │    │   analysis      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 📁 File Structure

```
biomni-chat-example/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── test_environment.py          # Environment verification
├── final_demo.py               # Complete research velocity demo
├── enhanced_demo.py            # Comparative workflow analysis
├── demo_multi_agent.py         # Basic multi-agent demonstration
├── multi_agent_biomni.py       # Multi-agent architecture
├── cbioportal_tools.py         # cBioPortal integration tools
├── test_cerebras_integration.py # Cerebras API testing
└── Biomni/                     # Biomni framework submodule
```

### 🐛 Troubleshooting

#### Common Issues

1. **API Key Errors**
```bash
# Verify keys are set
echo $CEREBRAS_API_KEY
echo $OPENAI_API_KEY
```

2. **cBioPortal Rate Limits**
- The demo includes built-in delays to respect API limits
- If rate limiting occurs, the demo will automatically retry

3. **Import Errors**
```bash
# Ensure Biomni is properly installed
pip install biomni --upgrade
```

4. **Performance Issues**
- For best results, run on a machine with good internet connectivity
- Cerebras inference speed depends on network latency to their API

### 📞 Support

For questions about this demo:
- Technical issues: Check troubleshooting section above
- Clinical questions: Designed for Memorial Sloan Kettering precision oncology use case
- API access: Ensure proper Cerebras and OpenAI API credentials

### 📄 License

This demo is for educational and research purposes. Please respect the terms of service of all APIs used.

---

**Memorial Sloan Kettering Cancer Center**  
*Precision Oncology Research Acceleration Demo*
