# MSK Quick Start Guide
## Multi-Agent Research Velocity Demo

### 🚀 5-Minute Setup

```bash
# 1. Clone and navigate
cd biomni-chat-example

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Set up API keys (Recommended method)
cp .env.example .env
# Edit .env file with your actual API keys:
# CEREBRAS_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

# 4. Verify setup
python test_environment.py
```

**Alternative: Environment Variables**
```bash
export CEREBRAS_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

### 🎮 Run the Demo

#### **For Live Presentation (Recommended)**
```bash
python final_demo.py --dry-run
```
*Uses cached responses - no API dependency*

#### **For Testing with Real APIs**
```bash
python final_demo.py
```

### 📊 Expected Results

```
🔬 RESEARCH VELOCITY RESULTS
==================================================
Traditional (GPT-4): 8 hypotheses, 45.2s total
Accelerated (Cerebras): 24 hypotheses, 28.7s total
🚀 Hypothesis Speedup: 3.0x more hypotheses
⏱️  Time Savings: 16.5s (36.5% faster)
📈 Velocity Gain: 1.6x research acceleration
```

### 📈 Generate Charts
```bash
python visualize_results.py
```

### 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| API key errors | Check environment variables |
| Import errors | Run `pip install -r requirements.txt` |
| Rate limiting | Use `--dry-run` mode |
| Network issues | Use cached responses |

### 📞 Support

**Technical Lead:** Solutions Architecture Team  
**Demo Files:** All in current directory  
**Documentation:** README.md, executive_summary.md

---

**Memorial Sloan Kettering Cancer Center**  
*Precision Oncology Research Acceleration*
