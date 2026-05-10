# MARCO: Multi-Agent Recommendation Framework

**Code for the paper:** *Fewer Tokens, Smaller Agents: Role-Aware Allocation for Efficient Multi-Agent Recommendation*

**Submitted to SIGIR'26 - Short Papers Track**

MARCO is a multi-agent framework for recommendation systems using structured Plan-Work-Solve reasoning. The system coordinates specialized agents (Planner, Analyst, Solver, and optional Reflector) through three-phase reasoning to perform rating prediction and sequential recommendation tasks.

---

## Overview

MARCO implements a role-aware multi-agent collaboration pattern:

- **Planner**: Decomposes tasks into structured execution plans
- **Analyst**: Analyzes users/items using retrieval tools
- **Solver**: Synthesizes analysis results into recommendations
- **Reflector** (optional): Performs quality checks and triggers refinement

**Supported Tasks:**
- `rp`: Rating Prediction
- `sr`: Sequential Recommendation

**Supported Datasets:**
- MovieLens 100k (`ml-100k`)
- Amazon Beauty (`Beauty`)
- Amazon Electronics (`Electronics`)
- Yelp 2020 (`Yelp2020`)

---

## Project Structure

```
MARCO/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── config/
│   ├── api-config.json         # API keys (create from example)
│   ├── agents/                 # Agent configurations
│   │   ├── planner.json
│   │   ├── analyst.json
│   │   ├── solver.json
│   │   └── reflector.json
│   ├── systems/marco/          # System configurations
│   │   ├── basic.json          # Planner + Analyst + Solver
│   │   └── reflector.json      # + Reflector agent
│   ├── prompts/                # Prompt templates
│   └── tools/                  # Tool configurations
├── marco/                      # Core framework
│   ├── agents/                 # Agent implementations
│   ├── systems/                # System orchestration
│   ├── llms/                   # LLM provider integrations
│   ├── tasks/                  # Task implementations
│   ├── tools/                  # Retrieval and utility tools
│   ├── evaluation/             # Metrics
│   └── dataset/                # Dataset handling
├── recommender/                # Baseline models
│   ├── models/                 # LightGCN, SASRec, BERT4Rec
│   └── run.py                  # Training script
├── data/                       # Datasets
└── logs/                       # Execution logs
```

---

## Installation

### 1. Create Environment
```powershell
conda create -n MARCO python=3.10
conda activate MARCO
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy and edit the API configuration:
```powershell
copy config\api-config-example.json config\api-config.json
```

Edit `config/api-config.json` with your API keys:
```json
{
    "providers": {
        "gemini": {
            "api_key": ["YOUR_KEY"],
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models"
        },
        "vertexai": {
            "project_id": "YOUR_GCP_PROJECT_ID",
            "location": "us-central1",
            "credentials_path": "config/vertex-service-account.json"
        },
        "openrouter": {
            "api_key": ["YOUR_KEY"],
            "base_url": "https://openrouter.ai/api/v1/chat/completions"
        },
        "openai": {
            "api_key": ["YOUR_KEY"],
            "base_url": "https://api.openai.com/v1/"
        },
        "ollama": {
            "base_url": "http://localhost:11434"
        }
    }
}
```

---

## How to Run

### Test (Quick Validation)
Test on a small sample:
```powershell
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/basic.json --task sr --samples 100
```

**Test Options:**
- `--samples N`: Number of samples to test (default: 5)
- `--random`: Random sampling instead of sequential
- `--last`: Sample from end of dataset
- `--offset N`: Skip first N samples
- `--offsetGT N`: Skip first N GT-filtered samples

### Evaluate (Full Dataset)
Run full evaluation with metrics:
```powershell
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/basic.json --task sr
```

**Evaluate Options:**
- `--topks`: Top-K values for ranking metrics (default: [1,3,5])

### With Reflector Agent
Enable quality checking and refinement:
```powershell
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/reflector.json --task sr --samples 100
```

### Specify LLM Provider/Model
```powershell
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/basic.json --task sr --samples 100 --provider gemini --model gemini-2.0-flash
```

**Available Providers:**
- `gemini`: Google Gemini
- `vertexai`: Vertex AI Gemini
- `openrouter`: 200+ models via OpenRouter
- `openai`: OpenAI models
- `ollama`: Local inference
- `huggingface`: HuggingFace models

---

## Data Preprocessing

Preprocess datasets before running:

```powershell
# MovieLens 100k
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 7

# Amazon categories (e.g., Beauty)
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 7

# Yelp 2020
python main.py --main Preprocess --data_dir data/Yelp2020 --dataset yelp2020 --n_neg_items 7
```

---

## Training Baseline Models

Train recommendation models for candidate generation:

```powershell
# LightGCN
python -m recommender.run --model lightgcn --data data/ml-100k/all.csv --epochs 200 --batch_size 2048 --lr 0.01 --export_topk 20

# SASRec
python -m recommender.run --model sasrec --data data/ml-100k/all.csv --epochs 100 --batch_size 256 --lr 0.001 --export_topk 20

# BERT4Rec
python -m recommender.run --model bert4rec --data data/ml-100k/all.csv --epochs 100 --batch_size 256 --lr 0.001 --export_topk 20
```

---

## Common Options

- `--verbose`: Log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
- `--max_his`: Maximum history length (default: 10)
- `--dataset`: Dataset name for prompt formatting

---

## Output

**Logs:** Saved to `logs/` with pattern:
```
{task}_{dataset}_{system}_{samples}_{timestamp}.txt
```

**Results:** Includes metrics, token usage, and sample-level outputs

---

## License

[To be added upon publication]

## Citation

[To be added upon publication]
