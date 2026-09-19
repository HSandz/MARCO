# MARCO: Multi-Agent Recommendation Framework

**Code for the paper:** *Fewer Tokens, Smaller Agents: Role-Aware Allocation for Efficient Multi-Agent Recommendation*

**Submitted to SOICT 2026**

MARCO is a multi-agent recommendation framework that coordinates specialized agents (Planner, Analyst, Solver, and optional Reflector) through structured Plan-Work-Solve reasoning for rating prediction and sequential recommendation.

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
│   ├── systems/single/         # Experimental comparison configuration
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
├── scripts/                    # Helper scripts
├── data/                       # Datasets
├── cached/                     # LLM cache
├── logs/                       # Execution logs
└── results/                    # Structured JSON run results
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
            "api_key": "YOUR_VERTEX_AI_API_KEY",
            "project_id": "YOUR_GCP_PROJECT_ID",
            "location": "global",
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
        "codexhub": {
            "api_key": ["YOUR_KEY"],
            "base_url": "https://api.codexhub.click/v1",
            "model": "oc/deepseek-v4-flash-free"
        },
        "ollama": {
            "base_url": "http://localhost:11434"
        }
    }
}
```

For Vertex AI, `api_key` is optional. If it is set, MARCO uses Vertex AI API-key authentication and ignores `project_id` and `credentials_path`; leave it empty to use a project/service-account setup.

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
- `--steps N`: Number of generation/evaluation steps (default: 1)
- `--topks`: Top-K values for ranking metrics (default: [1,3,5])

### With Reflector Agent
Enable quality checking and refinement:
```powershell
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/reflector.json --task sr --samples 100
```

Disable automatic reflector-triggered reruns with `--disable-reflection-rerun`.

### Specify LLM Provider/Model
```powershell
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/basic.json --task sr --samples 100 --provider gemini --model gemini-2.0-flash
```

**Available Providers:**
- `gemini`: Google Gemini
- `vertexai`: Vertex AI Gemini
- `openrouter`: 200+ models via OpenRouter
- `openai`: OpenAI models
- `codexhub`: CodexHub OpenAI-compatible models
- `ollama`: Local inference
- `huggingface`: HuggingFace models

### Vertex AI Batch Mode
Batch prediction is available for Vertex AI runs. Use `--batch` with either a GCS output prefix or an existing JSONL input:
```powershell
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/reflector.json --task sr --samples 200 --provider vertexai --model gemini-2.5-flash --batch --batch_gcs_uri_prefix gs://your-bucket/marco-batch/run-001
```

---

## Data Preprocessing

Preprocess datasets before running:

```powershell
# MovieLens 100k
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 9

# Amazon categories (e.g., Beauty)
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 9

# Yelp 2020
python main.py --main Preprocess --data_dir data/Yelp2020 --dataset yelp2020 --n_neg_items 9
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
- `--api_config`: API configuration path (default: `config/api-config.json`)
- `--no_cache`: Disable Planner/Analyst cache reads while still writing successful uncached responses
- `--provider`, `--model`: Override provider/model from config at runtime

Additional task entry points are available through `--main`: `Generation`, `TestGeneration`, `Sample`, and `Calculate`.

---

## Output

**Logs:** Saved to `logs/` with pattern:
```
{task}_{dataset}_{system}_{samples}_{timestamp}.log
```

**Results:** Saved to `results/` as JSON with the same base filename. Each file includes:
- run metadata, data file, and task configuration
- full metric summary, cumulative evaluation snapshots, runtime, token/API usage
- per-agent token/API usage and model details
- per-sample outputs, including solver ranked lists and reflection reruns

---

## License

[To be added upon publication]

## Citation

[To be added upon publication]
