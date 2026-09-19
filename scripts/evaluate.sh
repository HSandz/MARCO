#! /bin/bash

# Quick test on 100 samples from MovieLens-100k dataset on Sequential Recommendation task
## Evaluate on full dataset using --main Evaluate and without --samples
## Evaluate on Rating Prediction task using --task rp
## Evaluate with other datasets such as Amazon-Beauty by changing --data_file

### config : marco = Planner + Analyst + Solver
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/basic.json --task sr --samples 100
### config : marco = Planner + Analyst + Solver + Reflector
python main.py --main Test --data_file data/ml-100k/test.csv --system marco --system_config config/systems/marco/reflector.json --task sr --samples 100

### Other params
# --max_his : max history length for sequential recommendation task
# --steps : number of interaction steps between LLM agents
# --topks : top-K values for evaluation metrics, e.g., Recall@K, NDCG@K
# --samples : number of samples to evaluate, remove this option to evaluate on the full dataset
# --verbose : print the detailed interaction process