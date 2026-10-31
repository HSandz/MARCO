#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
    cat <<'USAGE'
Usage: bash scripts/training.sh [MODEL [DATA_FILE [TEST_FILE]]] [CLI_OPTIONS...]
Defaults: sasrec data/ml-100k/all.csv data/ml-100k/test.csv
Models: lightgcn, sasrec, bert4rec. Trains one model, then exports top-20 candidates.
TEST_FILE selects the MARCO export rows; training still uses the CLI's existing split.
Example: bash scripts/training.sh lightgcn data/Beauty/all.csv data/Beauty/test.csv --device cpu
Set PYTHON to select the Python interpreter. Relative paths use the repository root.
USAGE
    exit 0
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
model="${1:-sasrec}"
if (( $# > 0 )); then shift; fi
data_file="${1:-data/ml-100k/all.csv}"
if (( $# > 0 )); then shift; fi
test_file="${1:-$(dirname -- "$data_file")/test.csv}"
if (( $# > 0 )); then shift; fi

case "$model" in
    lightgcn) batch_size=2048; learning_rate=0.01 ;;
    sasrec|bert4rec) batch_size=256; learning_rate=0.001 ;;
    *) printf 'Unknown model: %s (see --help)\n' "$model" >&2; exit 2 ;;
esac
[[ -f "$data_file" ]] || { printf 'Training input not found: %s\n' "$data_file" >&2; exit 1; }
[[ -f "$test_file" ]] || { printf 'Export input not found: %s\n' "$test_file" >&2; exit 1; }
printf 'Training model=%s input=%s export_rows=%s topk=20\n' "$model" "$data_file" "$test_file"
exec "${PYTHON:-python}" -m recommender.run --model "$model" --data "$data_file" \
    --test_data "$test_file" --epochs 200 --batch_size "$batch_size" \
    --lr "$learning_rate" --export_topk 20 "$@"
