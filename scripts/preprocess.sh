#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
    cat <<'USAGE'
Usage: bash scripts/preprocess.sh [ml-100k|Beauty|Electronics|Yelp2020] [CLI_OPTIONS...]
Defaults to MovieLens-100K only, with 9 negative items per candidate list.
Amazon categories and Yelp are optional and must be selected explicitly.
Set PYTHON to select the Python interpreter. Run from any working directory.
USAGE
    exit 0
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
dataset="${1:-ml-100k}"
if (( $# > 0 )); then shift; fi

case "$dataset" in
    ml-100k) args=(--data_dir data/ml-100k --dataset ml-100k) ;;
    Beauty|Electronics) args=(--data_dir data --dataset amazon --amazon_category "$dataset") ;;
    Yelp2020) args=(--data_dir data/Yelp2020 --dataset yelp2020) ;;
    *) printf 'Unknown dataset: %s (see --help)\n' "$dataset" >&2; exit 2 ;;
esac

printf 'Preprocessing %s only (9 negative items by default).\n' "$dataset"
exec "${PYTHON:-python}" main.py --main Preprocess "${args[@]}" --n_neg_items 9 "$@"
