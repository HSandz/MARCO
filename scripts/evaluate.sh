#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
    cat <<'USAGE'
Usage: bash scripts/evaluate.sh [DATA_FILE [SYSTEM_CONFIG [TASK]]] [CLI_OPTIONS...]
Defaults: data/ml-100k/test.csv config/systems/marco/basic.json sr
Runs Evaluate on every row of the selected input, with one system configuration.
Example: bash scripts/evaluate.sh data/Beauty/test.csv config/systems/marco/reflector.json sr --topks '[1,3,5]'
Set PYTHON to select the Python interpreter. Relative paths use the repository root.
USAGE
    exit 0
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
data_file="${1:-data/ml-100k/test.csv}"
if (( $# > 0 )); then shift; fi
system_config="${1:-config/systems/marco/basic.json}"
if (( $# > 0 )); then shift; fi
task="${1:-sr}"
if (( $# > 0 )); then shift; fi

[[ -f "$data_file" ]] || { printf 'Input not found: %s\n' "$data_file" >&2; exit 1; }
[[ -f "$system_config" ]] || { printf 'Configuration not found: %s\n' "$system_config" >&2; exit 1; }
printf 'Evaluating input=%s config=%s task=%s\n' "$data_file" "$system_config" "$task"
exec "${PYTHON:-python}" main.py --main Evaluate --data_file "$data_file" \
    --system marco --system_config "$system_config" --task "$task" "$@"
