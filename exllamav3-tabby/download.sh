#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec python3 "$RECIPE_ROOT/tools/recipe.py" download "$@"
