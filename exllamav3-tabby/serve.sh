#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ "$(printenv DRY_RUN || :)" = 1 ]; then
  exec python3 "$RECIPE_ROOT/tools/recipe.py" config
fi
ulimit -n 65536 2>/dev/null || true
exec python3 "$RECIPE_ROOT/tools/recipe.py" serve
