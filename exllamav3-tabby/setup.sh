#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Intentionally non-mutating: adopt an isolated, preinstalled CUDA Python + runtime checkout.
# Model qualification used a local EXL3 development commit not yet fetchable as a published pin.
exec python3 "$RECIPE_ROOT/tools/recipe.py" setup --adopt "$@"
