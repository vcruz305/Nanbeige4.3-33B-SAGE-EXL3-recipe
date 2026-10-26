#!/usr/bin/env bash
# Delegates to the original SAGE campaign, if you have that private checkout.
# This script itself does not rent GPUs, download models, modify files, or schedule jobs.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: SAGE_CAMPAIGN_ROOT=/path/to/your/SAGE-EXL3-nanbeige43-33b-4090 bash quantization/run-stages.sh {help|preflight|status|capture|measure-seed|measure-pilot|measure-finish|compile|reference|encode-qad|gate-qad}" >&2
  exit 2
fi
if [ -z "$(printenv SAGE_CAMPAIGN_ROOT || :)" ]; then
  echo "SAGE_CAMPAIGN_ROOT is required; the private SAGE campaign isn't bundled." >&2
  exit 2
fi
runner="$SAGE_CAMPAIGN_ROOT/recipes/nanbeige43-33b-4090/run.sh"
if [ ! -f "$runner" ]; then
  echo "Campaign runner not found: $runner" >&2
  exit 3
fi
# The pinned campaign runner itself refuses a busy GPU and validates all prerequisites.
exec bash "$runner" "$1"
