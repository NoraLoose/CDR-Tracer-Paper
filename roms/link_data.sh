#!/bin/bash
# Link the Pacific ROMS run directories into this folder, so the notebooks'
# relative paths (roms/exp0/..., roms_marbl_dic/expCTRL/..., INPUT/...) resolve.
#
# Usage: ./link_data.sh [PACIFIC_ROOT]
# PACIFIC_ROOT is the directory holding roms/, roms_marbl_alk/, roms_marbl_dic/
# and INPUT/ (the layout of CDR-Tracer-ROMS/paper/pacific, with model output).
set -euo pipefail

PACIFIC_ROOT=${1:-/global/cfs/cdirs/m4746/Users/nora/DeficitTracer/experiments/pacific}
cd "$(dirname "$0")"

for d in roms roms_marbl_alk roms_marbl_dic INPUT; do
    if [ ! -d "$PACIFIC_ROOT/$d" ]; then
        echo "missing: $PACIFIC_ROOT/$d" >&2
        exit 1
    fi
    ln -sfn "$PACIFIC_ROOT/$d" "$d"
    echo "$d -> $PACIFIC_ROOT/$d"
done
