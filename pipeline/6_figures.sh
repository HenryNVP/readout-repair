#!/usr/bin/env bash
# Figure 2 (the perturbations, rendered on one Lift scene) and every table from the recorded runs.
source "$(dirname "$0")/common.sh"
python scripts/fig_shifts.py --out figures/fig_shifts.pdf \
  --titles 'turn + roll' 'red, dim + roll' 'upside down' zoomed faint invisible \
  --spec "${SPEC[bothcam]}" --spec "${SPEC[darkredwroll]}" --spec "${SPEC[bothflip]}" \
  --spec "${SPEC[bothzoom]}" --spec "${SPEC[faint]}" --spec "${SPEC[invisible]}"
python scripts/tables.py --table all
