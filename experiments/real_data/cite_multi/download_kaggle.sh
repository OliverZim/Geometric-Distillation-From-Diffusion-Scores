#!/usr/bin/env bash
# Download the four Kaggle "Open Problems - Multimodal Single-Cell Integration"
# (NeurIPS 2022) files that GAGA (Table 4 / Appx. H.7) and SF2M use for the
# Cite / Multi leave-one-timepoint-out benchmark. Mirrors GAGA's
# notebooks/flow_matching/{cite_data,multi_data}.ipynb: Cite = train+test
# inputs (RNA), Multi = train targets (RNA). The huge multiome ATAC inputs
# (~50 GB) are NOT needed.
#
# Prerequisites (one time, on kaggle.com):
#   1. Settings -> API -> "Create New Token" -> save as ~/.kaggle/access_token (chmod 600),
#      or "Create Legacy API Key" -> save as ~/.kaggle/kaggle.json. A new-style token
#      saved as kaggle.json is also accepted. No install needed: the Kaggle CLI is run
#      through uvx when it is not already on PATH.
#   2. Open https://www.kaggle.com/competitions/open-problems-multimodal/rules and
#      click "I Understand and Accept" (competition files cannot be downloaded otherwise).
set -euo pipefail
COMP=open-problems-multimodal
OUT="${1:-$(dirname "$0")/data/raw}"
mkdir -p "$OUT"

if command -v kaggle >/dev/null 2>&1; then
  KAGGLE=(kaggle)
elif [[ -x "$HOME/.local/bin/kaggle" ]]; then
  KAGGLE=("$HOME/.local/bin/kaggle")
elif command -v uvx >/dev/null 2>&1; then
  KAGGLE=(uvx kaggle)
else
  echo "ERROR: kaggle CLI not found; install it with 'uv tool install kaggle'." >&2
  exit 1
fi

# The 2.x CLI reads a new-style token from ~/.kaggle/access_token; a bare token
# saved under the legacy filename must be passed through the environment variable.
if [[ ! -f "$HOME/.kaggle/access_token" && -f "$HOME/.kaggle/kaggle.json" ]] &&
  ! grep -q '"username"' "$HOME/.kaggle/kaggle.json"; then
  export KAGGLE_API_TOKEN="$(<"$HOME/.kaggle/kaggle.json")"
fi

FILES=(metadata.csv train_cite_inputs.h5 test_cite_inputs.h5 train_multi_targets.h5)
for f in "${FILES[@]}"; do
  if [[ -s "$OUT/$f" ]]; then echo "have $f"; continue; fi
  echo ">>> $f"
  "${KAGGLE[@]}" competitions download -c "$COMP" -f "$f" -p "$OUT"
  if [[ -f "$OUT/$f.zip" ]]; then unzip -o -q "$OUT/$f.zip" -d "$OUT" && rm "$OUT/$f.zip"; fi
done
ls -lh "$OUT"
