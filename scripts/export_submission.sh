#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist"

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing dependency: $name" >&2
    return 1
  fi
}

require_cmd pandoc
mkdir -p "$OUT_DIR"

convert_doc() {
  local in_file="$1"
  local out_base="$2"

  local in_path="$ROOT_DIR/$in_file"
  if [[ ! -f "$in_path" ]]; then
    echo "Skip (not found): $in_file" >&2
    return 0
  fi

  echo "Exporting: $in_file"

  pandoc "$in_path" \
    --from gfm \
    --toc \
    --standalone \
    -o "$OUT_DIR/${out_base}.docx"

  if command -v wkhtmltopdf >/dev/null 2>&1; then
    pandoc "$in_path" \
      --from gfm \
      --toc \
      --standalone \
      --pdf-engine=wkhtmltopdf \
      -o "$OUT_DIR/${out_base}.pdf" || true
  fi
}

convert_doc "BUILDATHON_SUBMISSION.md" "Coverage_Concierge_Proposal"
convert_doc "PROJECT_BRIEF.md" "Coverage_Concierge_Project_Brief"
convert_doc "PITCH_90_SECONDS.md" "Coverage_Concierge_Pitch_90s"

echo

echo "Done. Outputs are in: $OUT_DIR"
echo "If you want PDF export, install wkhtmltopdf or use Word/Google Docs to save the DOCX as PDF."