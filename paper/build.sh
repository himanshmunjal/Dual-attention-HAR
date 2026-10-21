#!/usr/bin/env bash
# Builds paper.tex (self-contained), paper.pdf and paper.docx from paper_src.tex.
#   bash paper/build.sh            (run from the repo root or from paper/)
set -euo pipefail
cd "$(dirname "$0")"
python3 analysis.py > generated/analysis.log
python3 flatten.py paper_src.tex paper.tex
tectonic -X compile paper.tex --keep-logs --keep-intermediates > generated/tectonic.log 2>&1 || { tail -40 generated/tectonic.log; exit 1; }
python3 make_docx.py
echo "built: paper.tex paper.pdf paper.docx"
