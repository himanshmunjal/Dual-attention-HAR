"""Inline every \\input{...} so paper.tex is a single self-contained file."""
import re
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
text = re.sub(r"\\input\{([^}]+)\}", lambda m: Path(m.group(1) + ".tex").read_text().rstrip("\n"), text)
dst.write_text(text)
