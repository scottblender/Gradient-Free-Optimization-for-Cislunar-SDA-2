from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "scripts" / "plot_study_definition_figures.m"
text = path.read_text(encoding="utf-8")

old = "    stateText = strjoin(compose('%.12g',state),',\\,');"
new = "    latexSeparator = [',' char(92) ','];\n    stateText = strjoin(compose('%.12g',state),latexSeparator);"

if old not in text:
    raise SystemExit("Expected LaTeX state separator was not found")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

# Remove the one-time helper/workflow from the resulting commit.
for rel in [
    ".github/fix_latex_escape.py",
    ".github/workflows/fix-latex-escape.yml",
]:
    p = root / rel
    if p.exists():
        p.unlink()

print("Replaced literal \\, separator with char(92)-constructed LaTeX separator.")
