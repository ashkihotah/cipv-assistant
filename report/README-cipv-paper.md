This folder contains a research paper draft for the CIPV Assistant project.

Files:
- `cipv-assistant.tex`: Main paper source following the CEUR-WS `ceurart` class (two columns).
- `cipv-assistant.bib`: Bibliography entries cited in the paper.
- `sample-2col.tex`: Unmodified template reference (kept as requested).

Build (Windows, per included Makefile example uses `sample-2col`):
- To compile this paper, run a LaTeX build on `cipv-assistant.tex` using `lualatex` or `pdflatex` and `bibtex`.
- Example commands:
  1. `lualatex -shell-escape --output-directory=out cipv-assistant.tex`
  2. `bibtex out/cipv-assistant`
  3. `lualatex -shell-escape --output-directory=out cipv-assistant.tex`
  4. `lualatex -shell-escape --output-directory=out cipv-assistant.tex`

If you enable minted code listings, ensure Python and Pygments are available and keep `-shell-escape`.



