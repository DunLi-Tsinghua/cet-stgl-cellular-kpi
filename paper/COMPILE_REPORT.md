# COMPILE_REPORT

## Compile Commands

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Status

- pdflatex pass 1: PASS
- bibtex: PASS
- pdflatex pass 2: PASS
- pdflatex pass 3: PASS
- Output PDF: `main.pdf`
- Output pages: 11
- Output size: 429489 bytes
- BibTeX entries in `refs.bib`: 24

## Final Log Summary

- LaTeX warnings: 0
- Overfull boxes: 0
- Undefined references: 0
- Undefined citations: 0
- BibTeX errors: 0
- Duplicate destination warnings: 0
- Underfull hbox/vbox messages: 7

## Notes

- Remaining underfull messages are minor IEEE two-column line-breaking artifacts from long technical terms and table notes.
- No result numbers were changed to resolve layout issues.
- The appendix is compiled in one-column mode because `longtable` is not supported in IEEEtran two-column mode.
- The target template's author and affiliation metadata are retained.
