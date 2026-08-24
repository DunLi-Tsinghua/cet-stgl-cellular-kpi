# Compile Report

## Command

`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## Status

Compilation succeeded and produced `main.pdf`.

## Cross-Reference and Citation Check

- Undefined citations: none in the final log.
- Undefined references: none in the final log.
- Fatal LaTeX errors: none.
- Overfull hboxes: none detected in the final log.

## Remaining Warnings

- Underfull hbox warnings remain, mostly in narrow table/model-name cells and title/frontmatter wrapping.
- Duplicate destination warnings remain for frontmatter/page and appendix longtable anchors; these were present-style PDF anchor warnings and did not break compilation.
- One PDF inclusion warning remains because `figures/Fig_cet_stgl_model_architecture.pdf` is PDF 1.7 while pdfTeX reports a 1.5 inclusion ceiling.

## Notes

The user-made schematic figures `Fig0_cet_stgl_reliability_overview.pdf` and `Fig_cet_stgl_model_architecture.pdf` were not modified. Performance tables were updated to include the new LSTM/TCN neural sequence baseline rows.
