# Paper build

Tables are generated, not hand-written:

    python analysis/make_tables.py

writes `paper/tables/{results_main,contrast,ladder}.tex` from
`checkpoints/*/results.json`. Rerun it after any new seed or arm lands; never
edit the generated files.

`main.tex` targets the ICASSP author kit (`spconf.sty`, `IEEEbib.bst`), which
is not in this repo. No LaTeX toolchain is installed on this machine, so
compile on Overleaf or install texlive locally.

`refs.bib` is populated from the verified literature sweep. Every entry must
have a retrievable source; do not add a citation that has not been checked.
