# Makefile for "The Deliberation Gradient" paper
# Usage: make pdf

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: pdf clean view

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,synctex.gz,fls,fdb_latexmk}
	rm -f sections/*.aux
	rm -f *.run.xml *.bcf

view: pdf
	open $(MAIN).pdf  # macOS; use xdg-open on Linux
