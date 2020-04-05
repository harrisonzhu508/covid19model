.PHONY: all
all: usa uk belgium italy google_mobility

ecdc:
	cd ..;Rscript data/fetch-ecdc.R

usa:
	./fetch_USA.sh
uk:
	./fetch_UK.sh

belgium:
	./fetch_Belgium.sh

italy:
	./fetch_Italy.sh

google_mobility:
	./fetch_GoogleMobility.sh
	cp parse_google_pdfs.py ../data/GoogleMobility
	cd ..;python scripts/parse_google_pdfs.py
