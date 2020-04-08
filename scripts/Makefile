.PHONY: all
all: usa uk belgium italy

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
	# source ../venv/bin/activate
	cd ../data/Italy; python italy_prepare_regional_deaths.py

misc:
	./fetch_misc.sh
