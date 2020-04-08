#!/bin/sh
country="Belgium"
iso_code="BE"
mkdir -p ../data/${country}
. ./google_mobility_config.sh

wget --no-check-certificate -O ../data/Belgium/belgium.csv 'https://epistat.sciensano.be/Data/COVID19BE_MORT.csv'
wget --no-check-certificate -O ../data/${country}/google_mobility_covariates.csv https://raw.githubusercontent.com/MJHutchinson/covid19-google-mobility-extractor/master/data/${date}/${iso_code}/${date}_${iso_code}_${iso_code}_covariates.csv 