#!/bin/sh
country="UK"
iso_code="GB"
mkdir -p ../data/${country}
. ./google_mobility_config.sh

wget --no-check-certificate -O ../data/UK/uk.xlsx 'https://fingertips.phe.org.uk/documents/Historic%20COVID-19%20Dashboard%20Data.xlsx'
wget --no-check-certificate -O ../data/${country}/google_mobility_covariates.csv https://raw.githubusercontent.com/MJHutchinson/covid19-google-mobility-extractor/master/data/${date}/${iso_code}/${date}_${iso_code}_${iso_code}_covariates.csv 