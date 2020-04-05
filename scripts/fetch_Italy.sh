#!/bin/sh
wget --no-check-certificate -O ../data/Italy/mobility.csv 'https://data.humdata.org/dataset/40a9ea9e-0edb-49f7-a440-6aee3015961b/resource/d5384152-f198-466d-9be8-7f250c4edae9/download/average_network_degree_by_province_20200321_allusers.csv'
wget --no-check-certificate -O ../data/Italy/provinces.geojson 'https://github.com/openpolis/geojson-italy/raw/master/geojson/limits_IT_provinces.geojson'
