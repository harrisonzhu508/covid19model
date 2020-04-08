date="2020-03-29"

# US states
# states="Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota Ohio Oklahoma Oregon Pennsylvania Rhode_Island South_Carolina South_Dakota Tennessee Texas Utah Vermont Virginia Washington West_Virginia Wisconsin Wyoming"
states=""
mkdir -p ../data/US
for state in $states ; do
    curl -o -s ../data/US/${state}_google_mobility_covariates.csv https://raw.githubusercontent.com/MJHutchinson/covid19-google-mobility-extractor/master/data/${date}/${state}/${date}_US_${country}_covariates.csv 
done

# Countries
countries="GB IT DE ES FR"

for country in $countries ; do
    mkdir -p ../data/${country}
    curl -o -s ../data/${country}/google_mobility_covariates.csv https://raw.githubusercontent.com/MJHutchinson/covid19-google-mobility-extractor/master/data/${date}/${country}/${date}_${country}_${country}_covariates.csv 
done