mkdir -p ../data/GoogleMobility/
mkdir -p ../data/GoogleMobility/pdfs/
mkdir -p ../data/GoogleMobility/data/

states="Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota Ohio Oklahoma Oregon Pennsylvania Rhode_Island South_Carolina South_Dakota Tennessee Texas Utah Vermont Virginia Washington West_Virginia Wisconsin Wyoming"
date="2020-03-29"
for state in $states ; do
    curl -s -o ../data/GoogleMobility/pdfs/${date}_US_${state}_Mobility_Report_en.pdf https://www.gstatic.com/covid19/mobility/${date}_US_${state}_Mobility_Report_en.pdf
done

# rename the full US pdf slightly to conform to similar naming convention as the individual states
curl -s -o ../data/GoogleMobility/pdfs/${date}_US_US_Mobility_Report_en.pdf https://www.gstatic.com/covid19/mobility/${date}_US_Mobility_Report_en.pdf
    
# # non us countries conform to current naming pattern
countries="GB IT DE ES FR"

for country in $countries ; do
    curl -s -o ../data/GoogleMobility/pdfs/${date}_${country}_${country}_Mobility_Report_en.pdf https://www.gstatic.com/covid19/mobility/${date}_${country}_Mobility_Report_en.pdf
done