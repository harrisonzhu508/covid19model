from copy import deepcopy

import pandas as pd

regional_medical_data = pd.read_csv('regional_medical_data.csv')
mobility_data = pd.read_csv('google_mobility_covariates.csv')
province_name_translation = pd.read_csv('province_name_translation.csv')


mobility_data['date'] = pd.to_datetime(mobility_data['date']).dt.date
regional_medical_data['data'] = pd.to_datetime(regional_medical_data['data']).dt.date

regional_medical_data = regional_medical_data.set_index('denominazione_regione').join(province_name_translation.set_index('denominazione_regione')).reset_index()

mobility_data_TST = mobility_data[mobility_data['county'] == "Trentino-South Tyrol"]
mobility_data_Bolzano = deepcopy(mobility_data_TST)
mobility_data_Trento = deepcopy(mobility_data_TST)

mobility_data_Bolzano['county'] = "Bolzano"
mobility_data_Trento['county'] = "Trento"

mobility_data = mobility_data.append(mobility_data_Bolzano)
mobility_data = mobility_data.append(mobility_data_Trento)
mobility_data = mobility_data.reset_index()
mobility_data = mobility_data.set_index('county').join(province_name_translation.set_index('county'))
mobility_data = mobility_data.reset_index()
mobility_data.drop(mobility_data[mobility_data['county'] == "Trentino-South Tyrol"].index, inplace=True)
# mobility_data.drop(mobility_data[mobility_data['county'] == "Overall"].index, inplace=True)

mobility_data.sort_values(by=['county', 'date'], inplace=True)
mobility_data.reset_index()
mobility_data = mobility_data[['state', 'county', 'denominazione_regione','google_county','date','grocery/pharmacy','parks','residential','retail/recreation','transitstations','workplace']]


# Need to extend this data in BOTH directions to make it usable in the algorithm.
# BIG assumptions here. In the backwards direction, we are going to assume there was no
# impact on "typical" behaviour. This could either be by setting all dates prior to the
# start as 0% change, or by extrapolating an average of the first few days backwards.
# CHOSEN: set to 0%
# In the forwards direction, could either propagate an average forwards, or copy the previous 
# week forwards.
# CHOSEN: copy the week forwards

dates = mobility_data.date.unique()
first_day = pd.to_datetime(dates.min())
last_day = pd.to_datetime(dates.max())
prev_dates = pd.date_range(
    first_day - pd.to_timedelta(2 * 31, unit='d'), 
    first_day - pd.to_timedelta(1, unit='d'),
    )
last_week = pd.date_range(
    last_day - pd.to_timedelta(6, unit='d'),
    last_day
    )
max_medical_date = regional_medical_data['data'].max() 

for county in mobility_data.county.unique():
    state = mobility_data[mobility_data['county'] == county]['state'].unique()[0]
    google_county = mobility_data[mobility_data['county'] == county]['google_county'].unique()[0]
    denominazione_regione = mobility_data[mobility_data['county'] == county]['denominazione_regione'].unique()[0]

    # Add a 2 months at the start of 'normal' behaviour
    data = {
        'state' : 'IT',
        'county' : county,
        'denominazione_regione' : denominazione_regione,
        'google_county' : google_county,
        'date' : prev_dates,
        'grocery/pharmacy' : 0.0,
        'parks' : 0.0,
        'residential' : 0.0,
        'retail/recreation' : 0.0,
        'transitstations' : 0.0,
        'workplace' : 0.0,
    }
    new_data = pd.DataFrame(data)
    mobility_data = mobility_data.append(new_data)

    # add 3 weeks repeated to the end
    prev_week = mobility_data[
            (mobility_data['county'] == county) &
            (mobility_data['date'].isin(last_week))
        ]
    for i in range(1,4):
        next_week = prev_week.copy()
        next_week['date'] = prev_week['date'] + pd.to_timedelta(7 * i, unit='d')
        mobility_data = mobility_data.append(next_week)

mobility_data.sort_values(by=['county', 'date'], inplace=True)
mobility_data.reset_index()
mobility_data = mobility_data[['state', 'county', 'denominazione_regione','google_county','date','grocery/pharmacy','parks','residential','retail/recreation','transitstations','workplace']]

# remove data past the end of the medical data
mobility_data = mobility_data.reset_index(drop=True)
mobility_data.drop(mobility_data[mobility_data['date'] > max_medical_date].index, inplace=True)

mobility_data = mobility_data[['state', 'county', 'denominazione_regione','google_county','date','grocery/pharmacy','parks','residential','retail/recreation','transitstations','workplace']]
mobility_data.fillna(method='ffill', inplace=True)
mobility_data['date'] = pd.to_datetime(mobility_data['date']).dt.date

mobility_data.reset_index(drop=True).to_csv('google_mobility_covariates_processed.csv', index=False)

regional_medical_data['state'] = 'IT'
regional_medical_data = regional_medical_data[['state', 'county', 'google_county', 'denominazione_regione','data', 'totale_casi', 'deceduti']]
regional_medical_data.columns = ['state', 'county', 'google_county', 'denominazione_regione','date', 'total_positive_cases', 'deaths']

regional_medical_data.sort_values(by=['county', 'date'], inplace=True)

regional_medical_data['daily_deaths'] = regional_medical_data['deaths'].copy()
regional_medical_data['daily_cases'] = regional_medical_data['total_positive_cases'].copy()

for county in regional_medical_data.county.unique():
    index = regional_medical_data[regional_medical_data['county'] == county].index

    regional_medical_data.loc[index[1:], 'daily_deaths'] = regional_medical_data.loc[index[1:]]['deaths'].values - regional_medical_data.loc[index[:-1]]['deaths'].values
    regional_medical_data.loc[index[1:], 'daily_cases'] = regional_medical_data.loc[index[1:]]['total_positive_cases'].values - regional_medical_data.loc[index[:-1]]['total_positive_cases'].values

dates = regional_medical_data.date.unique()
first_day = pd.to_datetime(dates.min())
prev_dates = pd.date_range(
    first_day - pd.to_timedelta(31, unit='d'), 
    first_day - pd.to_timedelta(1, unit='d'),
    ).date

for county in regional_medical_data.county.unique():
    state = regional_medical_data[regional_medical_data['county'] == county]['state'].unique()[0]
    google_county = regional_medical_data[regional_medical_data['county'] == county]['google_county'].unique()[0]
    denominazione_regione = regional_medical_data[regional_medical_data['county'] == county]['denominazione_regione'].unique()[0]

    # Add a month at the start of no deaths and cases
    data = {
        'state' : 'IT',
        'county' : county,
        'denominazione_regione' : denominazione_regione,
        'google_county' : google_county,
        'date' : prev_dates,
        'deaths' : 0.0,
        'daily_deaths' : 0.0,
        'total_positive_cases' : 0.0,
        'daily_cases' : 0.0
    }

    new_data = pd.DataFrame(data)
    regional_medical_data = regional_medical_data.append(new_data)

regional_medical_data.sort_values(by=['county', 'date'], inplace=True)
regional_medical_data.reset_index()
regional_medical_data = regional_medical_data[['state', 'county', 'denominazione_regione','google_county','date','deaths','daily_deaths','total_positive_cases','daily_cases']]
regional_medical_data['date'] = pd.to_datetime(regional_medical_data['date']).dt.date

regional_medical_data.reset_index(drop=True).to_csv('regional_medical_data_processed.csv', index=False)