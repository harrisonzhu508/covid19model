from copy import deepcopy

import pandas as pd

regional_medical_data = pd.read_csv('regional_medical_data.csv')
mobility_data = pd.read_csv('google_mobility_covariates.csv')
province_name_translation = pd.read_csv('province_name_translation.csv')

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

mobility_data = mobility_data[['state', 'county', 'denominazione_regione','google_county','date','grocery/pharmacy','parks','residential','retail/recreation','transitstations','workplace']]

mobility_data.reset_index().to_csv('google_mobility_covariates_processed.csv', index=False)

regional_medical_data['state'] = 'IT'
regional_medical_data = regional_medical_data[['state', 'county', 'google_county', 'denominazione_regione','data', 'totale_casi', 'deceduti']]
regional_medical_data.columns = ['state', 'county', 'google_county', 'denominazione_regione','date', 'total_positive_cases', 'deaths']
regional_medical_data['date'] = pd.to_datetime(regional_medical_data['date'], format='%Y-%m-%d').dt.date

regional_medical_data.sort_values(by=['county', 'date'], inplace=True)

regional_medical_data.reset_index().to_csv('regional_medical_data_processed.csv', index=False)