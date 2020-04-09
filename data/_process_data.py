from copy import deepcopy

import pandas as pd

interventions = pd.read_csv('interventions.csv')
num_interventions = 6

# Select the rows containing actual data
interventions = interventions.iloc[:11, : num_interventions + 1]

interventions.to_csv('interventions_processed.csv', index=False)