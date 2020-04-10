import os
import json
import yaml
import pandas as pd
import numpy as np
from util import poly, dt_to_dec
from scipy.stats import gamma as gamma_scipy
from numpy.random import gamma as gamma_np
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm


from copy import deepcopy

class HierarchicalCountyDataset:
    """Base Dataset class containing attributes relating to the datasets used for the modelling and methods for data wrangling at county level

        Args:
            - config_dir
            - cases_dir
            - ifr_dir
            - serial_interval_dir
            - interventions_dir
            - num_counties
            - num_covariates
            country
            - N2: number of days including forecast
            - DEBUG: flag for debugging setting


        Attributes:
            - cases
            - serial_interval
            - num_covariates
            - country
            - DEBUG
            - ifr
            - covariate_names
            - covariates
            - random_covariates
    """

    def __init__(
        self,
        config_dir="config/catalog.yml",
        cases_dir="data/COVID-19-up-to-date.csv",
        medical_data_dir="data/Italy/regional_medical_data_processed.csv",
        ifr_dir="data/weighted_fatality.csv",
        serial_interval_dir="data/serial_interval.csv",
        interventions_dir="data/interventions.csv",
        mobility_dir="data/Italy/google_mobility_covariates_processed.csv",
        num_interventions=6,
        # num_covariates=1,
        country="Italy",
        N2=75,
        DEBUG=False,
    ):
        # with open(config_dir, "r") as stream:
        #     # merci https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
        #     try:
        #         config = yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)

        # read in all the datasets
        #read in the cointies in the country
        self.country = country
        # self.counties = config[country]
        # self.num_counties = len(self.counties)

        # read in global cases data
        # self.cases = pd.read_csv(cases_dir, encoding="ISO-8859-1")

        # 
        self.serial_interval = pd.read_csv(serial_interval_dir)

        medical_data = pd.read_csv(medical_data_dir)
        medical_data["date"] = medical_data["date"].apply(pd.to_datetime)

        self.cases = medical_data[['date', 'county', 'daily_cases', 'daily_deaths']]
        self.cases.columns = ['date', 'county', 'cases', 'deaths']

        # read in google mobility covariates
        mobility_covariates = pd.read_csv(mobility_dir)
        mobility_covariates["date"]= mobility_covariates["date"].apply(pd.to_datetime)
        mobility_covariates.drop(mobility_covariates[mobility_covariates.county == "Overall"].index, inplace=True)

        # print(sorted(list(medical_data['county'].unique())))
        # print(sorted(list(mobility_covariates['county'].unique())))

        if not sorted(list(medical_data['county'].unique())) == sorted(list(mobility_covariates['county'].unique())):
            raise AssertionError('Counties in the death data and mobility data do not match')

        self.counties = sorted(list(mobility_covariates['county'].unique()))
        self.num_counties = len(self.counties)

        self.covariates = mobility_covariates.sort_values("date")
        self.covariate_names = list(self.covariates.columns)[6:]
        self.num_covariates = len(self.covariate_names)
        # TODO: add in covariate names
        interventions = pd.read_csv(interventions_dir)
        self.num_interventions = num_interventions
        # whether to use smaller dataset for debugging
        self.DEBUG = DEBUG

        # process the datasets
        # remaing column and the UK in particular
        self.ifr = pd.read_csv(ifr_dir) # TODO: needs to be recomputed using county demographics
        # pick out the covariates for the countries (11 by default, 8 interventions)
        # num_covariates+1 because we need the Country index column too
        interventions = interventions.iloc[:11, : num_interventions + 1]
        interventions = interventions[interventions['Country'] == 'Italy']

        self.intervention_names = list(interventions.columns)[1:]
        # convert the dates to datetime
        for intervention_name in self.intervention_names:
            interventions[intervention_name] = interventions[intervention_name].apply(
                pd.to_datetime, format="%Y-%m-%d"
            )

        # making all covariates that happen after lockdown to have same date as lockdown
        non_lockdown_covariates = self.intervention_names.copy()
        non_lockdown_covariates.remove("lockdown")
        for intervention_name in non_lockdown_covariates:
            ind = interventions[intervention_name] > interventions["lockdown"]
            interventions[intervention_name][ind] = interventions["lockdown"][ind].copy(0)

        self.interventions = interventions


    def get_stan_data(self, N2, N0=6):
        """Returns a dictionary object containing data to be fed into the Stan compiler

        Args:

        N2: number of days including forecast

        """
        stan_data = {}

        # M, number of counties
        stan_data["M"] = self.num_counties
        # p1, numbe rof interventions
        stan_data["p1"] = self.num_interventions
        # p2, number of covariates
        stan_data["p2"] = self.num_covariates
        # TODO: this is hardcoded in base.r, beware
        # N0, TODO: Not sure what this is 
        stan_data["N0"] = N0
        # N2, total number of days to fit and predict.
        stan_data["N2"] = N2
        stan_data["SI"] = self.serial_interval["fit"][:N2]

        # TODO: we will use lists, but we need to be careful of stack memory in the future
        stan_data["EpidemicStart"] = []
        stan_data["N"] = []

        # initialise with number of interventions
        for i in range(1, self.num_interventions+1):
            stan_data["intervention{}".format(i)] = np.zeros((N2, self.num_counties))

        # create feature matrix TODO: think about how to construct feature matrix
        for i in range(1, self.num_covariates+1):
            stan_data["x{}".format(i)] = np.zeros((N2, self.num_counties))

        # store the covariates in a numpy array, initialised
        stan_data["deaths"] = np.ones((N2, self.num_counties)) * (-1)
        stan_data["cases"] = np.zeros((N2, self.num_counties)) * (-1)
        stan_data["f"] = np.zeros((N2, self.num_counties))

        # we will generate the dataset in this county order. Could also use a pandas dataframe, but not necessary in my opinion
        # to make this work with test data as indicated in the notebook Test-counties, just change line 136 from 
        #for county_num, county in tqdm(enumerate(self.counties)):
        #to for county_num, county in tqdm(enumerate(self.counties[:1])):

        for county_num, county in tqdm(enumerate(self.counties)):
            ifr = self.ifr["weighted_fatality"] # [self.ifr["county"] == county]
            cases = self.cases[self.cases["county"] == county]
            cases["date"] = cases["date"].apply(pd.to_datetime, format="%d-%m-%Y")

            covariates = self.covariates[self.covariates["county"] == county]

            cases["t"] = cases["date"].apply(lambda v: dt_to_dec(v))
            cases = cases.sort_values(by="t")
            cases = cases.reset_index()

            covariates["t"] = covariates["date"].apply(lambda v: dt_to_dec(v))
            covariates = covariates.sort_values(by="t")
            covariates = covariates.reset_index()

            # where the first case occurs
            index = cases[(cases["cases"] > 0)].index[0]
            # where the cumulative deaths reaches 10
            index_1 = cases[(cases["deaths"].cumsum() >= 10)].index[0]
            # 30 days before 10th death
            index_2 = index_1 - 30
            
            # TODO: what is the latter?
            print(
                "First non-zero cases is on day {}, and 30 days before 5 days is day {}".format(
                    index, index_2
                )
            )


            # TODO: HERE HARRISON
            if index_2 < 0:
                raise AssertionError(f'Not enough data before start of outbreak for {county}. Consider extending the data')

            date_2 = cases["t"][index_2]
            index_2_covariates = covariates[covariates["t"] == date_2].index[0]


            # # only care about this timeframe
            cases = cases[index_2 : cases.shape[0]]
            covariates = covariates[index_2_covariates : covariates.shape[0]]

            # update Epidemic Start day for each country
            stan_data["EpidemicStart"].append(index_1 + 1 - index_2)
            # turn intervention dates into boolean
            for intervention in self.intervention_names:
                cases[intervention] = (
                    cases["date"] > self.interventions[intervention].values[0]
                ) * 1

            # record dates for cases in the country
            # cases[county] = cases["date"]

            # Hazard estimation
            N = cases.shape[0]
            N_covariates = covariates.shape[0]
            print("{} has {} days of death data".format(county, N))
            print("{} has {} days of mobility data".format(county, N_covariates))

            # number of days to forecast
            forecast = N2 - N

            if forecast < 0:
                raise ValueError(f"Increase N2 to make it work. N2=N, forecast=N2-N. N2={N2}, N={N}")

            # discrete hazard rate from time t=0,...,99
            h = np.zeros(forecast + N)

            if self.DEBUG:
                mean = 18.8
                cv = 0.45

                loc = 1 / cv ** 2
                scale = mean * cv ** 2
                for i in range(len(h)):
                    h[i] = (
                        ifr * gamma_scipy.cdf(i, loc=loc, scale=scale)
                        - ifr * gamma_scipy.cdf(i - 1, loc=loc, scale=scale)
                    ) / (1 - ifr * gamma_scipy.cdf(i - 1, loc=loc, scale=scale))

            else:
                # infection to onset
                mean_1 = 5.1
                cv_1 = 0.86
                loc_1 = 1 / cv_1 ** 2
                scale_1 = mean_1 * cv_1 ** 2
                # onset to death
                mean_2 = 18.8
                cv_2 = 0.45
                loc_2 = 1 / cv_2 ** 2
                scale_2 = mean_2 * cv_2 ** 2
                # assume that IFR is probability of dying given infection
                x1 = gamma_np(shape=loc_1, scale=scale_1, size=int(5e6))
                # infection-to-onset ----> do all people who are infected get to onset?
                x2 = gamma_np(shape=loc_2, scale=scale_2, size=int(5e6))

                # CDF of sum of 2 gamma distributions
                gamma_cdf = ECDF(x1 + x2)

                # probability distribution of the infection-to-death distribution \pi_m in the paper
                def convolution(u):
                    return (ifr * gamma_cdf(u))[0]

                h[0] = convolution(1.5) - convolution(0)

                for i in range(1, len(h)):
                    h[i] = (convolution(i + 0.5) - convolution(i - 0.5)) / (
                        1 - convolution(i - 0.5)
                    )

            # TODO: Check these quantities via tests
            s = np.zeros(N2)
            s[0] = 1
            for i in range(1, N2):
                s[i] = s[i - 1] * (1 - h[i - 1])

            # slot in these values
            stan_data["N"].append(N)
            stan_data["f"][:, county_num] = h * s
            # stan_data["y"].append(cases["cases"].values[0])
            stan_data["deaths"][:N, county_num] = cases["deaths"]
            stan_data["cases"][:N, county_num] = cases["cases"]

            intervention2 = np.zeros((N2, self.num_interventions))
            intervention2[:N, :] = cases[self.intervention_names].values
            intervention2[N:N2, :] = intervention2[N - 1, :]
            intervention2 = pd.DataFrame(intervention2, columns=self.intervention_names)


            # WARNING the mobility data is not up to today. Any missing days are extrapolated
            # by taking the mean of the previous 3 days
            covariates2 = np.zeros((N2, self.num_covariates))
            covariates2[:N_covariates, :] = covariates[self.covariate_names].values
            covariates2[N_covariates:N2, :] = np.mean(covariates2[N_covariates-3:N_covariates, :], axis=0)
            covariates2 = pd.DataFrame(covariates2, columns=self.covariate_names)

            for j, intervention in enumerate(self.intervention_names):
                stan_data["intervention{}".format(j+1)][:, county_num] = intervention2[
                    intervention
                ]

            for j, covariate in enumerate(self.covariate_names):
                stan_data["x{}".format(j+1)][:, county_num] = covariates2[covariate].T

        # convert these arrays to integer dtype
        stan_data["cases"] = stan_data["cases"].astype(int)
        stan_data["deaths"] = stan_data["deaths"].astype(int)
        return stan_data


test = HierarchicalCountyDataset().get_stan_data(N2 = 75)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return list(obj)
        else:
            return super(NpEncoder, self).default(obj)

for k in test:
    print(k, type(test[k]))

with open('/Python/src/test.json', 'w') as file:
    json.dump(test, file, cls=NpEncoder, sort_keys=True)