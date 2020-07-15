"""
This module contains all France-specific data loading routines.
"""
import requests, json
import pandas as pd
import numpy as np

idx = pd.IndexSlice


def get_raw_covidtracking_data():
    """ Gets the current daily CSV from data.gouv.fr """
    # French official dataset: https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19
    # Limits (translated from above URL, more info in French) :
    # * Data by departements only contains tests for which residence departements of tested people could be known. Hence countrywide data contains more tests than sum of all departements.
    # * Data transmission can sometimes excess 9 days. Indicators are updated daily on test results reception.
    # 
    # We use daily data sorted by 'départements' (finest level available).
    url_by_dep = "https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675"
    data = pd.read_csv(url_by_dep, sep=";")
    return data


def process_covidtracking_data(data: pd.DataFrame, run_date: pd.Timestamp):
    data = data.rename(columns={"jour": "date", "cl_age90": "ageclass", "P": "positive", "T": "total"})

    # Drop data by age class ('0' age class is the sum of all age classes)
    data = data.drop(data[data['ageclass'] != 0].index)
    data = data.drop(columns=["ageclass"])

    # Convert date
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

    data = data.set_index(["dep", "date"]).sort_index()
    data = data[["positive", "total"]]

    return data.loc[idx[:, :(run_date - pd.DateOffset(1))], ["positive", "total"]]
    #return data.loc[idx[[:, :], ["positive", "total"]]


def get_and_process_covidtracking_data(run_date: pd.Timestamp):
    """ Helper function for getting and processing COVIDTracking data at once """
    data = get_raw_covidtracking_data()
    data = process_covidtracking_data(data, run_date)
    return data

