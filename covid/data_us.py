"""
This module contains all US-specific data loading and data cleaning routines.
"""
import requests
import pandas as pd
import numpy as np

idx = pd.IndexSlice


def get_raw_covidtracking_data():
    """ Gets the current daily CSV from COVIDTracking """
    url = "https://covidtracking.com/api/v1/states/daily.csv"
    data = pd.read_csv(url)
    return data


def process_covidtracking_data(data: pd.DataFrame, run_date: pd.Timestamp):
    """ Processes raw COVIDTracking data to be in a form for the GenerativeModel.
        In many cases, we need to correct data errors or obvious outliers."""
    data = data.rename(columns={"state": "region"})
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
    data = data.set_index(["region", "date"]).sort_index()
    data = data[["positive", "total"]]

    # Too little data or unreliable reporting in the data source.
    data = data.drop(["MP", "GU", "AS", "PR", "VI"])

    # On Jun 5 Covidtracking started counting probable cases too
    # which increases the amount by 5014.
    # https://covidtracking.com/screenshots/MI/MI-20200605-184320.png
    data.loc[idx["MI", pd.Timestamp("2020-06-05") :], "positive"] -= 5014

    # From CT: On June 19th, LDH removed 1666 duplicate and non resident cases
    # after implementing a new de-duplicaton process.
    data.loc[idx["LA", pd.Timestamp("2020-06-19") :], :] += 1666

    # Now work with daily counts
    data = data.diff().dropna().clip(0, None).sort_index()

    # Michigan missed 6/18 totals and lumped them into 6/19 so we've
    # divided the totals in two and equally distributed to both days.
    data.loc[idx["MI", pd.Timestamp("2020-06-18")], "total"] = 14871
    data.loc[idx["MI", pd.Timestamp("2020-06-19")], "total"] = 14871

    # Note that when we set total to zero, the model ignores that date. See
    # the likelihood function in GenerativeModel.build

    # Huge outlier in NJ causing sampling issues.
    data.loc[idx["NJ", pd.Timestamp("2020-05-11")], :] = 0

    # Huge outlier in CA causing sampling issues.
    data.loc[idx["CA", pd.Timestamp("2020-04-22")], :] = 0

    # Huge outlier in CA causing sampling issues.
    # TODO: generally should handle when # tests == # positives and that
    # is not an indication of positive rate.
    data.loc[idx["SC", pd.Timestamp("2020-06-26")], :] = 0

    # Two days of no new data then lumped sum on third day with lack of new total tests
    data.loc[idx["OR", pd.Timestamp("2020-06-26") : pd.Timestamp("2020-06-28")], 'positive'] = 174
    data.loc[idx["OR", pd.Timestamp("2020-06-26") : pd.Timestamp("2020-06-28")], 'total'] = 3296


    #https://twitter.com/OHdeptofhealth/status/1278768987292209154
    data.loc[idx["OH", pd.Timestamp("2020-07-01")], :] = 0
    data.loc[idx["OH", pd.Timestamp("2020-07-09")], :] = 0

    # Nevada didn't report total tests this day
    data.loc[idx["NV", pd.Timestamp("2020-07-02")], :] = 0

    # A bunch of incorrect values for WA data so nulling them out.
    data.loc[idx["WA", pd.Timestamp("2020-06-05") : pd.Timestamp("2020-06-07")], :] = 0
    data.loc[idx["WA", pd.Timestamp("2020-06-20") : pd.Timestamp("2020-06-21")], :] = 0

    # AL reported tests == positives
    data.loc[idx["AL", pd.Timestamp("2020-07-09")], :] = 0

    # Low reported tests
    data.loc[idx["AR", pd.Timestamp("2020-07-10")], :] = 0

    # Positives == tests
    data.loc[idx["MS", pd.Timestamp("2020-07-12")], :] = 0

    # Outlier dates in PA
    data.loc[
        idx[
            "PA",
            [
                pd.Timestamp("2020-06-03"),
                pd.Timestamp("2020-04-21"),
                pd.Timestamp("2020-05-20"),
            ],
        ],
        :,
    ] = 0

    # At the real time of `run_date`, the data for `run_date` is not yet available!
    # Cutting it away is important for backtesting!
    return data.loc[idx[:, :(run_date - pd.DateOffset(1))], ["positive", "total"]]


def get_and_process_covidtracking_data(run_date: pd.Timestamp):
    """ Helper function for getting and processing COVIDTracking data at once """
    data = get_raw_covidtracking_data()
    data = process_covidtracking_data(data, run_date)
    return data
