"""
This module contains all Israel-specific data loading and data cleaning routines.
"""
import requests
import pandas as pd
import numpy as np

idx = pd.IndexSlice


def get_raw_covidtracking_data_il():
    # Get the latest csv file from this link:
    baseurl = 'https://data.gov.il'
    nextapi = '/api/3/action/datastore_search?resource_id=d337959a-020a-4ed3-84f7-fca182292308&limit=100000'
    datafs = []
    while (nextapi):
        url = baseurl + nextapi
        with requests.get(url) as r:
            if (not r.json()['result']['records']):
                break
            datafs.append(pd.DataFrame(r.json()['result']['records']))
            nextapi = r.json()['result']['_links']['next']
    data = pd.concat(datafs)
    return(data)

def process_covidtracking_data_il(data: pd.DataFrame, run_date: pd.Timestamp):
    """ Processes raw COVIDTracking data to be in a form for the GenerativeModel.
        In many cases, we need to correct data errors or obvious outliers.
        Data looks like: 
    _id   test_date cough fever sore_throat shortness_of_breath head_ache corona_result age_60_and_above gender test_indication
0     1  2020-07-12     0     0           0                   0         0         שלילי              Yes   נקבה           Other
1     2  2020-07-12     0     0           0                   0         0         שלילי              Yes   נקבה           Other
2     3  2020-07-12     0     0           0                   0         0         חיובי               No    זכר           Other
"""
    # Remove results which are not positive or negative
    data = data[data['corona_result'] != 'אחר']
    # Keep only the tests that are not due to coming from abroad or known contact
    # This will give a better estimate of the spread
    data = data[data['test_indication'] == 'Other']
    # Translate results from Hebrew (did you have to do that MOH?)
    data['corona_result'].replace({'שלילי' : False, 'חיובי': True}, inplace=True)
    data = data.rename(columns={"test_date": "date"})
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    # Groupby date, count total tests and positive tests
    d1 = data[["date", "corona_result"]].groupby("date").count()
    d1 = d1.rename(columns = {"corona_result" : "total"})
    d2 = data[["date", "corona_result"]].groupby("date").sum()
    d2 = d2.rename(columns = {"corona_result" : "positive"})
    data = pd.concat([d1, d2], axis = 1)
    data['region'] = "Israel"
    # Reset the index    
    data = data.reset_index().set_index(["region", "date"]).sort_index()
    return(data.loc[idx[:, :(run_date - pd.DateOffset(1))], ["positive", "total"]])


def get_and_process_covidtracking_data_il(run_date: pd.Timestamp):
    """ Helper function for getting and processing COVIDTracking data at once """
    data = get_raw_covidtracking_data_il()
    data = process_covidtracking_data_il(data, run_date)
    return (data)
