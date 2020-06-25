import boto3
from io import BytesIO
import arviz as az
import os
import pandas as pd
import requests
import s3fs
import tempfile
import numpy as np
from covid.storage import (
    get_covidtracking_csv_key,
    get_processed_covidtracking_key,
    get_overall_output_key,
    get_inference_data_key,
    get_state_output_key,
)
from covid.data import (
    get_raw_covidtracking_data,
    process_covidtracking_data,
    summarize_inference_data,
)
from covid.models.generative import GenerativeModel


s3 = boto3.Session(
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY", None),
    aws_secret_access_key=os.environ.get("S3_SECRET_KEY", None),
).resource(service_name="s3", endpoint_url=os.environ.get("S3_URL", None))
fs = s3fs.S3FileSystem(
    anon=False,
    key=os.environ.get("S3_ACCESS_KEY", None),
    secret=os.environ.get("S3_SECRET_KEY", None),
    client_kwargs=dict(endpoint_url=os.environ.get("S3_URL", None)),
)
S3_BUCKET = os.environ.get("RTLIVE_S3_BUCKET")


def task_get_covidtracking_data(run_date: pd.Timestamp):
    """ Cache COVIDTracking daily data to s3 """
    data = get_raw_covidtracking_data()
    key = get_covidtracking_csv_key(run_date)
    with fs.open(f"{S3_BUCKET}/{key}", "w") as file:
        data.to_csv(file)


def task_process_covidtracking_data(run_date: pd.Timestamp):
    """ Process the COVIDTracking daily data into a form that works with our
        model. """
    key = get_covidtracking_csv_key(run_date)
    with fs.open(f"{S3_BUCKET}/{key}") as file:
        data = pd.read_csv(file, parse_dates=["date"])
    data = process_covidtracking_data(data, run_date)
    key = get_processed_covidtracking_key(run_date)
    with fs.open(f"{S3_BUCKET}/{key}", "w") as file:
        data.to_csv(file)


def task_run_model(country: str, region: str, run_date: pd.Timestamp):
    """ Run the Generative model for a given region on a given date, store
        inference data into S3. """
    key = get_processed_covidtracking_key(run_date)
    with fs.open(f"{S3_BUCKET}/{key}") as file:
        df = pd.read_csv(file, index_col=["region", "date"], parse_dates=["date"])

    model_input = df.xs(region)
    gm = GenerativeModel(region, model_input)
    gm.sample()

    inference_data = gm.inference_data

    # Ensure no divergences
    assert (
        gm.n_divergences == 0
    ), f"Model {region} had {gm.n_divergences} divergences, failing."

    # Ensure convergence
    R_HAT_LIMIT = 1.1
    r_hat = az.rhat(inference_data).to_dataframe().fillna(1.0)
    assert r_hat.le(R_HAT_LIMIT).all().all(), f"r_hat exceeded threshold, failing."

    with tempfile.NamedTemporaryFile() as fp:
        inference_data.to_netcdf(fp.name)
        fp.seek(0)
        s3.Bucket(S3_BUCKET).upload_fileobj(
            fp, get_inference_data_key(run_date, region, country=country)
        )
    return {"country": country, "region": region, "r_hat": r_hat}


def task_render_region_result(country: str, region: str, run_date: pd.Timestamp):
    """ Render a CSV with summary output for a given region """
    az.rcParams["data.load"] = "eager"

    with tempfile.NamedTemporaryFile() as fp:
        s3.Bucket(S3_BUCKET).download_file(
            get_inference_data_key(run_date, region, country=country), fp.name
        )
        fp.seek(0)
        inference_data = az.from_netcdf(fp.name)

    summary = summarize_inference_data(inference_data)
    key = get_state_output_key(run_date, region, country=country)
    with fs.open(f"{S3_BUCKET}/{key}", "w") as file:
        summary.to_csv(file)


def task_gather_region_results(country: str, regions: [str], run_date: pd.Timestamp):
    """ Collects all regions results and outputs them as a single file """
    dfs = []
    for region in regions:
        key = get_state_output_key(run_date, region, country=country)
        with fs.open(f"{S3_BUCKET}/{key}") as file:
            df = pd.read_csv(file)
        multi_index = pd.MultiIndex.from_product([[region], df.index])
        multi_index.set_names(["region", "index"], inplace=True)
        df.index = multi_index
        dfs.append(df)
    all_regions = pd.concat(dfs)
    all_key = get_overall_output_key(run_date, country=country)
    with fs.open(f"{S3_BUCKET}/{all_key}", "w") as file:
        all_regions.to_csv(file)
