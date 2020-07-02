import numpy
import pandas
import pytest

import arviz

import covid.data
import covid.models.generative


class TestDataUS:
    def test_get_raw(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        assert isinstance(df_raw, pandas.DataFrame)

    def test_process(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(df_raw, pandas.Timestamp('2020-06-25'))
        assert isinstance(df_processed, pandas.DataFrame)


class TestGenerative:
    def test_build(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(df_raw, pandas.Timestamp('2020-06-25'))
        model = covid.models.generative.GenerativeModel(
            region='NY',
            observed=df_processed.xs('NY')
        )
        model.build()
        assert "date" in model.coords

    def test_sample_and_idata(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(df_raw, pandas.Timestamp('2020-06-25'))
        model = covid.models.generative.GenerativeModel(
            region='NY',
            observed=df_processed.xs('NY')
        )
        model.build()
        model.sample(
            cores=1, chains=2, tune=5, draws=7
        )
        assert model.trace is not None
        idata = model.inference_data
        assert isinstance(idata, arviz.InferenceData)
        assert idata.posterior.attrs["model_version"] == model.version
        assert "date" in idata.posterior.coords
