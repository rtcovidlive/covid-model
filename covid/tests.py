import numpy
import pandas
import pytest

import arviz
import pymc3

import covid.data
import covid.models.generative


class TestDataUS:
    def test_get_raw(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        assert isinstance(df_raw, pandas.DataFrame)

    def test_process(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        run_date = pandas.Timestamp('2020-06-25')
        df_processed = covid.data.process_covidtracking_data(df_raw, run_date)
        assert isinstance(df_processed, pandas.DataFrame)
        assert df_processed.index.names == ("region", "date")
        # the last entry in the data is the day before `run_date`!
        assert df_processed.xs('NY').index[-1] < run_date
        assert df_processed.xs('NY').index[-1] == (run_date - pandas.DateOffset(1))
        assert "positive" in df_processed.columns
        assert "total" in df_processed.columns


class TestDataGeneralized:
    def test_get_unsupported(self):
        with pytest.raises(KeyError):
            covid.data.get_data(country="not_a_country", run_date=pandas.Timestamp("2020-06-20"))

    def test_get_us(self):
        import covid.data_us
        assert "us" in covid.data.LOADERS
        run_date = pandas.Timestamp('2020-06-25')
        result = covid.data.get_data("us", run_date)
        assert isinstance(result, pandas.DataFrame)
        assert result.index.names == ("region", "date")
        assert result.xs('NY').index[-1] < run_date
        assert result.xs('NY').index[-1] == (run_date - pandas.DateOffset(1))
        assert "positive" in result.columns
        assert "total" in result.columns


class TestGenerative:
    def test_build(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(df_raw, pandas.Timestamp('2020-06-25'))
        model = covid.models.generative.GenerativeModel(
            region='NY',
            observed=df_processed.xs('NY')
        )
        pmodel = model.build()
        assert isinstance(pmodel, pymc3.Model)
        # important coordinates
        assert "date" in pmodel.coords
        assert "nonzero_date" in pmodel.coords
        # important random variables
        expected_vars = set(['r_t', 'seed', 'infections', 'test_adjusted_positive', 'exposure', 'positive', 'alpha'])
        missing_vars = expected_vars.difference(set(pmodel.named_vars.keys()))
        assert not missing_vars, f'Missing variables: {missing_vars}'

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
        # check posterior
        assert idata.posterior.attrs["model_version"] == model.version
        assert "chain" in idata.posterior.coords
        assert "draw" in idata.posterior.coords
        assert "date" in idata.posterior.coords
        expected_vars = set(["r_t", "seed", "infections", "test_adjusted_positive", "exposure", "positive", "alpha"])
        missing_vars = expected_vars.difference(set(idata.posterior.keys()))
        assert not missing_vars, f'Missing {missing_vars} from posterior group'
        # check observed_data
        assert "nonzero_date" in idata.observed_data.coords
        expected_vars = set(["nonzero_positive"])
        missing_vars = expected_vars.difference(set(idata.observed_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from observed_data group'
        # check constant_data
        assert "date" in idata.constant_data.coords
        assert "nonzero_date" in idata.constant_data.coords
        expected_vars = set(["exposure", "tests", "observed_positive", "nonzero_observed_positive"])
        missing_vars = expected_vars.difference(set(idata.constant_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from constant_data group'
