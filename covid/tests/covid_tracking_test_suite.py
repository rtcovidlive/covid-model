import covid.data
import covid.data_us
import covid.models.generative

import arviz
import numpy as np
import pandas as pd
import pytest
import pymc3


class TestValidationBase:

    """
        Base validation functions to test raw and processed dataframes
    """

    def __init__(self):

        self.df = None
        self.model = None
        self._run_date = None


    @staticmethod
    def validate_loaders(loaders):
        for loader in loaders:
            assert loader in covid.data.LOADERS

    def is_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def is_model(self):
        assert isinstance(self.model, pymc3.Model)

    def validate_shape(self, rows, columns):
        assert self.df.shape[0] == rows
        assert self.df.shape[1] == columns

    def validate_types(self, columns):

        for col in columns:
            # need to handle different types
            assert self.df[col].dtype == np.float64

    def validate_columns(self, columns):
        for col in columns:
            assert col in self.df.columns

    def validate_indexes(self, indexes):
        assert self.df.index.names == indexes


class TestRawData(TestValidationBase):

    """
       Sub-Class that allows user to pull/validate raw data
    """

    def __init__(self):
        super(TestRawData).__init__()
        self.df = None

    def generate_data(self):

        self.df = covid.data.get_raw_covidtracking_data()

class TestProcessedData(TestValidationBase):

    """
        Sub-Class that allows user to pull/validate processed data
    """

    def __init__(self):

        super(TestProcessedData).__init__()
        self.df = None
        self.model = None
        self.idata = None

    def generate_data(self, geo=None, run_date=None):

        if geo:
            self.df = covid.data.get_data(country=geo, run_date=pd.Timestamp(run_date))
        else:
            raw = covid.data.get_raw_covidtracking_data()
            self.df = covid.data.process_covidtracking_data(raw, pd.Timestamp(run_date))

    def generate_model(self, run_date, geo, sample_params=None):

        self.generate_data(run_date=run_date)
        model = covid.models.generative.GenerativeModel(
            region=geo,
            observed=self.df.xs(geo)
        )

        # handle sample default differently
        # if decide to test for empty sample data
        if sample_params:
            model.sample(
                cores=sample_params.get("cores", 1),
                chains=sample_params.get("chains", 2),
                tune=sample_params.get("tune", 5),
                draws=sample_params.get("draws", 7)
            )
            assert model.trace is not None
            self.idata = model.inference_data
            assert self.idata.posterior.attrs["model_version"] == model.version

        self.model = model.build()

    def validate_data(self, run_date):
        run_date = pd.Timestamp(run_date)
        assert self.df.xs('NY').index[-1] < run_date
        assert self.df.xs('NY').index[-1] == (run_date - pd.DateOffset(1))

    def validate_summary(self, columns, indexes):

        # can build out this more to handle summary validation better
        summary = covid.data.summarize_inference_data(self.idata)

        assert list(summary.columns) == columns
        assert list(summary.index.names) == indexes

    def validate_posterior_data(self):

        assert isinstance(self.idata, arviz.InferenceData)
        assert "chain" in self.idata.posterior.coords
        assert "draw" in self.idata.posterior.coords
        assert "date" in self.idata.posterior.coords
        expected_vars = {"r_t", "seed", "infections", "test_adjusted_positive", "exposure", "positive", "alpha"}
        missing_vars = expected_vars.difference(set(self.idata.posterior.keys()))
        assert not missing_vars, f'Missing {missing_vars} from posterior group'

    def validate_observed_data(self):

        assert isinstance(self.idata, arviz.InferenceData)
        assert "nonzero_date" in self.idata.observed_data.coords
        expected_vars = {"nonzero_positive"}
        missing_vars = expected_vars.difference(set(self.idata.observed_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from observed_data group'

    def validate_constant_data(self):

        assert isinstance(self.idata, arviz.InferenceData)
        assert "date" in self.idata.constant_data.coords
        assert "nonzero_date" in self.idata.constant_data.coords
        expected_vars = {"exposure", "tests", "observed_positive", "nonzero_observed_positive"}
        missing_vars = expected_vars.difference(set(self.idata.constant_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from constant_data group'

    def validate_missing_variables(self):

        assert "date" in self.model.coords
        assert "nonzero_date" in self.model.coords

        expected_vars = {'r_t', 'seed', 'infections', 'test_adjusted_positive', 'exposure', 'positive', 'alpha'}
        missing_vars = expected_vars.difference(set(self.model.named_vars.keys()))
        assert not missing_vars, f'Missing variables: {missing_vars}'


class TestCovidTrackingBase:

    """
        An abstraction for test-cases to easily allow user to choose between
        raw or processed data
    """


    @pytest.fixture
    def source_type(self, source_arg):

        # lazy init but figure only raw and processed data exists
        # re-engineer this logic if application grows

        dataframe = pd.DataFrame([])

        if source_arg == "raw":
            dataframe = TestRawData()

        elif source_arg == "processed":
            dataframe = TestProcessedData()

        return dataframe




