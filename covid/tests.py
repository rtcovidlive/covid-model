import datetime
import numpy
import pandas
import pytest

import arviz
import fbprophet
import pymc3

import covid.data
import covid.data_preprocessing
import covid.models.generative


class TestDataUS:
    def test_get_raw(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        assert isinstance(df_raw, pandas.DataFrame)

    def test_process(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        run_date = pandas.Timestamp("2020-06-25")
        df_processed = covid.data.process_covidtracking_data(df_raw, run_date)
        assert isinstance(df_processed, pandas.DataFrame)
        assert df_processed.index.names == ("region", "date")
        # the last entry in the data is the day before `run_date`!
        assert df_processed.xs("NY").index[-1] < run_date
        assert df_processed.xs("NY").index[-1] == (run_date - pandas.DateOffset(1))
        assert "positive" in df_processed.columns
        assert "total" in df_processed.columns


class TestDataGeneralized:
    def test_get_unsupported(self):
        with pytest.raises(KeyError):
            covid.data.get_data(
                country="not_a_country", run_date=pandas.Timestamp("2020-06-20")
            )

    def test_get_us(self):
        import covid.data_us

        assert "us" in covid.data.LOADERS
        run_date = pandas.Timestamp("2020-06-25")
        result = covid.data.get_data("us", run_date)
        assert isinstance(result, pandas.DataFrame)
        assert result.index.names == ("region", "date")
        assert result.xs("NY").index[-1] < run_date
        assert result.xs("NY").index[-1] == (run_date - pandas.DateOffset(1))
        assert "positive" in result.columns
        assert "total" in result.columns


class TestGenerative:
    def test_build(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(
            df_raw, pandas.Timestamp("2020-06-25")
        )
        model = covid.models.generative.GenerativeModel(
            region="NY", observed=df_processed.xs("NY")
        )
        pmodel = model.build()
        assert isinstance(pmodel, pymc3.Model)
        # important coordinates
        assert "date" in pmodel.coords
        assert "nonzero_date" in pmodel.coords
        # important random variables
        expected_vars = set(
            [
                "r_t",
                "seed",
                "infections",
                "test_adjusted_positive",
                "exposure",
                "positive",
                "alpha",
            ]
        )
        missing_vars = expected_vars.difference(set(pmodel.named_vars.keys()))
        assert not missing_vars, f"Missing variables: {missing_vars}"

    def test_sample_and_idata(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(
            df_raw, pandas.Timestamp("2020-06-25")
        )
        model = covid.models.generative.GenerativeModel(
            region="NY", observed=df_processed.xs("NY")
        )
        model.build()
        model.sample(cores=1, chains=2, tune=5, draws=7)
        assert model.trace is not None
        idata = model.inference_data
        assert isinstance(idata, arviz.InferenceData)
        # check posterior
        assert idata.posterior.attrs["model_version"] == model.version
        assert "chain" in idata.posterior.coords
        assert "draw" in idata.posterior.coords
        assert "date" in idata.posterior.coords
        expected_vars = set(
            [
                "r_t",
                "seed",
                "infections",
                "test_adjusted_positive",
                "exposure",
                "positive",
                "alpha",
            ]
        )
        missing_vars = expected_vars.difference(set(idata.posterior.keys()))
        assert not missing_vars, f"Missing {missing_vars} from posterior group"
        # check observed_data
        assert "nonzero_date" in idata.observed_data.coords
        expected_vars = set(["nonzero_positive"])
        missing_vars = expected_vars.difference(set(idata.observed_data.keys()))
        assert not missing_vars, f"Missing {missing_vars} from observed_data group"
        # check constant_data
        assert "date" in idata.constant_data.coords
        assert "nonzero_date" in idata.constant_data.coords
        expected_vars = set(
            ["exposure", "tests", "observed_positive", "nonzero_observed_positive"]
        )
        missing_vars = expected_vars.difference(set(idata.constant_data.keys()))
        assert not missing_vars, f"Missing {missing_vars} from constant_data group"


class TestDataPreprocessing:
    @pytest.mark.parametrize(
        "country,region",
        [("US", "OK"), ("US", None), ("US", []), ("US", ["NY", "CA"]),],
    )
    def test_get_holidays(self, country, region):
        result = covid.data_preprocessing.get_holidays(country, region, years=[2020])
        assert isinstance(result, dict)
        assert len(result) > 0
        assert isinstance(tuple(result.keys())[0], datetime.date)
        assert isinstance(tuple(result.values())[0], str)

    @pytest.mark.parametrize(
        "country,region",
        [("US", "OK"), ("US", None), ("US", []), ("US", ["NY", "CA"]),],
    )
    @pytest.mark.parametrize("keep_data", [False, True])
    def test_predict_testcounts(self, keep_data, country, region):
        true_pattern = numpy.array([150, 150, 150, 300, 300, 10, 10] * 10)
        df_region = pandas.DataFrame(
            index=pandas.date_range(
                "2020-03-01", periods=len(true_pattern), freq="D", name="date"
            ),
            columns=["positive", "total"],
        )
        df_region.positive = numpy.random.randint(0, 1000, size=len(df_region))
        df_region.total = true_pattern + numpy.random.randint(
            -2, 2, size=len(df_region)
        )
        # remove last two weeks of testcounts:
        df_region.loc[df_region.iloc[-14:].index, "total"] = numpy.nan
        # also make a gap in the data
        df_region.loc[df_region.iloc[30:32].index, "total"] = numpy.nan

        # predict the missing last two weeks:
        result, m, forecast, holidays = covid.data_preprocessing.predict_testcounts(
            df_region.total,
            country=country,
            region=region,
            keep_data=keep_data,
            # disable mcmc for tests
            mcmc_samples=0,
            # don't predict the first 10 days
            ignore_before=df_region.index[10],
        )
        assert isinstance(result, pandas.Series)
        assert isinstance(m, fbprophet.Prophet)
        assert isinstance(forecast, pandas.DataFrame)
        assert isinstance(holidays, dict)
        mask_predict = numpy.isnan(df_region.total)
        numpy.testing.assert_array_equal(result.index, df_region.index)
        # first 10 remain untouched
        numpy.testing.assert_array_equal(result.values[:10], df_region.total[:10])

        if keep_data:
            # data was just copied
            numpy.testing.assert_array_equal(
                result.values[~mask_predict], df_region.total.values[~mask_predict]
            )
            # predictions should approximate the pattern
            numpy.testing.assert_allclose(
                result.values[mask_predict], true_pattern[mask_predict], atol=20
            )
        else:
            # everything after the 10th day should approximate the pattern
            numpy.testing.assert_allclose(
                result.values[10:], true_pattern[10:], atol=20
            )
