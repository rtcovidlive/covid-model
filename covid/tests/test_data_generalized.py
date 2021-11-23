from covid_tracking_test_suite import TestCovidTrackingBase
import pytest


class TestDataGeneralized(TestCovidTrackingBase):


    @pytest.mark.parametrize('source_arg', ["raw"])
    def test_supported_loaders(self, source_type):

        source_type.validate_loaders(["us"])

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_us_is_supported(self, source_type):

        source_type.generate_data('us', '2020-06-25')
        source_type.is_dataframe()

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_country_not_unsupported(self, source_type):
        with pytest.raises(KeyError):
            source_type.generate_data('not a country', '2020-06-25')

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_supported_summarized_inference(self, source_type):

        columns = ["mean", "median", "lower_80", "upper_80", "infections",
                   "test_adjusted_positive", "test_adjusted_positive_raw",
                   "positive", "tests"]
        indexes = ["date"]
        sample_params = {"cores": 1, "chains": 2, "tune": 5, "draws": 7}

        source_type.generate_model(run_date='2020-06-25', geo='NY', sample_params=sample_params)
        source_type.is_dataframe()
        source_type.validate_summary(columns, indexes)

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_not_supported_summarized_inference_headers(self, source_type):

        columns = ["this header is not in summary","mean", "median", "lower_80", "upper_80", "infections",
                   "test_adjusted_positive", "test_adjusted_positive_raw",
                   "positive", "tests"]
        indexes = ["date"]
        sample_params = {"cores": 1, "chains": 2, "tune": 5, "draws": 7}
        source_type.generate_model(run_date='2020-06-25', geo='NY', sample_params=sample_params)

        with pytest.raises(AssertionError):
            source_type.validate_summary(columns, indexes)







