from covid_tracking_test_suite import TestCovidTrackingBase
import pytest


class TestGenerative(TestCovidTrackingBase):

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_build(self, source_type):

        source_type.generate_model(run_date='2020-06-25', geo='NY')
        source_type.is_model()
        source_type.validate_missing_variables()

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_sample_and_idata(self, source_type):

        sample_params = {"cores": 1, "chains": 2, "tune": 5, "draws": 7}
        source_type.generate_model(run_date='2020-06-25', geo='NY', sample_params=sample_params)
        source_type.is_model()

        source_type.validate_posterior_data()
        source_type.validate_observed_data()
        source_type.validate_constant_data()


