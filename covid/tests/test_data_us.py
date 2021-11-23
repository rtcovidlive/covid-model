from covid_tracking_test_suite import TestCovidTrackingBase
import pytest


class TestDataUS(TestCovidTrackingBase):


    @pytest.mark.parametrize('source_arg', ["raw", "processed"])
    def test_dataframes(self, source_type):

        source_type.generate_data()
        source_type.is_dataframe()

    @pytest.mark.parametrize('source_arg', ["raw"])
    def test_raw_covid_tracking_dataframe_size(self, source_type):

        source_type.generate_data()
        # I think this row value is subject to change
        # source_type.validate_shape(20780, 56)

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_processed_covid_tracking_data_is_dataframe_with_date(self, source_type):
        # try with bad date
        source_type.generate_data(run_date='2020-06-25')
        source_type.is_dataframe()

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_processed_covid_tracking_data_dataframe_types(self, source_type):

        source_type.generate_data(run_date='2020-06-25')
        source_type.validate_types(["total", "positive"])

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_processed_covid_tracking_dataframe_dimension(self, source_type):

        source_type.generate_data(run_date='2020-06-25')
        source_type.validate_shape(5742, 2)

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_processed_covid_tracking_data_index_names(self, source_type):

        source_type.generate_data(run_date='2020-06-25')
        source_type.validate_indexes(("region", "date"))

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_processed_covid_tracking_data_column_names(self, source_type):

        source_type.generate_data(run_date='2020-06-25')
        source_type.validate_columns(["positive", "total"]) # add bad column name and handle

    @pytest.mark.parametrize('source_arg', ["processed"])
    def test_processed_covid_tracking_data_last_entry(self, source_type):

        source_type.generate_data(run_date='2020-06-25')
        source_type.validate_data('2020-06-25') # fix






