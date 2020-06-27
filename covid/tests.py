import numpy
import pandas
import pytest

import covid.data
import covid.models.generative


class TestDataUS:
    def test_get_raw(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        assert isinstance(df_raw, pandas.DataFrame)

    def test_process(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = process_covidtracking_data(df_raw, pandas.Timestamp('2020-06-25'))
        assert isinstance(df_processed, pandas.DataFrame)
