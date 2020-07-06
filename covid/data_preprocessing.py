import datetime
import logging
import pandas
import pathlib
import typing
import numpy

import fbprophet
import holidays

_log = logging.getLogger(__file__)


def get_holidays(
    country: str, region: str, years: typing.Sequence[int]
) -> typing.Dict[datetime.datetime, str]:
    """ Retrieve a dictionary of holidays in the region.

    Parameters
    ----------
    country : str
        name or short code of country (as used by https://github.com/dr-prodigy/python-holidays)
    region : str
        name or short code of province of state
    years : list
        years to get holidays for
    
    Returns
    -------
    holidays : dict
        datetime as keys, name of holiday as value
    """
    if not hasattr(holidays, country):
        raise KeyError(f'Country "{country}" was not found in the `holidays` package.')
    country_cls = getattr(holidays, country)
    is_province = region in country_cls.PROVINCES
    is_state = hasattr(country_cls, "STATES") and region in country_cls.STATES
    if is_province:
        return country_cls(years=years, prov=region)
    elif is_state:
        return country_cls(years=years, state=region)
    else:
        raise KeyError(f'Region "{region}" not found in {country} states or provinces.')


def predict_testcounts(
    testcounts: pandas.Series,
    *,
    country: str,
    region: str,
    keep_data: bool,
    predict_after: typing.Optional[
        typing.Union[datetime.datetime, pandas.Timestamp, str]
    ] = None,
    **kwargs,
) -> typing.Tuple[pandas.Series, fbprophet.Prophet, pandas.DataFrame]:
    """ Predict/smooth missing testcounts with Prophet.

    Parameters
    ----------
    observed : pandas.Series
        date-indexed series of observed testcounts
    country : str
        name or short code of country (as used by https://github.com/dr-prodigy/python-holidays)
    region : str
        name or short code of province of state
    keep_data : bool
        if True, existing entries are kept
        if False, existing entries are also predicted, resulting in a smoothed profile
    predict_after : timestamp
        all dates before this are ignored
        Use this argument to prevent an unrealistic upwards trend due to initial testing ramp-up
    **kwargs
        optional kwargs for the `fbprophet.Prophet`. For example:
        * growth: 'linear' or 'logistic' (default)
        * seasonality_mode: 'additive' or 'multiplicative' (default)
    
    Returns
    -------
    result : pandas.Series
        the date-indexed series of smoothed/predicted testcounts
    m : fbprophet.Prophet
        the phophet model
    forecast : pandas.DataFrame
        contains the model prediction
    holidays : dict
        dictionary of the holidays that were used in the model
    """
    if not predict_after:
        predict_after = testcounts.index[0]

    mask_fit = testcounts.index > predict_after
    if keep_data:
        mask_predict = numpy.logical_and(
            testcounts.index > predict_after, numpy.isnan(testcounts.values)
        )
    else:
        mask_predict = testcounts.index > predict_after

    relevant_holidays = get_holidays(
        country,
        region,
        years=set([testcounts.index[0].year, testcounts.index[-1].year]),
    )

    holiday_df = (
        pandas.DataFrame.from_dict(
            relevant_holidays, orient="index", columns=["holiday"]
        )
        .reset_index()
        .rename(columns={"index": "ds"})
    )

    # Config settings of forecast model
    days = (testcounts.index[-1] - testcounts.index[0]).days
    prophet_kwargs = dict(
        growth="logistic",
        seasonality_mode="multiplicative",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        holidays=holiday_df,
        # restrict number of potential changepoints:
        n_changepoints=int(numpy.ceil(days / 30)),
    )
    # override defaults with user-specified kwargs
    prophet_kwargs.update(kwargs)
    m = fbprophet.Prophet(**prophet_kwargs)

    # fit only the selected subset of the data
    df_fit = (
        testcounts.loc[mask_fit]
        .reset_index()
        .rename(columns={"date": "ds", "total": "y"})
    )

    if prophet_kwargs["growth"] == "logistic":
        cap = numpy.max(testcounts) * 1
        df_fit["floor"] = 0
        df_fit["cap"] = cap
    m.fit(df_fit)

    # predict for all dates in the input
    df_predict = testcounts.reset_index().rename(columns={"date": "ds"})
    if prophet_kwargs["growth"] == "logistic":
        df_predict["floor"] = 0
        df_predict["cap"] = cap
    forecast = m.predict(df_predict)

    # make a series of the result that has the same index as the input
    result = pandas.Series(index=testcounts.index, data=testcounts.copy().values)
    result.loc[mask_predict] = numpy.clip(
        forecast.set_index("ds").yhat, 0, forecast.yhat.max()
    )
    # full-length result series, model and forecast are returned
    return result, m, forecast, relevant_holidays
