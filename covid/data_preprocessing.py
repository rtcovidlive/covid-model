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
    country: str,
    region: typing.Optional[typing.Union[str, typing.List[str]]],
    years: typing.Sequence[int],
) -> typing.Dict[datetime.datetime, str]:
    """ Retrieve a dictionary of holidays in the region.

    Parameters
    ----------
    country : str
        name or short code of country (as used by https://github.com/dr-prodigy/python-holidays)
    region : optional, [str]
        if None or []: only nation-wide
        if "all": nation-wide and all regionals
        if "CA": nation-wide and those for region "CA"
        if ["CA", "NY", "FL"]: nation-wide and those for all listed regions

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
    use_states = hasattr(country_cls, "STATES")

    if not region:
        region = []
    if region == "all":
        # select all
        regions = country_cls.STATES if use_states else country_cls.PROVINCES
    else:
        regions = numpy.atleast_1d(region)

    result = country_cls(years=years)
    for region in regions:
        is_province = region in country_cls.PROVINCES
        is_state = use_states and region in country_cls.STATES
        if is_province:
            result.update(country_cls(years=years, prov=region))
        elif is_state:
            result.update(country_cls(years=years, state=region))
        else:
            raise KeyError(
                f'Region "{region}" not found in {country} states or provinces.'
            )
    return result


def predict_testcounts(
    testcounts: pandas.Series,
    *,
    country: str,
    region: typing.Optional[typing.Union[str, typing.List[str]]],
    keep_data: bool,
    ignore_before: typing.Optional[
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
    region : optional, [str]
        if None or []: only nation-wide
        if "all": nation-wide and all regionals
        if "CA": nation-wide and those for region "CA"
        if ["CA", "NY", "FL"]: nation-wide and those for all listed regions
    keep_data : bool
        if True, existing entries are kept
        if False, existing entries are also predicted, resulting in a smoothed profile
    ignore_before : timestamp
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
    if not ignore_before:
        ignore_before = testcounts.index[0]

    mask_fit = testcounts.index >= ignore_before
    if keep_data:
        mask_predict = numpy.logical_and(
            testcounts.index >= ignore_before, numpy.isnan(testcounts.values)
        )
    else:
        mask_predict = testcounts.index >= ignore_before

    years = set([testcounts.index[0].year, testcounts.index[-1].year])
    all_holidays = get_holidays(country, region, years=years)
    regions = numpy.atleast_1d(region)

    if region == "all" or len(regions) > 1:
        # distinguish between national/regional holidays
        national_holidays = get_holidays(country, region=None, years=years)

        holiday_df = pandas.DataFrame(
            data=[
                (
                    date,
                    name,
                    "national" if date in national_holidays.keys() else "regional",
                )
                for date, name in all_holidays.items()
            ],
            columns=["ds", "name", "holiday"],
        )
    else:
        # none, or only one region -> no distinction between national/regional holidays
        holiday_df = pandas.DataFrame(
            dict(
                holiday="holiday",
                name=list(all_holidays.values()),
                ds=pandas.to_datetime(list(all_holidays.keys())),
            )
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
        mcmc_samples=300,
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
    return result, m, forecast, all_holidays
