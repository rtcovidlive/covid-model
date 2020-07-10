import typing
import pandas as pd
import arviz as az
import numpy as np

from . data_us import (
    get_and_process_covidtracking_data,
    get_raw_covidtracking_data,
    process_covidtracking_data,
)

# Data loading functions for different countries may be registered here.
# For US, the data loader is pre-registered. Additional countries may be
# registered upon import of third-party modules.
# Data cleaning must be done by the data loader function!
LOADERS:typing.Dict[str, typing.Callable[[pd.Timestamp], pd.DataFrame]] = {
    'us': get_and_process_covidtracking_data,
}


def get_data(country: str, run_date: pd.Timestamp) -> pd.DataFrame:
    """ Retrieves data for a country using the registered data loader method.

    Parameters
    ----------
    country : str
        short code of the country (key in LOADERS dict)
    run_date : pd.Timestamp
        date when the analysis is performed

    Returns
    -------
    model_input : pd.DataFrame
        Data as returned by data loader function.
        Ideally "as it was on `run_date`", meaning that information such as corrections
        that became available after `run_date` should not be taken into account.
        This is important to realistically back-test how the model would have performed at `run_date`.
    """
    if not country in LOADERS:
        raise KeyError(f"No data loader for '{country}' is registered.")
    result = LOADERS[country](run_date)
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ("region", "date")
    assert "positive" in result.columns
    assert "total" in result.columns
    return result


def summarize_inference_data(inference_data: az.InferenceData):
    """ Summarizes an inference_data object into the form that we publish on rt.live """
    posterior = inference_data.posterior
    hdi_mass = 80
    hpdi = az.hdi(posterior.r_t, hdi_prob=hdi_mass / 100).r_t

    observed_positive = inference_data.constant_data.observed_positive.to_series()
    scale_to_positives = lambda data: observed_positive.mean() / np.mean(data) * data
    tests = inference_data.constant_data.tests.to_series()
    normalized_positive = observed_positive / tests.clip(0.1 * tests.max())

    summary = pd.DataFrame(
        data={
            "mean": posterior.r_t.mean(["draw", "chain"]),
            "median": posterior.r_t.median(["chain", "draw"]),
            f"lower_{hdi_mass}": hpdi[:, 0],
            f"upper_{hdi_mass}": hpdi[:, 1],
            "infections": scale_to_positives(
                posterior.infections.mean(["draw", "chain"])
            ),
            "test_adjusted_positive": scale_to_positives(
                posterior.test_adjusted_positive.mean(["draw", "chain"])
            ),
            "test_adjusted_positive_raw": scale_to_positives(normalized_positive),
            "positive": observed_positive,
            "tests": tests,
        },
        index=pd.Index(posterior.date.values, name="date"),
    )
    return summary
