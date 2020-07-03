import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pymc3 as pm
import arviz as az
import numpy as np
import pandas as pd
from scipy import stats as sps

import theano
import theano.tensor as tt
from theano.tensor.signal.conv import conv2d

from covid.patients import get_delay_distribution


class GenerativeModel:
    version = "1.0.0"

    def __init__(self, region: str, observed: pd.DataFrame, buffer_days=10):
        """ Takes a region (ie State) name and observed new positive and
            total test counts per day. buffer_days is the default number of
            blank days we pad on the leading edge of the time series because
            infections occur long before reports and we need to infer values
            on those days """

        first_index = observed.positive.ne(0).argmax()
        observed = observed.iloc[first_index:]
        new_index = pd.date_range(
            start=observed.index[0] - pd.Timedelta(days=buffer_days),
            end=observed.index[-1],
            freq="D",
        )
        observed = observed.reindex(new_index, fill_value=0)

        self._trace = None
        self._inference_data = None
        self.model = None
        self.observed = observed
        self.region = region

    @property
    def n_divergences(self):
        """ Returns the number of divergences from the current trace """
        assert self.trace != None, "Must run sample() first!"
        return self.trace["diverging"].nonzero()[0].size

    @property
    def inference_data(self):
        """ Returns an Arviz InferenceData object """
        assert self.trace, "Must run sample() first!"

        with self.model:
            posterior_predictive = pm.sample_posterior_predictive(self.trace)

        _inference_data = az.from_pymc3(
            trace=self.trace,
            posterior_predictive=posterior_predictive,
        )
        _inference_data.posterior.attrs["model_version"] = self.version

        return _inference_data

    @property
    def trace(self):
        """ Returns the trace from a sample() call. """
        assert self._trace, "Must run sample() first!"
        return self._trace

    def _scale_to_positives(self, data):
        """ Scales a time series to have the same mean as the observed positives
            time series. This is useful because many of the series we infer are
            relative to their true values so we make them comparable by putting
            them on the same scale. """
        scale_factor = self.observed.positive.mean() / np.mean(data)
        return scale_factor * data

    def _get_generation_time_interval(self):
        """ Create a discrete P(Generation Interval)
            Source: https://www.ijidonline.com/article/S1201-9712(20)30119-3/pdf """
        mean_si = 4.7
        std_si = 2.9
        mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))
        sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))
        dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)

        # Discretize the Generation Interval up to 20 days max
        g_range = np.arange(0, 20)
        gt = pd.Series(dist.cdf(g_range), index=g_range)
        gt = gt.diff().fillna(0)
        gt /= gt.sum()
        gt = gt.values
        return gt

    def _get_convolution_ready_gt(self, len_observed):
        """ Speeds up theano.scan by pre-computing the generation time interval
            vector. Thank you to Junpeng Lao for this optimization.
            Please see the outbreak simulation math here:
            https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html """
        gt = self._get_generation_time_interval()
        convolution_ready_gt = np.zeros((len_observed - 1, len_observed))
        for t in range(1, len_observed):
            begin = np.maximum(0, t - len(gt) + 1)
            slice_update = gt[1 : t - begin + 1][::-1]
            convolution_ready_gt[
                t - 1, begin : begin + len(slice_update)
            ] = slice_update
        convolution_ready_gt = theano.shared(convolution_ready_gt)
        return convolution_ready_gt

    def build(self):
        """ Builds and returns the Generative model. Also sets self.model """

        p_delay = get_delay_distribution()
        nonzero_days = self.observed.total.gt(0)
        len_observed = len(self.observed)
        convolution_ready_gt = self._get_convolution_ready_gt(len_observed)
        x = np.arange(len_observed)[:, None]

        coords = {
            "date": self.observed.index.values,
            "nonzero_date": self.observed.index.values[self.observed.total.gt(0)],
        }
        with pm.Model(coords=coords) as self.model:

            # Let log_r_t walk randomly with a fixed prior of ~0.035. Think
            # of this number as how quickly r_t can react.
            log_r_t = pm.GaussianRandomWalk(
                "log_r_t",
                sigma=0.035,
                dims=["date"]
            )
            r_t = pm.Deterministic("r_t", pm.math.exp(log_r_t), dims=["date"])

            # For a given seed population and R_t curve, we calculate the
            # implied infection curve by simulating an outbreak. While this may
            # look daunting, it's simply a way to recreate the outbreak
            # simulation math inside the model:
            # https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html
            seed = pm.Exponential("seed", 1 / 0.02)
            y0 = tt.zeros(len_observed)
            y0 = tt.set_subtensor(y0[0], seed)
            outputs, _ = theano.scan(
                fn=lambda t, gt, y, r_t: tt.set_subtensor(y[t], tt.sum(r_t * y * gt)),
                sequences=[tt.arange(1, len_observed), convolution_ready_gt],
                outputs_info=y0,
                non_sequences=r_t,
                n_steps=len_observed - 1,
            )
            infections = pm.Deterministic("infections", outputs[-1], dims=["date"])

            # Convolve infections to confirmed positive reports based on a known
            # p_delay distribution. See patients.py for details on how we calculate
            # this distribution.
            test_adjusted_positive = pm.Deterministic(
                "test_adjusted_positive",
                conv2d(
                    tt.reshape(infections, (1, len_observed)),
                    tt.reshape(p_delay, (1, len(p_delay))),
                    border_mode="full",
                )[0, :len_observed],
                dims=["date"]
            )

            # Picking an exposure with a prior that exposure never goes below
            # 0.1 * max_tests. The 0.1 only affects early values of Rt when
            # testing was minimal or when data errors cause underreporting
            # of tests.
            tests = pm.Data("tests", self.observed.total.values, dims=["date"])
            exposure = pm.Deterministic(
                "exposure",
                pm.math.clip(tests, self.observed.total.max() * 0.1, 1e9),
                dims=["date"]
            )

            # Test-volume adjust reported cases based on an assumed exposure
            # Note: this is similar to the exposure parameter in a Poisson
            # regression.
            positive = pm.Deterministic(
                "positive", exposure * test_adjusted_positive,
                dims=["date"]
            )

            # Save data as part of trace so we can access in inference_data
            observed_positive = pm.Data("observed_positive", self.observed.positive.values, dims=["date"])
            nonzero_observed_positive = pm.Data("nonzero_observed_positive", self.observed.positive[nonzero_days.values].values, dims=["nonzero_date"])

            positive_nonzero = pm.NegativeBinomial(
                "nonzero_positive",
                mu=positive[nonzero_days.values],
                alpha=pm.Gamma("alpha", mu=6, sigma=1),
                observed=nonzero_observed_positive,
                dims=["nonzero_date"]
            )

        return self.model

    def sample(
        self,
        cores=4,
        chains=4,
        tune=700,
        draws=200,
        target_accept=0.95,
        init="jitter+adapt_diag",
    ):
        """ Runs the PyMC3 model and stores the trace result in self.trace """

        if self.model is None:
            self.build()

        with self.model:
            self._trace = pm.sample(
                draws=draws,
                cores=cores,
                chains=chains,
                target_accept=target_accept,
                tune=tune,
                init=init,
            )

        return self
