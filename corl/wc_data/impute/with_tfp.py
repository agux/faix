import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


time_series_with_nans = [-1., 1., np.nan, 2.4, np.nan, 5]
observed_time_series = tfp.sts.MaskedTimeSeries(
  time_series=time_series_with_nans,
  is_missing=tf.math.is_nan(time_series_with_nans))


# Build model using observed time series to set heuristic priors.
linear_trend_model = tfp.sts.LocalLinearTrend(
  observed_time_series=observed_time_series)
model = tfp.sts.Sum([linear_trend_model],
                    observed_time_series=observed_time_series)


# Fit model to data
parameter_samples, _ = tfp.sts.fit_with_hmc(model, observed_time_series)


# Impute missing values
imputed_series_distribution = tfp.sts.impute_missing_values(
  model, observed_time_series, parameter_samples, include_observation_noise=True)
print('imputed means and stddevs: ',
      imputed_series_distribution.mean(),
      imputed_series_distribution.stddev(),
      imputed_series_distribution.)