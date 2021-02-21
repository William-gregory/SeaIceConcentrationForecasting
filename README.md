# Sea Ice Concentration Forecasting
This is follow-on work from forecasting September sea ice extent using Complex Networks and Gaussian Process Regression (see repository: ![SeaIceExtentForecasting](https://github.com/William-gregory/SeaIceExtentForecasting) )

This contains code used to generate spatial forecasts of sea ice concentration in the East Siberian and Laptev seas using multi-task Gaussian Process Regression, in a non-stationary inter-task covariance structure. This code is largely based on previous work by [Paciorek and Schervish, 2006](https://onlinelibrary.wiley.com/doi/pdf/10.1002/env.785) and [Rakitsch et al., 2013](https://proceedings.neurips.cc/paper/2013/file/59c33016884a62116be975a9bb8257e3-Paper.pdf).
<pre>




</pre>
In this example, I have generated a Complex Network based on monthly-mean August sea ice concentration data in the East Siberian and Laptev Seas, which will be used as inputs to the forecast model, as per the method outlined in [Gregory et al., 2020](https://discovery.ucl.ac.uk/id/eprint/10091542/1/Gregory_wafd190107.pdf)

![alt text](https://github.com/William-gregory/SeaIceConcentrationForecasting/blob/main/images/network_inputs.png)


We can then create a covariance function to model the similarity between tasks (September sea ice concentration anomaly grid cells). For one particular grid cell, this looks like:

![alt text](https://github.com/William-gregory/SeaIceConcentrationForecasting/blob/main/images/task_covariance.png)

Before finally generating the spatial forecast of sea ice concentration anomalies (forecast left, truth right)

![alt text](https://github.com/William-gregory/SeaIceConcentrationForecasting/blob/main/images/forecast_vs_truth.png)
