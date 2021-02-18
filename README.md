# Sea Ice Concentration Forecasting
This is follow-on work from the forecasting September sea ice extent using Complex Networks and Gaussian Process Regression (see repository: ![SeaIceExtentForecasting](https://github.com/William-gregory/SeaIceExtentForecasting) )

This contains code used to generate spatial forecasts of sea ice concentration in the East Siberian and Laptev seas using multi-task Gaussian Process Regression, in a non-stationary inter-task covariance structure. This code is largely based on previous work by:

Paciorek, C.J. and Schervish, M.J., 2006. Spatial modelling using a new class of nonstationary covariance functions. Environmetrics: The official journal of the International Environmetrics Society, 17(5), pp.483-506. [https://doi.org/10.1002/env.785](https://doi.org/10.1002/env.785)

Stegle, O., Lippert, C., Mooij, J.M., Larence, N.D. and Borgwardt, K., 2011. Efficient inference in matrix-variate Gaussian models with iid observation noise.

Rakitsch, B., Lippert, C., Borgwardt, K. and Stegle, O., 2013, December. It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals. In Proceedings of the 26th International Conference on Neural Information Processing Systems-Volume 1 (pp. 1466-1474).

![alt text](https://github.com/William-gregory/SeaIceConcentrationForecasting/blob/main/images/network_inputs.png)
