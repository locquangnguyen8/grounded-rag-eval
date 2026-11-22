# Closure Risk Forecasting Model

HEMP forecasts the risk that each harvest area will be closed in the next 7 days.

The model is a gradient-boosted decision tree classifier. Its main features are lagged catchment rainfall, river flow, salinity, water temperature and the number of days since the last closure.

The model is evaluated with a rolling-origin backtest: it is trained on all data before a cut-off date and tested on the following month, and this is repeated for twelve consecutive months. On this backtest the model reached a balanced accuracy of 0.78.

The model is retrained on the first day of every month. It is also retrained early if its balanced accuracy over the most recent four weeks falls below 0.70.

SHAP values are published alongside each forecast so growers can see which features drove the risk score.
