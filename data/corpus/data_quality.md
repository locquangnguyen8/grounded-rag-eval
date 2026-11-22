# Data Quality Control

Every reading passes three automated checks before it reaches dashboards or models.

1. Range check: the value must fall inside the physical range for that sensor. For example, salinity must be between 0 and 40 PSU.
2. Spike check: a reading is flagged if it differs from the previous hour's mean by more than three standard deviations.
3. Flatline check: a reading is flagged if the sensor reports six identical consecutive values, which usually means the sensor has stalled.

Each reading receives a quality flag: 1 means good, 3 means suspect and 4 means bad. Readings flagged 4 are excluded from the forecasting model. Readings flagged 3 are kept but shown with a warning.

A data engineer manually reviews every suspect or bad reading within 48 hours.
