# Tourism example

We compare the reconciliation results obtained from `pyhts` and R package `FoReco` and make sure the results are consistent when the base forecasts are identical.


## data

The Australian tourism dataset used in many literatures.

- `./data/tourism_S.csv` saves the summing matrix
- `./data/tourism.csv` saves the Australian monthly tourist visitor nights from January 1998 to December 2016.

## Run the example yourself


- Use `Rscript tourism.R` to generate base forecasts. It will generate base forecasts and residuals for the 555 time series at [1, 2, 3, 4, 6, 12] temporal aggregation levels, totally 12 csv files in `data` folder.
- Use `Rscript tourism_temporal.R` and `Rscript tourism_crosssectional.csv` to generate reconciled forecasts for 555 temporal hierarchies and 6 cross-sectional hierarchies, respectively.
- Run the notebook `tourism.ipynb` to compare the results.