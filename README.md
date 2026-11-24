# Weather Forecasting for Patna (ERA5)

Supervised models that forecast temperature and rain 1h, 6h and 24h ahead for one ERA5 grid cell near Patna, evaluated against strong baselines on a held-out year. This is a historical backtesting experiment on reanalysis data, not an operational forecast.

_The full write-up (results, method, limitations) will be added when the project is finished._

## Data and attribution

- **Data:** ERA5 hourly reanalysis (2011-2025) for the 0.25° grid cell at 25.5°N, 85.25°E, about 15 km from central Patna. It was downloaded through the [Open-Meteo](https://open-meteo.com/) Historical Weather API with `models=era5`. The raw data is not included in this repository.
- **Changes made:** the data was resampled into lagged features, daily totals and forecast targets, and used to train and evaluate models. Every number and plot in this project is derived from the data, not raw ERA5 values.
- [Weather data by Open-Meteo.com](https://open-meteo.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
- **Dataset:** C3S (2018). *ERA5 hourly data on single levels from 1940 to present* [Dataset]. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.adbb2d47 (licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Reanalysis reference:** Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999-2049. https://doi.org/10.1002/qj.3803
- **API citation:** Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649

Neither Open-Meteo, ECMWF nor the Copernicus programme endorses this project.
