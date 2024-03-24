# Basic Seasonal Adjustment Models in Time Series Analysis

This repository contains Jupyter notebooks that demonstrate the application of various seasonal adjustment methods to time series data. The primary focus is on the X-11 ARIMA model, SLT with LOESS (Locally Weighted Scatterplot Smoothing), and the CAMPLET model. These examples are designed to showcase how to preprocess, model, and adjust time series data to identify and account for seasonal variations accurately.

Examples of these can be found at jupyter notebook: `Seasonal_Adjustment_Examples.ipynb`

## Models Covered

- **X-11 ARIMA**: A comprehensive approach for modeling and forecasting time series data, addressing trends, seasonality, and irregular components through ARIMA modeling and the X11 procedure.
- **SLT with LOESS**: This method emphasizes the LOESS procedure for fitting local polynomials and computing weights, tailored for smoothing time series data and minimizing the influence of outliers.
- **CAMPLET**: Focuses on decomposing a time series into seasonal and non-seasonal components, illustrating its strength in adjusting data without revising historical figures as new observations emerge.

## Getting Started

To explore these examples, clone this repository to your local machine:

```
git clone https://github.com/<frankstack>/<SeasonalAdjustment>.git
```

Navigate into the project directory and launch Jupyter Notebook to open the provided `.ipynb` files:

```
cd <SeasonalAdjustment>
jupyter notebook
```

## Prerequisites

Ensure you have the following Python libraries installed to run the notebooks:

- Pandas
- NumPy
- Matplotlib
- statsmodels (for specific statistical models and seasonal decomposition)

These can be installed via pip using:

```
pip install pandas numpy matplotlib statsmodels
```

## Usage

Each notebook is self-contained and walks through the process of applying a specific seasonal adjustment method to a dataset. The steps include preprocessing, model fitting, seasonal adjustment, and visualization of the results. Code cells can be executed in sequence to replicate the analysis and observe the impact of each method on the adjusted time series.

## Contributions

Contributions are welcome! If you have improvements or corrections, please open a pull request. For major changes, open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## References

A comprehensive list of references, including papers and additional resources, is provided within each notebook for further reading and context about the methods and their applications in time series analysis.

---
