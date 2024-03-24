"""
>>> Seasonal Adjustment Features File (SEAFU)

This document encompasses the core features required for the demonstration of seasonal adjustment within the FM5222 course framework.

Main Models: "Customized Implementation"
Comparable Models: "Statsmodel Implementation"

Created by: Frank Ygnacio
Date of Creation: March 22, 2024
"""

# import base libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEA_DATATOOLS import x13PathLocal
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.nonparametric.smoothers_lowess import lowess

#-----------------------------------------------------------------------------------------------------#
#######################################################################################################
#######################################################################################################
#################### Model 1.A: X11-ARIMA MODEL | Customized Implementation ###########################
#######################################################################################################
#######################################################################################################

# function to test for stationarity using the Augmented Dickey-Fuller test
def test_stationarity(timeseries):
    """
    Tests for stationarity in a time series using the Augmented Dickey-Fuller test.

    Params:
        - 'timeseries': A pandas Series representing the time series data.

    Output:
        - p-value from the ADF test. A p-value below 0.05 suggests stationarity.
    """    
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]  # p-value

# function to find the order of differencing (d) in ARIMA model
def find_order_of_differencing(series):
    """
    Determines the order of differencing (d) required to make a time series stationary.

    Params:
        - 'series': A pandas Series representing the time series data.

    Output:
        - Integer representing the order of differencing required (0, 1, 2, or 3).
    """    
    # test if the time series is already stationary
    p_value = test_stationarity(series)
    
    if p_value < 0.05:
        # series is stationary, no differencing needed
        return 0
    else:
        # series is not stationary, differencing needed
        differenced_series = series.diff().dropna()
        p_value = test_stationarity(differenced_series)
        if p_value < 0.05:
            return 1
        
        else:
            # second differencing
            differenced_series = differenced_series.diff().dropna()
            p_value = test_stationarity(differenced_series)
            
            if p_value < 0.05:
                return 2
            else:
                # rare case, but could require further differencing
                return 3

# function to perform the ARIMA modeling and forecasting
def arima_model_forecast(series, forecast_length = 12, pq = (1,1)):
    """    
    Fits an ARIMA model to the series and forecasts future values.

    Params:
        - 'series': A pandas Series representing the time series data.
        - 'forecast_length': Number of periods to forecast into the future.
        - 'pq': Tuple of integers for ARIMA(p, d, q) model orders.

    Output:
        - Pandas Series with the original data followed by the forecasted values.
    """    
    
    # find the order of differencing needed
    d = find_order_of_differencing(series)

    # perform the ARIMA modelling
    model = ARIMA(
        series, order=(pq[0], d, pq[1]), 
        seasonal_order=(pq[0], d, pq[1], forecast_length)
    )
    result = model.fit()
    forecast = result.forecast(steps=forecast_length)   
    
    # extend the original series with the forecasted datapoints
    new_org_series = pd.concat([series, forecast])
    return new_org_series

def simple_X11_arima_adjustment(series, window=12, forecast_length = 12, pq = (1,1)):
    """
    >>> CORE FUNCTION:    
    
    Perform a simplified X-11 ARIMA seasonal adjustment on a time series.

    This function extends the series using an ARIMA model forecast, 
    extracts the trend using a centered moving average, computes and normalizes seasonal factors, 
    and adjusts the series by removing the seasonal component. 

    The approach assumes a multiplicative time series model, such as:
    
                                                Ot = Tt * St * It
                                                
    where:
    
        * 'Ot' (observed time series) 
        * 'Tt' (trend component) 
        * 'St' (seasonal component) 
        * 'It' (irregular component). 
        
    The goal is to isolate and remove 'St' to analyze the underlying trend and irregular movements in the series.

    Params:
        - 'series': Pandas Series representing the time series data.
        - 'window': Size of the moving average window, typically 12 for monthly data.
        - 'forecast_length': Number of periods to forecast, extending the series for seasonal adjustment.
        - 'pq': Tuple (p,q) where 'p' is the ARIMA model autoregressive order and 'q' is the moving average order.

    Output:
        A pandas DataFrame containing the following components:
            - 'observed': Original time series data.
            - 'trend': Estimated trend component.
            - 'seasonal': Estimated seasonal component.
            - 'irregular': Estimated irregular component.
            - 'adjusted_series': seasonally adjusted series, as the original series over the seasonal component.

    This function provides a basic implementation of the X-11 ARIMA process, 
    suitable for educational purposes and not intended for official statistical analysis.

    Note: It is assumed that the index of the input series consists of datetime values, 
    which allows for the proper alignment of seasonal factors.

    Example:
        adjusted_result = simple_X11_arima_adjustment(data_series, window=12, forecast_length=12, pq=(1,1))
    """

    # step 0: perform ARIMA adjusment
    extended_series = arima_model_forecast(series, forecast_length, pq)
    
    # step 1: trend extraction using a moving average, centered around the current observation (what matters) 
    trend_component = extended_series.rolling(window=window, center=True).mean()
    
    # step 2: de-trended series (i.e., (Ot / Tt) = St * It)
    de_trended_series = extended_series / trend_component # = St * It (equivalent to differentiation)
    
    # step 3: calculate & normalize seasonal factors
    # > seasonal factors defined as the normalized average of the de-trended series (regular cycle behaviour assumed)
    seasonal_factors = de_trended_series.groupby(de_trended_series.index.month).mean() # get St as mean (goal: erase "It")
    seasonal_factors = seasonal_factors / seasonal_factors.mean() # normalization of St
    
    # step 4: adjust original series by seasonal factors (aligning the seasonal factors with the dates)
    seasonal_component = series.index.map(lambda d: seasonal_factors[d.month]) # align St by index Date
    seasonal_component = pd.Series(seasonal_component, index=series.index)
    #         dropp St from the extended series (i.e.m (Ot / St) = Tt * It)
    seasonally_adjusted = extended_series / seasonal_component 
    
    # optional: let's extract only the Irregular component
    irregular_component = de_trended_series / seasonal_component
    
    # define output dataframe
    result = pd.DataFrame({
        'observed': series,
        'trend': trend_component.loc[series.index],
        'seasonal': seasonal_component.loc[series.index],
        'irregular': irregular_component.loc[series.index],
        'adjusted_series': seasonally_adjusted.loc[series.index],
    })
    return result

#######################################################################################################
##################### Model 1.B: X11-ARIMA MODEL | Statsmodel Implementation ##########################
#######################################################################################################

def statsmodels_x13_arima(series, x13path=x13PathLocal, forecast_periods = 12):
    """
    COMPARABLE FUNCTION
    
    Apply the X-13ARIMA-SEATS seasonal adjustment and forecasting using the statsmodels package.

    This function interfaces with the X-13ARIMA-SEATS program to decompose the input series into seasonal, 
    trend, and irregular components, and to extend the series with forecasted values.

    Params:
        - 'series': pandas Series representing the time series data.
        - 'x13path': The path to the X-13ARIMA-SEATS executable. It is required if not in the default location.
        - 'forecast_periods': Number of periods to forecast beyond the end of the series (default is 12).

    Output:
        A pandas DataFrame with the following columns:
            - 'observed': Original time series data.
            - 'trend': The trend component estimated by the X-13ARIMA-SEATS model.
            - 'seasonal': The seasonal component estimated by the X-13ARIMA-SEATS model.
            - 'irregular': The irregular component estimated by the X-13ARIMA-SEATS model.
            - 'adjusted_series': The seasonally adjusted series.

    The seasonal adjustment is based on an underlying multiplicative model that assumes the observed 
    time series 'Ot' can be decomposed into a trend component 'Tt', a seasonal component 'St', 
    and an irregular component 'It', such that 'Ot = Tt * St * It'. 
    The objective is to remove the seasonal component 'St' from the original series.

    Important:
        The function relies on the X-13ARIMA-SEATS procedure provided by the U.S. Census Bureau. More details:
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.x13.x13_arima_analysis.html
    """
    # define the X13-ARIMA model using the PATH with the executable file
    modelFit = x13_arima_analysis(
        series, x12path = x13path, prefer_x13=True, forecast_periods = forecast_periods, trading=True, freq=12
    )
    
    # define output as dataframe
    result = pd.DataFrame({
        'observed': series,
        'trend': modelFit.trend,
        'seasonal': series - modelFit.seasadj,
        'irregular': modelFit.irregular,
        'adjusted_series': modelFit.seasadj,
    })
    
    return result


#-----------------------------------------------------------------------------------------------------#
#######################################################################################################
#######################################################################################################
#################### Model 2.A: STL with LOESS  | Customized Implementation ###########################
#######################################################################################################
#######################################################################################################

# Provided functions with potential adjustments
def loess_smooth(series, fraction=0.1):
    """
    Apply LOESS smoothing to a series using the specified fraction.

    Params:
        - 'series': Series to smooth.
        - 'fraction': Fraction of the dataset used in local regression.

    Output:
        - Smoothed series.
        
    Important:
        - 'lowes' is the smooth filter based on LOESS regression. More details at:
           https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html 
    By default, LOWES use a Tricube Weight Function: 
        > http://rspatial.r-forge.r-project.org/spgwr/reference/gwr.tricube.html
    """
    # 'endog' stands for the endogenous variable, or the dependent data, 
    # which here refers to the series values to be smoothed.
    endog = series.values
    
    # 'exog' stands for the exogenous variable, or the independent data. 
    # in this case, it's a sequence of integers with the same length as 'series', 
    # acting as the x-values for LOESS.
    exog = np.arange(len(series))
    
    # perform the LOESS smoothing. 'lowess' returns an array with the smoothed data, 
    # with 'frac' specifying the fraction of data used in each local regression.
    # 'return_sorted=False' specifies that the output should be in the same order as the input.
    result = lowess(endog, exog, frac=fraction, return_sorted=False)
    
    # convert the result array back into a pandas Series object, using the original series' index.
    # this ensures the smoothed values correspond to the original data's ordering.
    return pd.Series(result, index=series.index)

def extract_seasonal(series, period, fraction=0.1, iterations=3):
    """
    Iteratively extract and smooth the seasonal component of a series.

    Params:
        - 'series': Series from which to extract the seasonal component.
        - 'period': Seasonal period length.
        - 'fraction': Fraction of the dataset used in local regression.
        - 'iterations': Number of iterations to refine the seasonal estimate.

    Output:
        - Seasonal component.
    """
    # loop for the given number of iterations to refine the seasonal component 
    for i in range(iterations):  
        # detrend the series using LOESS smoothing | notice that the series is updated (i.e., refined)
        detrended = series - loess_smooth(series, fraction=fraction)  # series - trend_updated
        # group by month (or other period index) and calculate the mean to find the seasonal effect
        seasonal = detrended.groupby(detrended.index.month).transform('mean')  
        # replace the original series with the seasonal component for further refinement in the next iteration
        series = seasonal  
        
    return seasonal  

def stl_decomposition_adjustment(series, period, trend_fraction=0.1, seasonal_fraction=0.1, iterations=3):
    """
    >>> CORE FUNCTION:
    
    Simplified STL decomposition using iterative LOESS smoothing for trend and seasonal components.

    Params:
        - 'series': Pandas Series containing the time series data.
        - 'period': Number of observations per seasonal cycle.
        - 'trend_fraction': Fraction of data used when estimating each y-value in trend LOESS.
        - 'seasonal_fraction': Fraction of data used when estimating each y-value in seasonal LOESS.
        - 'iterations': Number of iterations for refining the seasonal component.

    Output:
        A pandas DataFrame with the following columns:
            - 'observed': Original time series data.
            - 'trend': The trend component estimated by the STL model.
            - 'seasonal': The seasonal component estimated by the STL model.
            - 'irregular': The irregular component estimated by the STL model.
            - 'adjusted_series': The seasonally adjusted series.

    Important:
        This function provides a basic STL decomposition without the full complexity of the official STL algorithm.
        It performs sequential steps to extract the trend and seasonal components.

    More details can be found at:
        - https://www.wessa.net/download/stl.pdf (base paper on STL)
        - https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html (STL statsmodels)
    """
    # compute the trend_component using LOESS
    trend = loess_smooth(series, fraction=trend_fraction)
    
    # compute the seasonal component 
    seasonal = extract_seasonal(series, period, fraction=seasonal_fraction, iterations=iterations)
    
    # find the irregular component
    irregular = series - trend - seasonal
    
    # find the seasonal adjusted time series
    seasonally_adjusted = series - seasonal
    
    # define output as dataframe
    result = pd.DataFrame({
        'observed': series,
        'trend': trend,
        'seasonal': seasonal,
        'irregular': irregular,
        'adjusted_series': seasonally_adjusted,
    })    
    
    return result

#######################################################################################################
##################### Model 2.B: STL with LOESS  | Statsmodel Implementation ##########################
#######################################################################################################

def statsmodel_stl_decomposition(series, period, seasonal=12, trend=None):
    """
    COMPARABLE FUNCTION
    
    Performs STL decomposition on a time series using the statsmodels implementation.

    Params:
        - 'series': pandas Series containing the time series data.
        - 'period': Number of observations in one seasonal cycle.
        - 'seasonal': Length of the seasonal smoother (default is 7).
        - 'trend': Length of the trend smoother, automatically determined if not provided.
        - 'robust': If True, uses a robust LOESS that is less sensitive to outliers.

    Output:
        - DataFrame with columns for observed, trend, seasonal, residual, and adjusted series.

    Important:
        The 'trend' length must be an odd number. If not provided or even, 
        it's set to an odd number that respects the seasonal period. 
        This function decomposes the series into seasonal and trend components, 
        allowing analysis of the underlying trend and irregular movements. 
        
    More details can be found at:
        - https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html (statsmodel functionality)
    """
    
    # updated trend and season for odd values only
    if trend % 2 == 0: trend-=1
    if seasonal % 2 == 0: seasonal-=1
        
    # perform STL decomposition
    stl = STL(series, period=period, seasonal=seasonal, trend=trend)
    result = stl.fit()
    
    # define output dataframe
    df_result = pd.DataFrame({
        'observed': series,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'irregular': result.resid,
        'adjusted_series': series - result.seasonal
    })
    
    return df_result


#-----------------------------------------------------------------------------------------------------#
#######################################################################################################
#######################################################################################################
###################### Model 3: CAMPLET Model | Customized Implementation #############################
#######################################################################################################
#######################################################################################################

def calculate_initial_seasonal_component(time_series, cycle_length=12):
    """
    Calculates the initial seasonal components based on the average values of groups 
    of corresponding observations over several cycles.

    Args:
    - time_series: Numpy array, the time series data.
    - cycle_length: Integer, the length of the cycle in the data (default is 12 for monthly data).

    Returns:
    - Numpy array with the initial seasonal component for each period in the cycle.
    """    
    # calculate the number of complete cycles in the time series
    years = int(len(time_series) / cycle_length)
    
    # compute average values for each period within the cycle across all cycles
    period_averages = np.array([np.mean(time_series[m::cycle_length]) for m in range(cycle_length)])
    
    # calculate the overall average of the time series for the complete cycles
    overall_average = np.mean(time_series[:years*cycle_length])
    
    # determine the initial seasonal component by subtracting the overall average from period averages
    initial_seasonal_component = period_averages - overall_average # := Si = \bar{Y}_i - \bar{y} 
    
    return initial_seasonal_component

def apply_seasonal_adjustment(time_series, initial_seasonal_component, cycle_length=12):
    """
    Applies seasonal adjustment to the time series using the initial seasonal components.

    Args:
    - time_series: Numpy array, the time series data to adjust.
    - initial_seasonal_component: Numpy array, initial seasonal components calculated for the series.
    - cycle_length: Integer, the length of the cycle in the data (default is 12 for monthly data).

    Returns:
    - Two Numpy arrays: the adjusted non-seasonal component and the seasonal component.
    """    
    # initialize arrays for the adjusted non-seasonal and seasonal components
    revised_NS = np.zeros_like(time_series)
    revised_S = np.zeros_like(time_series)
    
    # apply the initial seasonal components to each period in the series
    for i in range(len(time_series)):
        # identify the period within the cycle
        period = i % cycle_length  
        # apply the seasonal component
        revised_S[i] = initial_seasonal_component[period]
        # calculate the non-seasonal component
        revised_NS[i] = time_series[i] - revised_S[i]  
    
    return revised_NS, revised_S

def camplet_adjustment(series, cycle_length=12):
    """
    >>> CORE FUNCTION:
    
    An simplified version of the CAMPLET adjustment process for decomposing time series data
    into seasonal and non-seasonal components without the need for revision of past data points.

    Args:
    - time_series: Numpy array, the time series data to adjust.
    - cycle_length: Integer, the length of the cycle in the data (e.g., 12 for monthly data).

    Returns:
    - Tuple containing three Numpy arrays: the adjusted series, non-seasonal component, and seasonal component.
    
    Important:
        > More details can be found:
            - http://www.camplet.net/camplet-in-a-nutshell/
            - https://drive.google.com/drive/folders/1gqqE0o52CX6ic-CfEpquFfCfZ2OoY57i (CAMPLET repository)
    """    
    # calculate the initial seasonal components for the time series
    initial_seasonal_component = calculate_initial_seasonal_component(series, cycle_length)
    
    # apply seasonal adjustment using the initial seasonal components
    non_seasonal_component, seasonal_component = \
    apply_seasonal_adjustment(series, initial_seasonal_component, cycle_length)
    
    # define output dataframe
    df_result = pd.DataFrame({
        'observed': series,
        'seasonal': seasonal_component,
        'non_seasonal': non_seasonal_component,
    })
    
    return df_result    

