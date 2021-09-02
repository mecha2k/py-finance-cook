import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import warnings
import os

from datetime import date, datetime
from dotenv import load_dotenv
from icecream import ic

plt.style.use("seaborn-colorblind")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 150
warnings.simplefilter(action="ignore", category=FutureWarning)

load_dotenv(verbose=True)
quandl.ApiConfig.api_key = os.getenv("Quandl")

if __name__ == "__main__":
    src_data = "data/yf_capm.pkl"
    start = datetime(2000, 1, 1)
    end = datetime(2020, 12, 31)
    try:
        capm = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        capm = yf.download(["AMZN", "^GSPC"], start=start, end=end, adjusted=True, progress=False)
        capm.to_pickle(src_data)

    ## Implementing the CAPM in Python
    df = capm["2014-1":"2018-12"]
    df.info()
    ic(df.head())

    X = (
        df["Adj Close"]
        .rename(columns={"AMZN": "asset", "^GSPC": "market"})
        .resample("M")
        .last()
        .pct_change()
        .dropna()
    )
    ic(X.head())
    # 5. Calculate beta using the covariance approach:
    covariance = X.cov().iloc[0, 1]
    benchmark_variance = X.market.var()
    beta = covariance / benchmark_variance
    ic(beta)
    # 6. Prepare the input and estimate CAPM as a linear regression:
    y = X.pop("asset")
    X = sm.add_constant(X)
    capm_model = sm.OLS(y, X).fit()
    ic(capm_model.summary())

    # # #### Risk-free rate (13 Week Treasury Bill)
    #
    # # In[9]:
    #
    #
    # # period lenght in days
    # N_DAYS = 90
    #
    # # download data from Yahoo finance
    # df_rf = yf.download('^IRX',
    #                     start=START_DATE,
    #                     end=END_DATE,
    #                     progress=False)
    #
    # # resample to monthly by taking last value from each month
    # rf = df_rf.resample('M').last().Close / 100
    #
    # # calculate the corresponding daily risk-free return
    # rf = ( 1 / (1 - rf * N_DAYS / 360) )**(1 / N_DAYS)
    #
    # # convert to monthly and subtract 1
    # rf = (rf ** 30) - 1
    #
    # # plot the risk-free rate
    # rf.plot(title='Risk-free rate (13 Week Treasury Bill)')
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch4_im2.png')
    # plt.show()


# # #### Risk-free rate (3-Month Treasury Bill)
#
# # In[10]:
#
#
# import pandas_datareader.data as web
#
# # download the data
# rf = web.DataReader('TB3MS', 'fred', start=START_DATE, end=END_DATE)
#
# # convert to monthly
# rf = (1 + (rf / 100)) ** (1 / 12) - 1
#
# # plot the risk-free rate
# rf.plot(title='Risk-free rate (3-Month Treasury Bill)')
#
# plt.tight_layout()
# # plt.savefig('images/ch4_im3.png')
# plt.show()
#
#
# # ## Implementing the Fama-French three-factor model in Python
#
# # ### How to do it...
#
# # 1. Import the libraries:
#
# # In[11]:
#
#
# import pandas as pd
# import yfinance as yf
# import statsmodels.formula.api as smf
#
#
# # 2. Download data from prof. French's website:
#
# # In[15]:
#
#
# # download the zip file from Prof. French's website
# get_ipython().system('wget http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip')
#
# # unpack the zip
# get_ipython().system('unzip -a F-F_Research_Data_Factors_CSV.zip')
#
# # remove the zip
# get_ipython().system('rm F-F_Research_Data_Factors_CSV.zip')
#
#
# # 3. Define parameters:
#
# # In[12]:
#
#
# RISKY_ASSET = 'FB'
# START_DATE = '2013-12-31'
# END_DATE = '2018-12-31'
#
#
# # 4. Load data from the source CSV file and keep only the monthly data:
#
# # In[18]:
#
#
# # load data from csv
# factor_df = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3)
#
# # identify where the annual data starts
# STR_TO_MATCH = ' Annual Factors: January-December '
# indices = factor_df.iloc[:, 0] == STR_TO_MATCH
# start_of_annual = factor_df[indices].index[0]
#
# # keep only monthly data
# factor_df = factor_df[factor_df.index < start_of_annual]
#
#
# # 5. Rename columns of the DataFrame, set a datetime index and filter by dates:
#
# # In[19]:
#
#
# # rename columns
# factor_df.columns = ['date', 'mkt', 'smb', 'hml', 'rf']
#
# # convert strings to datetime
# factor_df['date'] = pd.to_datetime(factor_df['date'],
#                                    format='%Y%m') \
#                       .dt.strftime("%Y-%m")
#
# # set index
# factor_df = factor_df.set_index('date')
#
# # filter only required dates
# factor_df = factor_df.loc[START_DATE:END_DATE]
#
#
# # 6. Convert the values to numeric and divide by 100:
#
# # In[20]:
#
#
# factor_df = factor_df.apply(pd.to_numeric,
#                             errors='coerce') \
#                      .div(100)
# factor_df.head()
#
#
# # 7. Download the prices of the risky asset:
#
# # In[21]:
#
#
# asset_df = yf.download(RISKY_ASSET,
#                        start=START_DATE,
#                        end=END_DATE,
#                        adjusted=True,
#                        progress=False)
#
# print(f'Downloaded {asset_df.shape[0]} rows of data.')
#
#
# # 8. Calculate monthly returns on the risky asset:
#
# # In[22]:
#
#
# y = asset_df['Adj Close'].resample('M')                          .last()                          .pct_change()                          .dropna()
#
# y.index = y.index.strftime('%Y-%m')
# y.name = 'rtn'
# y.head()
#
#
# # 9. Merge the datasets and calculate excess returns:
#
# # In[23]:
#
#
# ff_data = factor_df.join(y)
# ff_data['excess_rtn'] = ff_data.rtn - ff_data.rf
#
#
# # 10. Estimate the three-factor model:
#
# # In[24]:
#
#
# # define and fit the regression model
# ff_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml',
#                    data=ff_data).fit()
#
# # print results
# print(ff_model.summary())
#
#
# # ### There's more...
#
# # 1. Import the libraries:
#
# # In[25]:
#
#
# from pandas_datareader.famafrench import get_available_datasets
# import pandas_datareader.data as web
#
#
# # 2. Print available datasets (here only first 5):
#
# # In[26]:
#
#
# get_available_datasets()[:5]
#
#
# # 3. Download the selected dataset:
#
# # In[27]:
#
#
# ff_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
#                          start='2014-01-01')
#
#
# # In[28]:
#
#
# ff_dict.keys()
#
#
# # 4. Inspect the description of the dataset
#
# # In[29]:
#
#
# print(ff_dict['DESCR'])
#
#
# # 5. View the monthly dataset:
#
# # In[30]:
#
#
# ff_dict[0].head()
#
#
# # ## Implementing the rolling three-factor model on a portfolio of assets
#
# # ### How to do it...
#
# # 1. Import the libraries:
#
# # In[1]:
#
#
# import pandas as pd
# import yfinance as yf
# import statsmodels.formula.api as smf
# import pandas_datareader.data as web
#
#
# # 2. Define the parameters:
#
# # In[2]:
#
#
# ASSETS = ['AMZN', 'GOOG', 'AAPL', 'MSFT']
# WEIGHTS = [0.25, 0.25, 0.25, 0.25]
# START_DATE = '2009-12-31'
# END_DATE = '2018-12-31'
#
#
# # 3. Download the factor related data:
#
# # In[3]:
#
#
# df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
#                                  start=START_DATE)[0]
# df_three_factor = df_three_factor.div(100)
# df_three_factor.index = df_three_factor.index.format()
#
#
# # 4. Download the prices of risky assets from Yahoo Finance:
#
# # In[4]:
#
#
# asset_df = yf.download(ASSETS,
#                        start=START_DATE,
#                        end=END_DATE,
#                        adjusted=True,
#                        progress=False)
#
# print(f'Downloaded {asset_df.shape[0]} rows of data.')
#
#
# # 5. Calculate the monthly returns on the risky assets:
#
# # In[5]:
#
#
# asset_df = asset_df['Adj Close'].resample('M')                                 .last()                                 .pct_change()                                 .dropna()
# # reformat index for joining
# asset_df.index = asset_df.index.strftime('%Y-%m')
#
#
# # 6. Calculate the portfolio returns:
#
# # In[6]:
#
#
# asset_df['portfolio_returns'] = np.matmul(asset_df[ASSETS].values,
#                                           WEIGHTS)
# asset_df.head()
#
#
# # In[10]:
#
#
# asset_df.plot()
#
#
# # 7. Merge the datasets:
#
# # In[11]:
#
#
# ff_data = asset_df.join(df_three_factor).drop(ASSETS, axis=1)
# ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
# ff_data['portf_ex_rtn'] = ff_data.portf_rtn - ff_data.rf
#
#
# # In[12]:
#
#
# ff_data.head()
#
#
# # 8. Define a function for the rolling n-factor model
#
# # In[15]:
#
#
# def rolling_factor_model(input_data, formula, window_size):
#     '''
#     Function for estimating the Fama-French (n-factor) model using a rolling window of fixed size.
#
#     Parameters
#     ------------
#     input_data : pd.DataFrame
#         A DataFrame containing the factors and asset/portfolio returns
#     formula : str
#         `statsmodels` compatible formula representing the OLS regression
#     window_size : int
#         Rolling window length.
#
#     Returns
#     -----------
#     coeffs_df : pd.DataFrame
#         DataFrame containing the intercept and the three factors for each iteration.
#     '''
#
#     coeffs = []
#
#     for start_index in range(len(input_data) - window_size + 1):
#         end_index = start_index + window_size
#
#         # define and fit the regression model
#         ff_model = smf.ols(
#             formula=formula,
#             data=input_data[start_index:end_index]
#         ).fit()
#
#         # store coefficients
#         coeffs.append(ff_model.params)
#
#     coeffs_df = pd.DataFrame(
#         coeffs,
#         index=input_data.index[window_size - 1:]
#     )
#
#     return coeffs_df
#
#
# # 9. Estimate the rolling three-factor model and plot the results:
#
# # In[16]:
#
#
# MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
# results_df = rolling_factor_model(ff_data,
#                                   MODEL_FORMULA,
#                                   window_size=60)
# results_df.plot(title = 'Rolling Fama-French Three-Factor model')
#
# plt.tight_layout()
# # plt.savefig('images/ch4_im8.png')
# plt.show()
#
#
# # ## Implementing the four- and five-factor models in Python
#
# # ### How to do it...
#
# # 1. Import the libraries:
#
# # In[3]:
#
#
# import pandas as pd
# import yfinance as yf
# import statsmodels.formula.api as smf
# import pandas_datareader.data as web
#
#
# # 2. Specify the risky asset and the time horizon:
#
# # In[4]:
#
#
# RISKY_ASSET = 'AMZN'
# START_DATE = '2013-12-31'
# END_DATE = '2018-12-31'
#
#
# # 3. Download the risk factors from prof. French's website:
#
# # In[ ]:
#
#
# # three factors
# df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
#                                  start=START_DATE)[0]
# df_three_factor.index = df_three_factor.index.format()
#
# # momentum factor
# df_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench',
#                         start=START_DATE)[0]
# df_mom.index = df_mom.index.format()
#
# # five factors
# df_five_factor = web.DataReader('F-F_Research_Data_5_Factors_2x3',
#                                 'famafrench',
#                                 start=START_DATE)[0]
# df_five_factor.index = df_five_factor.index.format()
#
#
# # 4. Download the data of the risky asset from Yahoo Finance:
#
# # In[50]:
#
#
# asset_df = yf.download(RISKY_ASSET,
#                        start=START_DATE,
#                        end=END_DATE,
#                        adjusted=True,
#                        progress=False)
#
# print(f'Downloaded {asset_df.shape[0]} rows of data.')
#
#
# # 5. Calculate monthly returns:
#
# # In[45]:
#
#
# y = asset_df['Adj Close'].resample('M')                          .last()                          .pct_change()                          .dropna()
#
# y.index = y.index.strftime('%Y-%m')
# y.name = 'return'
#
#
# # 6. Merge the datasets for the four-factor models:
#
# # In[46]:
#
#
# # join all datasets on the index
# four_factor_data = df_three_factor.join(df_mom).join(y)
#
# # rename columns
# four_factor_data.columns = ['mkt', 'smb', 'hml', 'rf', 'mom', 'rtn']
#
# # divide everything (except returns) by 100
# four_factor_data.loc[:, four_factor_data.columns != 'rtn'] /= 100
#
# # convert index to datetime
# four_factor_data.index = [pd.to_datetime(x, format='%Y-%m') for x in four_factor_data.index]
#
# # select period of interest
# four_factor_data = four_factor_data.loc[START_DATE:END_DATE]
#
# # calculate excess returns
# four_factor_data['excess_rtn'] = four_factor_data.rtn - four_factor_data.rf
#
# four_factor_data.head()
#
#
# # 7. Merge the datasets for the five-factor models:
#
# # In[47]:
#
#
# # join all datasets on the index
# five_factor_data = df_five_factor.join(y)
#
# # rename columns
# five_factor_data.columns = ['mkt', 'smb', 'hml', 'rmw', 'cma', 'rf', 'rtn']
#
# # divide everything (except returns) by 100
# five_factor_data.loc[:, five_factor_data.columns != 'rtn'] /= 100
#
# # convert index to datetime
# five_factor_data.index = [pd.to_datetime(x, format='%Y-%m') for x in five_factor_data.index]
#
# # select period of interest
# five_factor_data = five_factor_data.loc[START_DATE:END_DATE]
#
# # calculate excess returns
# five_factor_data['excess_rtn'] = five_factor_data.rtn - five_factor_data.rf
#
# five_factor_data.head()
#
#
# # 8. Estimate the four-factor model:
#
# # In[48]:
#
#
# four_factor_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml + mom',
#                             data=four_factor_data).fit()
#
# print(four_factor_model.summary())
#
#
# # 9. Estimate the five-factor model:
#
# # In[49]:
#
#
# five_factor_model = smf.ols(
#     formula='excess_rtn ~ mkt + smb + hml + rmw + cma',
#     data=five_factor_data
# ).fit()
#
# print(five_factor_model.summary())
#
#
# # In[ ]:
