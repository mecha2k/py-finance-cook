import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import quandl
import intrinio_sdk as intrinio
import requests
import os
import warnings
from icecream import ic
from dotenv import load_dotenv
from pprint import pprint
from datetime import datetime

load_dotenv(verbose=True)

plt.style.use("seaborn")
plt.rcParams["figure.dpi"] = 300
warnings.simplefilter(action="ignore", category=FutureWarning)
quandl.ApiConfig.api_key = os.getenv("Quandl")
api_key = os.getenv("Alpha_vantage")

src_data = "data/aapl.csv"
start = datetime(2000, 1, 1)
end = datetime(2020, 12, 31)
try:
    aapl = pd.read_csv(src_data, parse_dates=["Date"])
    aapl = aapl.set_index("Date")
    print("data reading from file...")
except FileNotFoundError:
    aapl = yf.download("AAPL", start=start, end=end)
    aapl.to_csv(src_data)
ic(aapl.head())

# df_yahoo = yf.download("AAPL", start="2000-01-01", end="2010-12-31", progress=False)
# print(f"Downloaded {df_yahoo.shape[0]} rows of data.")
# ic(df_yahoo.head())
#
# df_quandl = quandl.get(dataset="WIKI/AAPL", start_date="2000-01-01", end_date="2010-12-31")
# print(f"Downloaded {df_quandl.shape[0]} rows of data.")
# ic(df_quandl.head())
#
#
# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey={api_key}"
# res = requests.get(url)
# pprint(res.json())

# # # 2. Authenticate using the personal API key and select the API:
# intrinio.ApiClient().configuration.api_key["api_key"] = os.getenv("Intrinio")
# intrinio.ApiClient().allow_retries(True)
# response = intrinio.SecurityApi().get_security_stock_prices(
#     identifier="AAPL",
#     start_date="2000-01-01",
#     end_date="2010-12-31",
#     frequency="daily",
#     page_size=100,
# )
# response_list = [x.to_dict() for x in response.stock_prices]
# df_intrinio = pd.DataFrame(response_list).sort_values("date")
# df_intrinio.set_index("date", inplace=True)
# print(f"Downloaded {df_intrinio.shape[0]} rows of data.")
# ic(df_intrinio.head())
#
# df = yf.download("AAPL", start="2000-01-01", end="2010-12-31", progress=False)
df = aapl
df = df.loc[:, ["Adj Close"]]
df.rename(columns={"Adj Close": "adj_close"}, inplace=True)

df["simple_rtn"] = df.adj_close.pct_change()
df["log_rtn"] = np.log(df.adj_close / df.adj_close.shift(1))
ic(df.head())

# df_all_dates = pd.DataFrame(index=pd.date_range(start="1999-12-31", end="2010-12-31", freq="D"))
# df = df_all_dates.join(df["adj_close"], how="left").fillna(method="ffill").asfreq("M")
df_all_dates = df["adj_close"].resample("M").mean()
ic(df_all_dates["1999-12-31":"2010-12-31"])

# ic(df_all_dates.join(df["adj_close"], how="left").head(10))
# ic(df_all_dates.join(df[["adj_close"]], how="left").fillna(method="ffill"))
# ic(df_all_dates.join(df[["adj_close"]], how="left").fillna(method="ffill").asfreq("M"))

df = df_all_dates["1999-12-31":"2010-12-31"]
# 3. Download inflation data from Quandl:
df_cpi = quandl.get(dataset="RATEINF/CPI_USA", start_date="1999-12-01", end_date="2010-12-31")
df_cpi.rename(columns={"Value": "cpi"}, inplace=True)

# 4. Merge inflation data to prices:
df_merged = df.join(df_cpi, how="left")

# 5. Calculate simple returns and inflation rate:
df_merged["simple_rtn"] = df_merged.adj_close.pct_change()
df_merged["inflation_rate"] = df_merged.cpi.pct_change()

# 6. Adjust returns for inflation:
df_merged["real_rtn"] = (df_merged.simple_rtn + 1) / (df_merged.inflation_rate + 1) - 1
ic(df_merged.head())
#
# # ## Changing frequency
# df = yf.download("AAPL", start="2000-01-01", end="2010-12-31", auto_adjust=False, progress=False)
#
# # keep only the adjusted close price
# df = df.loc[:, ["Adj Close"]]
# df.rename(columns={"Adj Close": "adj_close"}, inplace=True)
#
# # calculate simple returns
# df["log_rtn"] = np.log(df.adj_close / df.adj_close.shift(1))
#
# # remove redundant data
# df.drop("adj_close", axis=1, inplace=True)
# df.dropna(axis=0, inplace=True)
# ic(df.head())
#
#
# def realized_volatility(x):
#     return np.sqrt(np.sum(x ** 2))
#
#
# # 3. Calculate monthly realized volatility:
# df_rv = df.groupby(pd.Grouper(freq="M")).apply(realized_volatility)
# df_rv.rename(columns={"log_rtn": "rv"}, inplace=True)
#
# # 4. Annualize the values:
# df_rv.rv = df_rv.rv * np.sqrt(12)
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(df)
# ax[1].plot(df_rv)
# plt.tight_layout()
# plt.savefig("images/ch1_im6.png")

# # download data as pandas DataFrame
# df = yf.download("MSFT", auto_adjust=False, progress=False)
# df = df.loc[:, ["Adj Close"]]
# df.rename(columns={"Adj Close": "adj_close"}, inplace=True)
#
# # create simple and log returns
# df["simple_rtn"] = df.adj_close.pct_change()
# df["log_rtn"] = np.log(df.adj_close / df.adj_close.shift(1))
#
# # dropping NA's in the first row
# df.dropna(how="any", inplace=True)


# # ### How to do it...
#
# # #### the `plot` method of pandas
#
# # In[14]:
#
#
# fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#
# # add prices
# df.adj_close.plot(ax=ax[0])
# ax[0].set(title = 'MSFT time series',
#           ylabel = 'Stock price ($)')
#
# # add simple returns
# df.simple_rtn.plot(ax=ax[1])
# ax[1].set(ylabel = 'Simple returns (%)')
#
# # add log returns
# df.log_rtn.plot(ax=ax[2])
# ax[2].set(xlabel = 'Date',
#           ylabel = 'Log returns (%)')
#
# ax[2].tick_params(axis='x',
#                   which='major',
#                   labelsize=12)
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im7.png')
# plt.show()
#
#
# # #### `plotly` + `cufflinks`
#
# # 1. Import the libraries and handle the settings:
#
# # In[14]:
#
#
# import cufflinks as cf
# from plotly.offline import iplot, init_notebook_mode
#
# # set up settings (run it once)
# # cf.set_config_file(world_readable=True, theme='pearl',
# #                    offline=True)
#
# # initialize notebook display
# init_notebook_mode()
#
#
# # 2. Create the plots:
#
# # In[15]:
#
#
# df.iplot(subplots=True, shape=(3,1), shared_xaxes=True, title='MSFT time series')
#
#
# # ## Identifying outliers
#
# # 0: Repeat the steps from recipe *Converting prices to returns*:
#
# # In[3]:
#
#
# import pandas as pd
# import yfinance as yf
#
#
# # In[4]:
#
#
# df = yf.download('AAPL',
#                  start='2000-01-01',
#                  end='2010-12-31',
#                  progress=False)
#
# df = df.loc[:, ['Adj Close']]
# df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
#
#
# # In[5]:
#
#
# df['simple_rtn'] = df.adj_close.pct_change()
#
#
# # In[6]:
#
#
# df.head()
#
#
# # 1. Calculate the rolling mean and standard deviation:
#
# # In[7]:
#
#
# df_rolling = df[['simple_rtn']].rolling(window=21)                                .agg(['mean', 'std'])
# df_rolling.columns = df_rolling.columns.droplevel()
#
#
# # 2. Join the rolling metrics to the original data:
#
# # In[8]:
#
#
# df_outliers = df.join(df_rolling)
#
#
# # 3. Define a function for detecting outliers:
#
# # In[9]:
#
#
# def indentify_outliers(row, n_sigmas=3):
#     '''
#     Function for identifying the outliers using the 3 sigma rule.
#     The row must contain the following columns/indices: simple_rtn, mean, std.
#
#     Parameters
#     ----------
#     row : pd.Series
#         A row of a pd.DataFrame, over which the function can be applied.
#     n_sigmas : int
#         The number of standard deviations above/below the mean - used for detecting outliers
#
#     Returns
#     -------
#     0/1 : int
#         An integer with 1 indicating an outlier and 0 otherwise.
#     '''
#     x = row['simple_rtn']
#     mu = row['mean']
#     sigma = row['std']
#
#     if (x > mu + n_sigmas * sigma) | (x < mu - n_sigmas * sigma):
#         return 1
#     else:
#         return 0
#
#
# # 4. Identify the outliers and extract their values for later use:
#
# # In[10]:
#
#
# df_outliers['outlier'] = df_outliers.apply(indentify_outliers,
#                                            axis=1)
# outliers = df_outliers.loc[df_outliers['outlier'] == 1,
#                            ['simple_rtn']]
#
#
# # 5. Plot the results:
#
# # In[11]:
#
#
# fig, ax = plt.subplots()
#
# ax.plot(df_outliers.index, df_outliers.simple_rtn,
#         color='blue', label='Normal')
# ax.scatter(outliers.index, outliers.simple_rtn,
#            color='red', label='Anomaly')
# ax.set_title("Apple's stock returns")
# ax.legend(loc='lower right')
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im9.png')
# plt.show()
#
#
# # ## Investigating stylized facts of asset returns
#
# # ### How to do it...
#
# # 1. Import the libraries:
#
# # In[3]:
#
#
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import seaborn as sns
# import scipy.stats as scs
# import statsmodels.api as sm
# import statsmodels.tsa.api as smt
#
#
# # 2. Download the S&P 500 data and calculate the returns:
#
# # In[4]:
#
#
# df = yf.download('^GSPC',
#                  start='1985-01-01',
#                  end='2018-12-31',
#                  progress=False)
#
# df = df[['Adj Close']].rename(columns={'Adj Close': 'adj_close'})
# df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
# df = df[['adj_close', 'log_rtn']].dropna(how = 'any')
#
#
# # #### Fact 1 - Non-Gaussian distribution of returns
#
# # 1. Calculate the Normal PDF using the mean and standard deviation of the observed returns:
#
# # In[5]:
#
#
# r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
# mu = df.log_rtn.mean()
# sigma = df.log_rtn.std()
# norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)
#
#
# # 2. Plot the histogram and the Q-Q Plot:
#
# # In[6]:
#
#
# fig, ax = plt.subplots(1, 2, figsize=(16, 8))
#
# # histogram
# sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])
# ax[0].set_title('Distribution of S&P 500 returns', fontsize=16)
# ax[0].plot(r_range, norm_pdf, 'g', lw=2,
#            label=f'N({mu:.2f}, {sigma**2:.4f})')
# ax[0].legend(loc='upper left');
#
# # Q-Q plot
# qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
# ax[1].set_title('Q-Q plot', fontsize = 16)
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im10.png')
# plt.show()
#
#
# # 3. Print the summary statistics of the log returns:
#
# # In[7]:
#
#
# jb_test = scs.jarque_bera(df.log_rtn.values)
#
# print('---------- Descriptive Statistics ----------')
# print('Range of dates:', min(df.index.date), '-', max(df.index.date))
# print('Number of observations:', df.shape[0])
# print(f'Mean: {df.log_rtn.mean():.4f}')
# print(f'Median: {df.log_rtn.median():.4f}')
# print(f'Min: {df.log_rtn.min():.4f}')
# print(f'Max: {df.log_rtn.max():.4f}')
# print(f'Standard Deviation: {df.log_rtn.std():.4f}')
# print(f'Skewness: {df.log_rtn.skew():.4f}')
# print(f'Kurtosis: {df.log_rtn.kurtosis():.4f}')
# print(f'Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}')
#
#
# # #### Fact 2 - Volatility Clustering
#
# # 1. Run the following code to visualize the log returns series:
#
# # In[8]:
#
#
# df.log_rtn.plot(title='Daily S&P 500 returns', figsize=(10, 6))
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im12.png')
# plt.show()
#
#
# # #### Fact 3 - Absence of autocorrelation in returns
#
# # 1. Define the parameters for creating the Autocorrelation plots:
#
# # In[9]:
#
#
# N_LAGS = 50
# SIGNIFICANCE_LEVEL = 0.05
#
#
# # 2. Run the following code to create ACF plot of log returns:
#
# # In[10]:
#
#
# acf = smt.graphics.plot_acf(df.log_rtn,
#                             lags=N_LAGS,
#                             alpha=SIGNIFICANCE_LEVEL)
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im13.png')
# plt.show()
#
#
# # #### Fact 4 - Small and decreasing autocorrelation in squared/absolute returns
#
# # In[11]:
#
#
# fig, ax = plt.subplots(2, 1, figsize=(12, 10))
#
# smt.graphics.plot_acf(df.log_rtn ** 2, lags=N_LAGS,
#                       alpha=SIGNIFICANCE_LEVEL, ax = ax[0])
# ax[0].set(title='Autocorrelation Plots',
#           ylabel='Squared Returns')
#
# smt.graphics.plot_acf(np.abs(df.log_rtn), lags=N_LAGS,
#                       alpha=SIGNIFICANCE_LEVEL, ax = ax[1])
# ax[1].set(ylabel='Absolute Returns',
#           xlabel='Lag')
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im14.png')
# plt.show()
#
#
# # #### Fact 5 - Leverage effect
#
# # 1. Calculate volatility measures as moving standard deviations
#
# # In[12]:
#
#
# df['moving_std_252'] = df[['log_rtn']].rolling(window=252).std()
# df['moving_std_21'] = df[['log_rtn']].rolling(window=21).std()
#
#
# # 2. Plot all the series:
#
# # In[13]:
#
#
# fig, ax = plt.subplots(3, 1, figsize=(18, 15),
#                        sharex=True)
#
# df.adj_close.plot(ax=ax[0])
# ax[0].set(title='S&P 500 time series',
#           ylabel='Price ($)')
#
# df.log_rtn.plot(ax=ax[1])
# ax[1].set(ylabel='Log returns (%)')
#
# df.moving_std_252.plot(ax=ax[2], color='r',
#                        label='Moving Volatility 252d')
# df.moving_std_21.plot(ax=ax[2], color='g',
#                       label='Moving Volatility 21d')
# ax[2].set(ylabel='Moving Volatility',
#           xlabel='Date')
# ax[2].legend()
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im15.png')
# plt.show()
#
#
# # ### There's more
#
# # 1. Download and preprocess the prices of S&P 500 and VIX:
#
# # In[14]:
#
#
# df = yf.download(['^GSPC', '^VIX'],
#                  start='1985-01-01',
#                  end='2018-12-31',
#                  progress=False)
# df = df[['Adj Close']]
# df.columns = df.columns.droplevel(0)
# df = df.rename(columns={'^GSPC': 'sp500', '^VIX': 'vix'})
#
#
# # 2. Calculate log returns:
#
# # In[15]:
#
#
# df['log_rtn'] = np.log(df.sp500 / df.sp500.shift(1))
# df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))
# df.dropna(how='any', axis=0, inplace=True)
#
#
# # 3. Plot a scatterplot with the returns on the axes and fit a regression line to identify trend:
#
# # In[16]:
#
#
# corr_coeff = df.log_rtn.corr(df.vol_rtn)
#
# ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df,
#                  line_kws={'color': 'red'})
# ax.set(title=f'S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})',
#        ylabel='VIX log returns',
#        xlabel='S&P 500 log returns')
#
# # plt.tight_layout()
# # plt.savefig('images/ch1_im16.png')
# plt.show()
#
