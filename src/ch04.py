import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import warnings
import os
import pandas_datareader.data as web

from pandas_datareader.famafrench import get_available_datasets
from datetime import date, datetime
from dotenv import load_dotenv
from icecream import ic

plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 300
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
    ic(X.cov())
    ic(X.corr())
    ic(X.market.var())
    ic(beta)
    ic(covariance / np.sqrt(benchmark_variance) / np.sqrt(X.asset.var()))
    # 6. Prepare the input and estimate CAPM as a linear regression:
    y = X.pop("asset")
    X = sm.add_constant(X)
    capm_model = sm.OLS(y, X).fit()
    ic(X.head())
    ic(capm_model.summary())

    #### Risk-free rate (13 Week Treasury Bill)
    N_DAYS = 90
    src_data = "data/yf_irx.pkl"
    try:
        irx = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        irx = yf.download("^IRX", start=start, end=end, adjusted=True, progress=False)
        irx.to_pickle(src_data)

    df_rf = irx["2014":"2018"]
    rf = df_rf.resample("M").last().Close / 100
    # calculate the corresponding daily risk-free return
    rf = (1 / (1 - rf * N_DAYS / 360)) ** (1 / N_DAYS)
    # convert to monthly and subtract 1
    rf = (rf ** 30) - 1

    rf.plot(title="Risk-free rate (13 Week Treasury Bill)")
    plt.tight_layout()
    plt.savefig("images/ch4_im2.png")

    #### Risk-free rate (3-Month Treasury Bill)
    src_data = "data/yf_tb3ms.pkl"
    try:
        irx = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        irx = web.DataReader("TB3MS", "fred", start=start, end=end)
        irx.to_pickle(src_data)

    rf = irx["2014":"2018"]
    rf = (1 + (rf / 100)) ** (1 / 12) - 1

    rf.plot(title="Risk-free rate (3-Month Treasury Bill)")
    plt.tight_layout()
    plt.savefig("images/ch4_im3.png")

    ## Implementing the Fama-French three-factor model in Python
    # download the zip file from Prof. French's website
    # wget http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip
    factor_df = pd.read_csv("data/F-F_Research_Data_Factors.csv", skiprows=3)

    # identify where the annual data starts
    RISKY_ASSET = "FB"
    START_DATE = "2013-12-31"
    END_DATE = "2018-12-31"

    STR_TO_MATCH = " Annual Factors: January-December "
    indices = factor_df.iloc[:, 0] == STR_TO_MATCH
    start_of_annual = factor_df[indices].index[0]

    # keep only monthly data
    factor_df = factor_df[factor_df.index < start_of_annual]

    # 5. Rename columns of the DataFrame, set a datetime index and filter by dates:
    factor_df.columns = ["date", "mkt", "smb", "hml", "rf"]
    factor_df["date"] = pd.to_datetime(factor_df["date"], format="%Y%m").dt.strftime("%Y-%m")
    factor_df = factor_df.set_index("date")
    factor_df = factor_df.loc[START_DATE:END_DATE]

    # 6. Convert the values to numeric and divide by 100:
    factor_df = factor_df.apply(pd.to_numeric, errors="coerce").div(100)
    ic(factor_df.head())

    # 7. Download the prices of the risky asset:
    src_data = "data/yf_fbook.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download("FB", start=start, end=end, adjusted=True, progress=False)
        data.to_pickle(src_data)
    asset_df = data["2014":"2018"]
    print(f"Downloaded {asset_df.shape[0]} rows of data.")

    # 8. Calculate monthly returns on the risky asset:
    y = asset_df["Adj Close"].resample("M").last().pct_change().dropna()
    y.index = y.index.strftime("%Y-%m")
    y.name = "rtn"
    ic(y.head())
    # 9. Merge the datasets and calculate excess returns:
    ff_data = factor_df.join(y)
    ff_data["excess_rtn"] = ff_data.rtn - ff_data.rf
    # 10. Estimate the three-factor model:
    ff_model = smf.ols(formula="excess_rtn ~ mkt + smb + hml", data=ff_data).fit()
    print(ff_model.summary())

    # ic(get_available_datasets()[:5])
    # ff_dict = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=start)
    # ic(ff_dict.keys())
    # ic(ff_dict["DESCR"])
    # ic(ff_dict[0].head())

    ## Implementing the rolling three-factor model on a portfolio of assets
    ASSETS = ["AMZN", "GOOG", "AAPL", "MSFT"]
    WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    src_data = "data/pd_ff_factor.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=start)[0]
        data.to_pickle(src_data)
    df_three_factor = data["2010":"2018"]
    df_three_factor = df_three_factor.div(100)
    df_three_factor.index = df_three_factor.index.format()

    src_data = "data/yf_assets.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download(ASSETS, start=start, end=end, adjusted=True, progress=False)
        data.to_pickle(src_data)
    asset_df = data["2010":"2018"]
    print(f"Downloaded {asset_df.shape[0]} rows of data.")

    # 5. Calculate the monthly returns on the risky assets:
    asset_df = asset_df["Adj Close"].resample("M").last().pct_change().dropna()
    # reformat index for joining
    asset_df.index = asset_df.index.strftime("%Y-%m")
    # 6. Calculate the portfolio returns:
    asset_df["portfolio_returns"] = np.matmul(asset_df[ASSETS].values, WEIGHTS)
    ic(asset_df.head())

    asset_df.plot()

    ff_data = asset_df.join(df_three_factor).drop(ASSETS, axis=1)
    ff_data.columns = ["portf_rtn", "mkt", "smb", "hml", "rf"]
    ff_data["portf_ex_rtn"] = ff_data.portf_rtn - ff_data.rf
    ic(ff_data.head())

    def rolling_factor_model(input_data, formula, window_size):
        """
        Function for estimating the Fama-French (n-factor) model using a rolling window of fixed size.
        Parameters
        ------------
        input_data : pd.DataFrame
            A DataFrame containing the factors and asset/portfolio returns
        formula : str
            `statsmodels` compatible formula representing the OLS regression
        window_size : int
            Rolling window length.
        Returns
        -----------
        coeffs_df : pd.DataFrame
            DataFrame containing the intercept and the three factors for each iteration.
        """

        coeffs = []
        for start_index in range(len(input_data) - window_size + 1):
            end_index = start_index + window_size
            # define and fit the regression model
            ff_model = smf.ols(formula=formula, data=input_data[start_index:end_index]).fit()
            # store coefficients
            coeffs.append(ff_model.params)
        coeffs_df = pd.DataFrame(coeffs, index=input_data.index[window_size - 1 :])
        return coeffs_df

    # 9. Estimate the rolling three-factor model and plot the results:
    MODEL_FORMULA = "portf_ex_rtn ~ mkt + smb + hml"
    results_df = rolling_factor_model(ff_data, MODEL_FORMULA, window_size=60)
    results_df.plot(title="Rolling Fama-French Three-Factor model")

    plt.clf()
    plt.tight_layout()
    plt.savefig("images/ch4_im8.png")

    ## Implementing the four- and five-factor models in Python
    src_data = "data/pd_ff_factor.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=start)[0]
        data.to_pickle(src_data)
    df_three_factor = data["2014":"2018"]
    df_three_factor.index = df_three_factor.index.format()

    src_data = "data/ff_momentum.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = web.DataReader("F-F_Momentum_Factor", "famafrench", start=start)[0]
        data.to_pickle(src_data)
    df_mom = data["2014":"2018"]
    df_mom.index = df_mom.index.format()

    src_data = "data/ff_5_factor.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start)[0]
        data.to_pickle(src_data)
    df_five_factor = data["2014":"2018"]
    df_five_factor.index = df_five_factor.index.format()

    src_data = "data/yf_amzn.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download("AMZN", start=start, end=end, adjusted=True, progress=False)
        data.to_pickle(src_data)
    asset_df = data["2014":"2018"]
    ic(asset_df.head())

    y = asset_df["Adj Close"].resample("M").last().pct_change().dropna()
    y.index = y.index.strftime("%Y-%m")
    y.name = "return"

    four_factor_data = df_three_factor.join(df_mom).join(y)
    four_factor_data.columns = ["mkt", "smb", "hml", "rf", "mom", "rtn"]
    # divide everything (except returns) by 100
    four_factor_data.loc[:, four_factor_data.columns != "rtn"] /= 100
    # convert index to datetime
    four_factor_data.index = [pd.to_datetime(x, format="%Y-%m") for x in four_factor_data.index]
    # select period of interest
    four_factor_data = four_factor_data.loc[START_DATE:END_DATE]
    # calculate excess returns
    four_factor_data["excess_rtn"] = four_factor_data.rtn - four_factor_data.rf
    ic(four_factor_data.head())

    # join all datasets on the index
    five_factor_data = df_five_factor.join(y)
    five_factor_data.columns = ["mkt", "smb", "hml", "rmw", "cma", "rf", "rtn"]
    # divide everything (except returns) by 100
    five_factor_data.loc[:, five_factor_data.columns != "rtn"] /= 100
    # convert index to datetime
    five_factor_data.index = [pd.to_datetime(x, format="%Y-%m") for x in five_factor_data.index]
    # select period of interest
    five_factor_data = five_factor_data.loc[START_DATE:END_DATE]
    # calculate excess returns
    five_factor_data["excess_rtn"] = five_factor_data.rtn - five_factor_data.rf
    five_factor_data.head()

    # 8. Estimate the four-factor model:
    four_factor_model = smf.ols(
        formula="excess_rtn ~ mkt + smb + hml + mom", data=four_factor_data
    ).fit()
    print(four_factor_model.summary())

    # 9. Estimate the five-factor model:
    five_factor_model = smf.ols(
        formula="excess_rtn ~ mkt + smb + hml + rmw + cma", data=five_factor_data
    ).fit()
    print(five_factor_model.summary())
