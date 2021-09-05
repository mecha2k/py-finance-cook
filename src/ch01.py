import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import quandl
import intrinio_sdk as intrinio
import os
import warnings
from icecream import ic
from dotenv import load_dotenv
from datetime import datetime

import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode

import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt

cf.set_config_file(world_readable=True, theme="pearl", offline=True)
# init_notebook_mode()

load_dotenv(verbose=True)
quandl.ApiConfig.api_key = os.getenv("Quandl")

plt.style.use("seaborn")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = [8, 5]
warnings.simplefilter(action="ignore", category=FutureWarning)

# 2. Authenticate using the personal API key and select the API:
# intrinio.ApiClient().set_api_key(os.getenv("Intrinio"))
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


if __name__ == "__main__":
    src_data = "data/yf_aapl.csv"
    cpi_data = "data/qn_cpi.csv"
    sp500_data = "data/yf_sp500.csv"
    start = datetime(2000, 1, 1)
    end = datetime(2018, 12, 31)
    try:
        aapl = pd.read_csv(src_data, parse_dates=["Date"])
        cpi = pd.read_csv(cpi_data, parse_dates=["Date"])
        sp500 = pd.read_csv(sp500_data, parse_dates=["Date"])
        aapl = aapl.set_index("Date")
        cpi = cpi.set_index("Date")
        sp500.set_index("Date", inplace=True)
        print("data reading from file...")
    except FileNotFoundError:
        aapl = yf.download("aapl", start=start, end=end)
        sp500 = yf.download("^GSPC", start=start, end=end, progress=False)
        cpi = quandl.get(dataset="RATEINF/CPI_USA", start_date=start, end_date=end)
        cpi.rename(columns={"Value": "cpi"}, inplace=True)
        aapl.to_csv(src_data)
        cpi.to_csv(cpi_data)
        sp500.to_csv(sp500_data)
    ic(aapl.head())
    ic(cpi.head())
    ic(sp500.head())
    ic(f"{cpi.shape[0]} rows of data downloaded.")

    df = aapl.loc[:, ["Adj Close"]]
    df.rename(columns={"Adj Close": "adj_close"}, inplace=True)
    df["simple_rtn"] = df.adj_close.pct_change()
    df["log_rtn"] = np.log(df.adj_close / df.adj_close.shift(1))
    ic(df.head())

    df_all_dates = pd.DataFrame(index=pd.date_range(start=start, end=end))
    ic(df_all_dates.join(df[["adj_close"]], how="left"))
    ic(df_all_dates.join(df[["adj_close"]], how="left").fillna(method="ffill"))
    ic(df_all_dates.join(df[["adj_close"]], how="left").fillna(method="ffill").asfreq("M"))
    df = df_all_dates.join(df[["adj_close"]], how="left").fillna(method="ffill").asfreq("M")
    ic(df)

    # 3. Download inflation data from Quandl:
    # df_cpi = quandl.get(dataset="RATEINF/CPI_USA", start_date="1999-12-01", end_date="2010-12-31")
    # df_cpi.rename(columns={"Value": "cpi"}, inplace=True)
    df_cpi = cpi

    # 4. Merge inflation data to prices:
    df_merged = df.join(df_cpi, how="left")

    # 5. Calculate simple returns and inflation rate:
    df_merged["simple_rtn"] = df_merged.adj_close.pct_change()
    df_merged["inflation_rate"] = df_merged.cpi.pct_change()

    # 6. Adjust returns for inflation:
    df_merged["real_rtn"] = (df_merged.simple_rtn + 1) / (df_merged.inflation_rate + 1) - 1
    ic(df_merged.head())

    ## Changing frequency
    # df = yf.download("AAPL", start="2000-01-01", end="2010-12-31", auto_adjust=False, progress=False)
    df = aapl

    # keep only the adjusted close price
    df = df.loc[:, ["Adj Close"]]
    df.rename(columns={"Adj Close": "adj_close"}, inplace=True)

    # calculate simple returns
    df["log_rtn"] = np.log(df.adj_close / df.adj_close.shift(1))

    # remove redundant data
    df.drop("adj_close", axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    ic(df.head())

    def realized_volatility(x):
        return np.sqrt(np.sum(x ** 2))

    # 3. Calculate monthly realized volatility:
    ic(df.groupby(pd.Grouper(freq="M")))
    ic(df.resample("M").mean())
    ic(df.resample("M").mean().apply(realized_volatility))
    df_rv = df.groupby(pd.Grouper(freq="M")).apply(realized_volatility)
    df_rv.rename(columns={"log_rtn": "rv"}, inplace=True)
    ic(df_rv)

    # 4. Annualize the values:
    df_rv.rv = df_rv.rv * np.sqrt(12)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(df)
    ax[1].plot(df_rv)
    plt.tight_layout()
    plt.savefig("images/ch1_im6.png")

    # download data as pandas DataFrame
    # df = yf.download("MSFT", auto_adjust=False, progress=False)
    df = aapl
    df = df.loc[:, ["Adj Close"]]
    df.rename(columns={"Adj Close": "adj_close"}, inplace=True)

    # create simple and log returns
    df["simple_rtn"] = df.adj_close.pct_change()
    df["log_rtn"] = np.log(df.adj_close / df.adj_close.shift(1))

    # dropping NA's in the first row
    df.dropna(how="any", inplace=True)
    df.info()

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    df.adj_close.plot(ax=ax[0])
    ax[0].set(title="MSFT time series", ylabel="Stock price ($)")
    df.simple_rtn.plot(ax=ax[1])
    ax[1].set(ylabel="Simple returns (%)")
    df.log_rtn.plot(ax=ax[2])
    ax[2].set(xlabel="Date", ylabel="Log returns (%)")
    ax[2].tick_params(axis="x", which="major", labelsize=12)
    plt.tight_layout()
    plt.savefig("images/ch1_im7.png")

    # df.iplot(subplots=True, shape=(3, 1), shared_xaxes=True, title="MSFT time series")

    ## Identifying outliers
    df = aapl
    df = df.loc[:, ["Adj Close"]]
    df.rename(columns={"Adj Close": "adj_close"}, inplace=True)
    df["simple_rtn"] = df.adj_close.pct_change()
    ic(df.head())

    df_rolling = df[["simple_rtn"]].rolling(window=21).agg(["mean", "std"])
    ic(df_rolling)
    ic(df_rolling.columns)
    df_rolling.columns = df_rolling.columns.droplevel(0)
    ic(df_rolling)
    ic(df_rolling.columns)
    df_outliers = df.join(df_rolling)
    ic(df_outliers)

    def indentify_outliers(row, n_sigmas=3):
        x = row["simple_rtn"]
        mu = row["mean"]
        sigma = row["std"]
        if (x > mu + n_sigmas * sigma) | (x < mu - n_sigmas * sigma):
            return 1
        else:
            return 0

    df_outliers["outlier"] = df_outliers.apply(indentify_outliers, axis=1)
    outliers = df_outliers.loc[df_outliers["outlier"] == 1, ["simple_rtn"]]
    ic(outliers)

    fig, ax = plt.subplots()
    ax.plot(df_outliers.index, df_outliers.simple_rtn, color="blue", label="Normal")
    ax.scatter(outliers.index, outliers.simple_rtn, color="red", label="Anomaly")
    ax.set_title("Apple's stock returns")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("images/ch1_im9.png")
    plt.close()

    ## Investigating stylized facts of asset returns
    df = sp500
    df = df[["Adj Close"]].rename(columns={"Adj Close": "adj_close"})
    df["log_rtn"] = np.log(df.adj_close / df.adj_close.shift(1))
    df = df[["adj_close", "log_rtn"]].dropna(how="any")

    #### Fact 1 - Non-Gaussian distribution of returns
    # 1. Calculate the Normal PDF using the mean and standard deviation of the observed returns:
    r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
    mu = df.log_rtn.mean()
    sigma = df.log_rtn.std()
    norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])
    ax[0].set_title("Distribution of S&P 500 returns", fontsize=16)
    ax[0].plot(r_range, norm_pdf, "g", lw=2, label=f"N({mu:.2f}, {sigma**2:.4f})")
    ax[0].legend(loc="upper left")

    # Q-Q plot
    qq = sm.qqplot(df.log_rtn.values, line="s", ax=ax[1])
    ax[1].set_title("Q-Q plot", fontsize=16)
    plt.tight_layout()
    plt.savefig("images/ch1_im10.png")
    plt.close()

    jb_test = scs.jarque_bera(df.log_rtn.values)

    print("---------- Descriptive Statistics ----------")
    print("Range of dates:", min(df.index.date), "-", max(df.index.date))
    print("Number of observations:", df.shape[0])
    print(f"Mean: {df.log_rtn.mean():.4f}")
    print(f"Median: {df.log_rtn.median():.4f}")
    print(f"Min: {df.log_rtn.min():.4f}")
    print(f"Max: {df.log_rtn.max():.4f}")
    print(f"Standard Deviation: {df.log_rtn.std():.4f}")
    print(f"Skewness: {df.log_rtn.skew():.4f}")
    print(f"Kurtosis: {df.log_rtn.kurtosis():.4f}")
    print(f"Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}")

    #### Fact 2 - Volatility Clustering
    df.log_rtn.plot(title="Daily S&P 500 returns", figsize=(10, 6))
    plt.tight_layout()
    plt.savefig("images/ch1_im12.png")

    #### Fact 3 - Absence of autocorrelation in returns
    N_LAGS = 50
    SIGNIFICANCE_LEVEL = 0.05
    acf = smt.graphics.plot_acf(df.log_rtn, lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL)
    plt.tight_layout()
    plt.savefig("images/ch1_im13.png")

    #### Fact 4 - Small and decreasing autocorrelation in squared/absolute returns
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    smt.graphics.plot_acf(df.log_rtn ** 2, lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL, ax=ax[0])
    ax[0].set(title="Autocorrelation Plots", ylabel="Squared Returns")
    smt.graphics.plot_acf(np.abs(df.log_rtn), lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL, ax=ax[1])
    ax[1].set(ylabel="Absolute Returns", xlabel="Lag")
    plt.tight_layout()
    plt.savefig("images/ch1_im14.png")
    plt.close()

    #### Fact 5 - Leverage effect
    df["moving_std_252"] = df[["log_rtn"]].rolling(window=252).std()
    df["moving_std_21"] = df[["log_rtn"]].rolling(window=21).std()

    fig, ax = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
    df.adj_close.plot(ax=ax[0])
    ax[0].set(title="S&P 500 time series", ylabel="Price ($)")
    df.log_rtn.plot(ax=ax[1])
    ax[1].set(ylabel="Log returns (%)")
    df.moving_std_252.plot(ax=ax[2], color="r", label="Moving Volatility 252d")
    df.moving_std_21.plot(ax=ax[2], color="g", label="Moving Volatility 21d")
    ax[2].set(ylabel="Moving Volatility", xlabel="Date")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig("images/ch1_im15.png")
    plt.close()

    src_data = "data/yf_sp500_vix.pkl"
    start = datetime(1985, 1, 1)
    end = datetime(2018, 12, 31)
    try:
        df = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        df = yf.download(["^GSPC", "^VIX"], start=start, end=end, progress=False)
        df.to_pickle(src_data)
    ic(df.head())
    ic(df.columns)
    ic(df.axes)
    ic(df.keys())

    df = df[["Adj Close"]]
    ic(df.head())
    df.columns = df.columns.droplevel(0)
    ic(df.head())

    df = df.rename(columns={"^GSPC": "sp500", "^VIX": "vix"})
    ic(df)

    df["log_rtn"] = np.log(df.sp500 / df.sp500.shift(1))
    df["vol_rtn"] = np.log(df.vix / df.vix.shift(1))
    df.dropna(how="any", axis=0, inplace=True)
    corr_coeff = df.log_rtn.corr(df.vol_rtn)
    ic(corr_coeff)

    ax = sns.regplot(x="log_rtn", y="vol_rtn", data=df, line_kws={"color": "red"})
    ax.set(
        title=f"S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})",
        ylabel="VIX log returns",
        xlabel="S&P 500 log returns",
    )
    plt.tight_layout()
    plt.savefig("images/ch1_im16.png")
    plt.close()
