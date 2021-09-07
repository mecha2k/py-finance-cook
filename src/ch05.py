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

from arch import arch_model
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
    src_data = "data/yf_google.pkl"
    start = datetime(2000, 1, 1)
    end = datetime(2020, 12, 31)
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download("GOOG", start=start, end=end, adjusted=True, progress=False)
        data.to_pickle(src_data)
    df = data["2015":"2018"]

    returns = 100 * df["Adj Close"].pct_change().dropna()
    returns.name = "asset_returns"
    print(f"Average return: {round(returns.mean(), 2)}%")

    returns.plot(title=f"GOOG returns: 2015-01 - 2018-12")
    plt.tight_layout()
    plt.savefig("images/ch5_im1.png")

    model = arch_model(returns, mean="Zero", vol="ARCH", p=1, o=0, q=0)
    model_fitted = model.fit(disp="off")
    print(model_fitted.summary())

    model_fitted.plot(annualize="D")
    plt.tight_layout()
    plt.savefig("images/ch5_im3.png")

    ## Explaining stock returns' volatility with GARCH models
    model = arch_model(returns, mean="Zero", vol="GARCH", p=1, o=0, q=1)
    model_fitted = model.fit(disp="off")
    print(model_fitted.summary())

    model_fitted.plot(annualize="D")
    plt.tight_layout()
    plt.savefig("images/ch5_im5.png")

    ## Implementing CCC-GARCH model for multivariate volatility forecasting
    RISKY_ASSETS = ["GOOG", "MSFT", "AAPL"]
    N = len(RISKY_ASSETS)

    src_data = "data/yf_assets_1.pkl"
    start = datetime(2000, 1, 1)
    end = datetime(2020, 12, 31)
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download(RISKY_ASSETS, start=start, end=end, adjusted=True, progress=False)
        data.to_pickle(src_data)
    df = data["2015":"2018"]

    returns = 100 * df["Adj Close"].pct_change().dropna()
    returns.plot(subplots=True, title=f"Stock returns: 2015-01 - 2018-12")
    plt.tight_layout()
    plt.savefig("images/ch5_im6.png")

    coeffs = []
    cond_vol = []
    std_resids = []
    models = []

    for asset in returns.columns:
        model = arch_model(returns[asset], mean="Constant", vol="GARCH", p=1, o=0, q=1).fit(
            update_freq=0, disp="off"
        )
        coeffs.append(model.params)
        cond_vol.append(model.conditional_volatility)
        std_resids.append(model.resid / model.conditional_volatility)
        models.append(model)

    coeffs_df = pd.DataFrame(coeffs, index=returns.columns)
    cond_vol_df = (
        pd.DataFrame(cond_vol).transpose().set_axis(returns.columns, axis="columns", inplace=False)
    )
    std_resids_df = (
        pd.DataFrame(std_resids)
        .transpose()
        .set_axis(returns.columns, axis="columns", inplace=False)
    )
    ic(coeffs_df)

    # 8. Calculate the constant conditional correlation matrix (R):
    R = std_resids_df.transpose().dot(std_resids_df).div(len(std_resids_df))
    # 9. Calculate the 1-step ahead forecast of the conditional covariance matrix :
    diag = []
    D = np.zeros((N, N))
    # populate the list with conditional variances
    for model in models:
        diag.append(model.forecast(horizon=1).variance.values[-1][0])
    # take the square root to obtain volatility from variance
    diag = np.sqrt(np.array(diag))
    # fill the diagonal of D with values from diag
    np.fill_diagonal(D, diag)
    # calculate the conditional covariance matrix
    H = np.matmul(np.matmul(D, R.values), D)
    ic(H)

    ## Forecasting the conditional covariance matrix using DCC-GARCH
