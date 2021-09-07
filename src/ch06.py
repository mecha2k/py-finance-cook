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
import QuantLib as ql

from arch import arch_model
from scipy.stats import norm
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


def simulate_gbm(s_0, mu, sigma, n_sims, T, N, random_seed=42, antithetic_var=False):
    """
    Function used for simulating stock returns using Geometric Brownian Motion.
    Parameters
    ------------
    s_0 : float
        Initial stock price
    mu : float
        Drift coefficient
    sigma : float
        Diffusion coefficient
    n_sims : int
        Number of simulations paths
    T : float
        Length of the forecast horizon, same unit as dt
    N : int
        Number of time increments in the forecast horizon
    random_seed : int
        Random seed for reproducibility
    antithetic_var : bool
        Boolean whether to use antithetic variates approach to reduce variance
    Returns
    -----------
    S_t : np.ndarray
        Matrix (size: n_sims x (T+1)) containing the simulation results.
        Rows respresent sample paths, while columns point of time.
    """

    np.random.seed(random_seed)
    dt = T / N
    # Brownian
    if antithetic_var:
        dW_ant = np.random.normal(scale=np.sqrt(dt), size=(int(n_sims / 2), N + 1))
        dW = np.concatenate((dW_ant, -dW_ant), axis=0)
    else:
        dW = np.random.normal(scale=np.sqrt(dt), size=(n_sims, N + 1))
    # simulate the evolution of the process
    S_t = s_0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * dW, axis=1))
    S_t[:, 0] = s_0
    return S_t


def black_scholes_analytical(S_0, K, T, r, sigma, type_="call"):
    """
    Function used for calculating the price of European options
    using the analytical form of the Black-Scholes model.
    Parameters
    ------------
    S_0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Annualized risk-free rate
    sigma : float
        Standard deviation of the stock returns
    type_ : str
        Type of the option. Allowable: ['call', 'put']
    Returns
    -----------
    option_premium : float
        The premium on the option calculated using the Black-Scholes model
    """
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if type_ == "call":
        val = S_0 * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif type_ == "put":
        val = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S_0 * norm.cdf(-d1, 0, 1)
    else:
        raise ValueError("Wrong input for type!")
    return val


def gaussian_brownian_motion():
    src_data = "data/yf_msft.pkl"
    start = datetime(2000, 1, 1)
    end = datetime(2020, 12, 31)
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download("MSFT", start=start, end=end, adjusted=True, progress=False)
        data.to_pickle(src_data)
    df = data["2019-1":"2019-7"]

    adj_close = df["Adj Close"]
    returns = adj_close.pct_change().dropna()
    print(f"Average return: {100 * returns.mean():.2f}%")

    ax = returns.plot()
    ax.set_title(f"MSFT returns: 2019-1 ~ 2019-7", fontsize=16)
    plt.tight_layout()
    plt.savefig("images/ch6_im1.png")

    train = returns["2019-01-01":"2019-06-30"]
    test = returns["2019-07-01":"2019-07-31"]

    T = len(test)
    N = len(test)
    S_0 = adj_close.loc[train.index[-1]]
    N_SIM = 100
    mu = train.mean()
    sigma = train.std()

    gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T, N)

    last_train_date = train.index[-1].date()
    first_test_date = test.index[0].date()
    last_test_date = test.index[-1].date()
    plot_title = f"MSFT Simulation " f"({first_test_date}:{last_test_date})"

    selected_indices = adj_close[last_train_date:last_test_date].index
    index = [date.date() for date in selected_indices]
    gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations), index=index)

    ax = gbm_simulations_df.plot(alpha=0.2, legend=False)
    (line_1,) = ax.plot(index, gbm_simulations_df.mean(axis=1), color="red")
    (line_2,) = ax.plot(index, adj_close[last_train_date:last_test_date], color="blue")
    ax.set_title(plot_title, fontsize=16)
    ax.legend((line_1, line_2), ("mean", "actual"))
    plt.tight_layout()
    plt.savefig("images/ch6_im2.png")
    plt.close()


def european_option_simulation(S_0, K, T, r, sigma, n_sims, type_="call", random_seed=42):
    """
    Function used for calculating the price of European options using Monte Carlo simulations.
    Parameters
    ------------
    S_0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Annualized risk-free rate
    sigma : float
        Standard deviation of the stock returns
    n_sims : int
        Number of paths to simulate
    type_ : str
        Type of the option. Allowable: ['call', 'put']
    random_seed : int
        Random seed for reproducibility
    Returns
    -----------
    option_premium : float
        The premium on the option calculated using Monte Carlo simulations
    """
    np.random.seed(random_seed)
    rv = np.random.normal(0, 1, size=n_sims)
    S_T = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * rv)
    if type_ == "call":
        payoff = np.maximum(0, S_T - K)
    elif type_ == "put":
        payoff = np.maximum(0, K - S_T)
    else:
        raise ValueError("Wrong input for type!")
    premium = np.mean(payoff) * np.exp(-r * T)
    return premium


def lsmc_american_option(S_0, K, T, N, r, sigma, n_sims, option_type, poly_degree, random_seed=42):
    """
    Function used for calculating the price of American options using Least Squares Monte Carlo
    algorithm of Longstaff and Schwartz (2001).
    Parameters
    ------------
    S_0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    N : int
        Number of time increments in the forecast horizon
    r : float
        Annualized risk-free rate
    sigma : float
        Standard deviation of the stock returns
    n_sims : int
        Number of paths to simulate
    option_type : str
        Type of the option. Allowable: ['call', 'put']
    poly_degree : int
        Degree of the polynomial to fit in the LSMC algorithm
    random_seed : int
        Random seed for reproducibility
    Returns
    -----------
    option_premium : float
        The premium on the option
    """
    dt = T / N
    discount_factor = np.exp(-r * dt)
    gbm_simulations = simulate_gbm(
        s_0=S_0, mu=r, sigma=sigma, n_sims=n_sims, T=T, N=N, random_seed=random_seed
    )
    if option_type == "call":
        payoff_matrix = np.maximum(gbm_simulations - K, np.zeros_like(gbm_simulations))
    elif option_type == "put":
        payoff_matrix = np.maximum(K - gbm_simulations, np.zeros_like(gbm_simulations))
    else:
        payoff_matrix = None
    value_matrix = np.zeros_like(payoff_matrix)
    value_matrix[:, -1] = payoff_matrix[:, -1]

    for t in range(N - 1, 0, -1):
        regression = np.polyfit(
            gbm_simulations[:, t], value_matrix[:, t + 1] * discount_factor, poly_degree
        )
        continuation_value = np.polyval(regression, gbm_simulations[:, t])
        value_matrix[:, t] = np.where(
            payoff_matrix[:, t] > continuation_value,
            payoff_matrix[:, t],
            value_matrix[:, t + 1] * discount_factor,
        )
    option_premium = np.mean(value_matrix[:, 1] * discount_factor)
    return option_premium


def american_options_montecarlo():
    S_0 = 36
    K = 40
    r = 0.06
    sigma = 0.2
    T = 1  # 1 year
    N = 50
    dt = T / N
    N_SIMS = 10 ** 5
    discount_factor = np.exp(-r * dt)
    OPTION_TYPE = "put"
    POLY_DEGREE = 5

    gbm_sims = simulate_gbm(s_0=S_0, mu=r, sigma=sigma, n_sims=N_SIMS, T=T, N=N)
    payoff_matrix = np.maximum(K - gbm_sims, np.zeros_like(gbm_sims))

    # 5. Define the value matrix and fill in the last column (time T)
    value_matrix = np.zeros_like(payoff_matrix)
    value_matrix[:, -1] = payoff_matrix[:, -1]
    # 6. Iteratively calculate the continuation value and the value vector in the given time
    for t in range(N - 1, 0, -1):
        regression = np.polyfit(
            gbm_sims[:, t], value_matrix[:, t + 1] * discount_factor, POLY_DEGREE
        )
        continuation_value = np.polyval(regression, gbm_sims[:, t])
        value_matrix[:, t] = np.where(
            payoff_matrix[:, t] > continuation_value,
            payoff_matrix[:, t],
            value_matrix[:, t + 1] * discount_factor,
        )
    # 7. Calculate the option premium:
    option_premium = np.mean(value_matrix[:, 1] * discount_factor)
    print(f"The premium on the specified American {OPTION_TYPE} option is {option_premium:.3f}")
    # 8. Calculate the premium of a European put with the same parameters:
    black_scholes_analytical(S_0=S_0, K=K, T=T, r=r, sigma=sigma, type_="put")
    # 9. As an extra check, calculate the prices of the American and European call options:
    european_call_price = black_scholes_analytical(S_0=S_0, K=K, T=T, r=r, sigma=sigma)
    american_call_price = lsmc_american_option(
        S_0=S_0,
        K=K,
        T=T,
        N=N,
        r=r,
        sigma=sigma,
        n_sims=N_SIMS,
        option_type="call",
        poly_degree=POLY_DEGREE,
    )
    print(
        f"The price of the European call is {european_call_price:.3f}, "
        f"and the American call's price (using {N_SIMS} simulations) is {american_call_price:.3f}"
    )
    ic(european_option_simulation(S_0, K, T, r, sigma, N_SIMS, type_="put"))


def american_options_quantlib():
    S_0 = 36
    r = 0.06
    sigma = 0.2
    K = 40
    OPTION_TYPE = "put"
    POLY_DEGREE = 5
    R_SEED = 42
    N_SIMS = 10 ** 5
    N = 50

    # 2. Specify the calendar and the day counting convention:
    calendar = ql.UnitedStates()
    day_counter = ql.ActualActual()
    # 3. Specify the valuation date and the expiry date of the option:
    valuation_date = ql.Date(1, 1, 2018)
    expiry_date = ql.Date(1, 1, 2019)
    ql.Settings.instance().evaluationDate = valuation_date
    # 4. Define the option type (call/put), type of exercise and the payoff:
    if OPTION_TYPE == "call":
        option_type_ql = ql.Option.Call
    elif OPTION_TYPE == "put":
        option_type_ql = ql.Option.Put
    else:
        option_type_ql = None
    exercise = ql.AmericanExercise(valuation_date, expiry_date)
    payoff = ql.PlainVanillaPayoff(option_type_ql, K)

    # 5. Prepare the market-related data:
    u = ql.SimpleQuote(S_0)
    r = ql.SimpleQuote(r)
    sigma = ql.SimpleQuote(sigma)
    # 6. Specify the market-related curves:
    # volatility = ql.BlackConstantVol(valuation_date, calendar, sigma, day_counter)
    # risk_free_rate = ql.FlatForward(valuation_date, r, day_counter)
    underlying = ql.QuoteHandle(u)
    volatility = ql.BlackConstantVol(0, ql.TARGET(), ql.QuoteHandle(sigma), day_counter)
    risk_free_rate = ql.FlatForward(0, ql.TARGET(), ql.QuoteHandle(r), day_counter)
    # 7. Plug in the market-related data into the BS process:
    bs_process = ql.BlackScholesProcess(
        underlying,
        ql.YieldTermStructureHandle(risk_free_rate),
        ql.BlackVolTermStructureHandle(volatility),
    )
    # 8. Instantiate the Monte Carlo engine for the American options:
    engine = ql.MCAmericanEngine(
        bs_process,
        "PseudoRandom",
        timeSteps=N,
        polynomOrder=POLY_DEGREE,
        seedCalibration=R_SEED,
        requiredSamples=N_SIMS,
    )
    # 9. Instantiate the `option` object and set its pricing engine:
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    # 10. Calculate the option premium:
    option_premium_ql = option.NPV()
    print(f"The value of the American {OPTION_TYPE} option is: {option_premium_ql:.3f}")

    u_0 = u.value()  # original value
    h = 0.01
    u.setValue(u_0 + h)
    P_plus_h = option.NPV()
    u.setValue(u_0 - h)
    P_minus_h = option.NPV()
    u.setValue(u_0)  # set back to the original value

    delta = (P_plus_h - P_minus_h) / (2 * h)
    print(f"Delta of the option: {delta:.2f}")


def value_at_risk():
    np.random.seed(42)
    risky_assets = ["GOOG", "FB"]
    SHARES = [5, 5]
    T = 1
    N_SIMS = 10 ** 5

    src_data = "data/yf_assets_c06_1.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download(risky_assets, start=start, end=end, adjusted=True, progress=False)
        data.to_pickle(src_data)
    df = data["2018-1":"2018-12"]
    print(f"Downloaded {df.shape[0]} rows of data.")
    ic(df.head())

    adj_close = df["Adj Close"]
    returns = adj_close.pct_change().dropna()
    plot_title = f'{" vs. ".join(risky_assets)} returns: 2018-01 - 2018-12'
    returns.plot(title=plot_title)
    plt.tight_layout()
    plt.savefig("images/ch6_im3.png")
    print(f"Correlation between returns: {returns.corr().values[0,1]:.2f}")

    cov_mat = returns.cov()
    ic(cov_mat)

    # 6. Perform the Cholesky decomposition of the covariance matrix:
    chol_mat = np.linalg.cholesky(cov_mat)
    ic(chol_mat)
    # 7. Draw correlated random numbers from Standard Normal distribution:
    rv = np.random.normal(size=(N_SIMS, len(risky_assets)))
    correlated_rv = np.transpose(np.matmul(chol_mat, np.transpose(rv)))

    r = np.mean(returns, axis=0).values
    sigma = np.std(returns, axis=0).values
    S_0 = adj_close.values[-1, :]
    P_0 = np.sum(SHARES * S_0)
    # 9. Calculate the terminal price of the considered stocks:
    S_T = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * correlated_rv)
    # 10. Calculate the terminal portfolio value and calculate the portfolio returns:
    P_T = np.sum(SHARES * S_T, axis=1)
    P_diff = P_T - P_0
    # 11. Calculate VaR:
    P_diff_sorted = np.sort(P_diff)
    percentiles = [0.01, 0.1, 1.0]
    var = np.percentile(P_diff_sorted, percentiles)
    for x, y in zip(percentiles, var):
        print(f"1-day VaR with {100-x}% confidence: {-y:.2f}$")

    ax = sns.distplot(P_diff, kde=False)
    ax.set_title(
        """Distribution of possible 1-day changes in portfolio value
                 1-day 99% VaR""",
        fontsize=16,
    )
    ax.axvline(var[2], 0, 10000)
    plt.tight_layout()
    plt.savefig("images/ch6_im4.png")

    var = np.percentile(P_diff_sorted, 5)
    expected_shortfall = P_diff_sorted[P_diff_sorted <= var].mean()
    print(
        f"The 1-day 95% VaR is {-var:.2f}$, and the accompanying Expected Shortfall is {-expected_shortfall:.2f}$."
    )


if __name__ == "__main__":
    gaussian_brownian_motion()

    ## Pricing European Options using Simulations
    S_0 = 100
    K = 100
    r = 0.05
    sigma = 0.50
    T = 1  # 1 year
    N = 252  # 252 days in a year
    dt = T / N  # time step
    N_SIMS = 1000000  # number of simulations
    discount_factor = np.exp(-r * T)

    # 4. Valuate the call option using the specified parameters:
    ic(black_scholes_analytical(S_0=S_0, K=K, T=T, r=r, sigma=sigma, type_="call"))
    # 5. Simulate the stock path using GBM:
    gbm_sims = simulate_gbm(s_0=S_0, mu=r, sigma=sigma, n_sims=N_SIMS, T=T, N=N)
    # 6. Calculate the option premium:
    premium = discount_factor * np.mean(np.maximum(0, gbm_sims[:, -1] - K))
    ic(premium)
    ic(black_scholes_analytical(S_0=S_0, K=K, T=T, r=r, sigma=sigma, type_="put"))

    ## Pricing American Options with Least Squares Monte Carlo
    american_options_montecarlo()

    ## Pricing American Options using Quantlib
    american_options_quantlib()

    ## Estimating Value-at-risk using Monte Carlo
    value_at_risk()
