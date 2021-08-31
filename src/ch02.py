import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import cufflinks as cf
import backtrader as bt

from datetime import datetime
from icecream import ic
from plotly.offline import iplot, init_notebook_mode

plt.style.use("default")
plt.style.use("seaborn-colorblind")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 2000
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    src_data = "data/yf_twtr.pkl"
    start = datetime(2010, 1, 1)
    end = datetime(2018, 12, 31)
    try:
        twtr = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        twtr = yf.download("TWTR", start=start, end=end, auto_adjust=True)
        twtr.to_pickle(src_data)

    df_twtr = twtr["2018-1-1":"2018-12-31"]
    df_twtr.info()
    ic(df_twtr)

    src_data = "data/yf_aapl.pkl"
    start = datetime(2010, 1, 1)
    end = datetime(2020, 12, 31)
    try:
        aapl = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        aapl = yf.download("AAPL", start=start, end=end, auto_adjust=True)
        aapl.to_pickle(src_data)

    # cf.go_offline()
    # init_notebook_mode()
    # cf.set_config_file(world_readable=True, theme="pearl", offline=True)

    # qf = cf.QuantFig(df_twtr, title="Twitter's Stock Price", legend="top", name="TWTR")
    # qf.add_volume()
    # qf.add_sma(periods=20, column="Close", color="red")
    # qf.add_ema(periods=20, color="green")
    # qf.iplot()

    ## Backtesting a Strategy Based on Simple Moving Average
    class SmaSignal(bt.Signal):
        params = (("period", 20),)

        def __init__(self):
            super().__init__()
            self.lines.signal = self.data - bt.ind.SMA(period=self.p.period)

    aapl_df = aapl["2018-1-1":"2018-12-31"]
    data = bt.feeds.PandasData(dataname=aapl_df)
    ic(aapl_df.head())

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    # cerebro.plot(iplot=False, volume=True, width=8, height=6)

    #### Strategy
    class SmaStrategy(bt.Strategy):
        params = (("ma_period", 20),)

        def __init__(self):
            # keep track of close price in the series
            self.data_close = self.datas[0].close
            # keep track of pending orders/buy price/buy commission
            self.order = None
            self.price = None
            self.comm = None
            # add a simple moving average indicator
            self.sma = bt.ind.SMA(self.datas[0], period=self.params.ma_period)

        def log(self, txt):
            dt = self.datas[0].datetime.date(0).isoformat()
            print(f"{dt}, {txt}")

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # order already submitted/accepted - no action required
                return
            # report executed order
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(
                        f"BUY EXECUTED --- Price: {order.executed.price:.2f}, "
                        f"Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}"
                    )
                    self.price = order.executed.price
                    self.comm = order.executed.comm
                else:
                    self.log(
                        f"SELL EXECUTED --- Price: {order.executed.price:.2f}, "
                        f"Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}"
                    )
            # report failed order
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log("Order Failed")
            # set no pending order
            self.order = None

        def notify_trade(self, trade):
            if not trade.isclosed:
                return
            self.log(f"OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")

        def next(self):
            # do nothing if an order is pending
            if self.order:
                return
            # check if there is already a position
            if not self.position:
                # buy condition
                if self.data_close[0] > self.sma[0]:
                    self.log(f"BUY CREATED --- Price: {self.data_close[0]:.2f}")
                    self.order = self.buy()
            else:
                # sell condition
                if self.data_close[0] < self.sma[0]:
                    self.log(f"SELL CREATED --- Price: {self.data_close[0]:.2f}")
                    self.order = self.sell()

    cerebro = bt.Cerebro(stdstats=False)

    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    cerebro.addstrategy(SmaStrategy)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    class SmaStrategy1(bt.Strategy):
        params = (("ma_period", 20),)

        def __init__(self):
            # keep track of close price in the series
            self.data_close = self.datas[0].close
            # keep track of pending orders
            self.order = None
            # add a simple moving average indicator
            self.sma = bt.ind.SMA(self.datas[0], period=self.params.ma_period)

        def log(self, txt):
            """Logging function"""
            dt = self.datas[0].datetime.date(0).isoformat()
            print(f"{dt}, {txt}")

        def notify_order(self, order):
            # set no pending order
            self.order = None

        def next(self):
            # do nothing if an order is pending
            if self.order:
                return
            # check if there is already a position
            if not self.position:
                # buy condition
                if self.data_close[0] > self.sma[0]:
                    self.order = self.buy()
            else:
                # sell condition
                if self.data_close[0] < self.sma[0]:
                    self.order = self.sell()

        def stop(self):
            self.log(
                f"(ma_period = {self.params.ma_period:2d}) --- Terminal Value: {self.broker.getvalue():.2f}"
            )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.optstrategy(SmaStrategy1, ma_period=range(10, 31))
    cerebro.broker.setcash(1000.0)
    cerebro.run(maxcpus=8)

    ## Calculating Bollinger Bands and testing a buy/sell strategy
    class BBand_Strategy(bt.Strategy):
        params = (
            ("period", 20),
            ("devfactor", 2.0),
        )

        def __init__(self):
            # keep track of close price in the series
            self.data_close = self.datas[0].close
            self.data_open = self.datas[0].open
            # keep track of pending orders/buy price/buy commission
            self.order = None
            self.price = None
            self.comm = None
            # add Bollinger Bands indicator and track the buy/sell signals
            self.b_band = bt.ind.BollingerBands(
                self.datas[0], period=self.p.period, devfactor=self.p.devfactor
            )
            self.buy_signal = bt.ind.CrossOver(self.datas[0], self.b_band.lines.bot)
            self.sell_signal = bt.ind.CrossOver(self.datas[0], self.b_band.lines.top)

        def log(self, txt):
            """Logging function"""
            dt = self.datas[0].datetime.date(0).isoformat()
            print(f"{dt}, {txt}")

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # order already submitted/accepted - no action required
                return
            # report executed order
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(
                        f"BUY EXECUTED --- Price: {order.executed.price:.2f}, "
                        f"Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}"
                    )
                    self.price = order.executed.price
                    self.comm = order.executed.comm
                else:
                    self.log(
                        f"SELL EXECUTED --- Price: {order.executed.price:.2f}, "
                        f"Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}"
                    )
            # report failed order
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log("Order Failed")
            # set no pending order
            self.order = None

        def notify_trade(self, trade):
            if not trade.isclosed:
                return
            self.log(f"OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")

        def next_open(self):
            if not self.position:
                if self.buy_signal > 0:
                    # calculate the max number of shares ('all-in')
                    size = int(self.broker.getcash() / self.datas[0].open)
                    # buy order
                    self.log(
                        f"BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, "
                        f"Open: {self.data_open[0]}, Close: {self.data_close[0]}"
                    )
                    self.buy(size=size)
            else:
                if self.sell_signal < 0:
                    # sell order
                    self.log(f"SELL CREATED --- Size: {self.position.size}")
                    self.sell(size=self.position.size)

    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.addstrategy(BBand_Strategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    backtest_result = cerebro.run()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print(backtest_result[0].analyzers.returns.get_analysis())
    # cerebro.plot(iplot=False, volume=False)

    ic(backtest_result[0].analyzers)
    returns_dict = backtest_result[0].analyzers.time_return.get_analysis()
    returns_df = pd.DataFrame(
        list(returns_dict.items()), columns=["report_date", "return"]
    ).set_index("report_date")
    returns_df.plot(title="Portfolio returns")
    plt.tight_layout()
    plt.savefig("images/ch2_im9.png")

    ## Calculating the relative strength index and testing a long/short strategy
    # class RsiSignalStrategy(bt.SignalStrategy):
    #     params = dict(rsi_periods=14, rsi_upper=70, rsi_lower=30, rsi_mid=50)
    #
    #     def __init__(self):
    #         # add RSI indicator
    #         rsi = bt.indicators.RSI(
    #             period=self.p.rsi_periods, upperband=self.p.rsi_upper, lowerband=self.p.rsi_lower
    #         )
    #         # add RSI from TA-lib just for reference
    #         bt.talib.RSI(self.data, plotname="TA_RSI")
    #         # long condition (with exit)
    #         rsi_signal_long = bt.ind.CrossUp(rsi, self.p.rsi_lower, plot=False)
    #         self.signal_add(bt.SIGNAL_LONG, rsi_signal_long)
    #         self.signal_add(bt.SIGNAL_LONGEXIT, -(rsi > self.p.rsi_mid))
    #         # short condition (with exit)
    #         rsi_signal_short = -bt.ind.CrossDown(rsi, self.p.rsi_upper, plot=False)
    #         self.signal_add(bt.SIGNAL_SHORT, rsi_signal_short)
    #         self.signal_add(bt.SIGNAL_SHORTEXIT, rsi < self.p.rsi_mid)
    #
    # cerebro = bt.Cerebro(stdstats=False)
    # cerebro.addstrategy(RsiSignalStrategy)
    # cerebro.adddata(data)
    # cerebro.broker.setcash(1000.0)
    # cerebro.broker.setcommission(commission=0.001)
    # cerebro.addobserver(bt.observers.BuySell)
    # cerebro.addobserver(bt.observers.Value)
    # print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    # cerebro.run()
    # print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    # cerebro.plot(iplot=False, volume=False)

    ## Building an Interactive Dashboard for Technical Analysis
    # import ipywidgets as wd
    # from ipywidgets import interact, interact_manual
    # import cufflinks as cf
    # import pandas as pd
    # import yfinance as yf
    # from plotly.offline import iplot, init_notebook_mode
    #
    # stocks = ["TWTR", "MSFT", "GOOGL", "FB", "TSLA", "AAPL"]
    # indicators = ["Bollinger Bands", "MACD", "RSI"]
    #
    # def ta_dashboard(
    #     asset,
    #     indicator,
    #     start_date,
    #     end_date,
    #     bb_k,
    #     bb_n,
    #     macd_fast,
    #     macd_slow,
    #     macd_signal,
    #     rsi_periods,
    #     rsi_upper,
    #     rsi_lower,
    # ):
    #
    #     df = yf.download(asset, start=start_date, end=end_date, progress=False, auto_adjust=True)
    #     qf = cf.QuantFig(df, title=f"TA Dashboard - {asset}", legend="right", name=f"{asset}")
    #     if "Bollinger Bands" in indicator:
    #         qf.add_bollinger_bands(periods=bb_n, boll_std=bb_k)
    #     if "MACD" in indicator:
    #         qf.add_macd(fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
    #     if "RSI" in indicator:
    #         qf.add_rsi(
    #             periods=rsi_periods, rsi_upper=rsi_upper, rsi_lower=rsi_lower, showbands=True
    #         )
    #     return qf.iplot()
    #
    # stocks_selector = wd.Dropdown(options=stocks, value=stocks[0], description="Asset")
    # indicator_selector = wd.SelectMultiple(
    #     description="Indicator", options=indicators, value=[indicators[0]]
    # )
    # start_date_selector = wd.DatePicker(
    #     description="Start Date", value=pd.to_datetime("2018-01-01"), continuous_update=False
    # )
    # end_date_selector = wd.DatePicker(
    #     description="End Date", value=pd.to_datetime("2018-12-31"), continuous_update=False
    # )
    # main_selector_label = wd.Label("Main parameters", layout=wd.Layout(height="45px"))
    # main_selector_box = wd.VBox(
    #     children=[
    #         main_selector_label,
    #         stocks_selector,
    #         indicator_selector,
    #         start_date_selector,
    #         end_date_selector,
    #     ]
    # )
    #
    # bb_label = wd.Label("Bollinger Bands")
    # n_param = wd.IntSlider(
    #     value=20, min=1, max=40, step=1, description="N:", continuous_update=False
    # )
    # k_param = wd.FloatSlider(
    #     value=2, min=0.5, max=4, step=0.5, description="k:", continuous_update=False
    # )
    # bollinger_box = wd.VBox(children=[bb_label, n_param, k_param])
    #
    # macd_label = wd.Label("MACD")
    # macd_fast = wd.IntSlider(
    #     value=12, min=2, max=50, step=1, description="Fast avg:", continuous_update=False
    # )
    # macd_slow = wd.IntSlider(
    #     value=26, min=2, max=50, step=1, description="Slow avg:", continuous_update=False
    # )
    # macd_signal = wd.IntSlider(
    #     value=9, min=2, max=50, step=1, description="MACD signal:", continuous_update=False
    # )
    # macd_box = wd.VBox(children=[macd_label, macd_fast, macd_slow, macd_signal])
    #
    # rsi_label = wd.Label("RSI")
    # rsi_periods = wd.IntSlider(
    #     value=14, min=2, max=50, step=1, description="RSI periods:", continuous_update=False
    # )
    # rsi_upper = wd.IntSlider(
    #     value=70, min=1, max=100, step=1, description="Upper Thr:", continuous_update=False
    # )
    # rsi_lower = wd.IntSlider(
    #     value=30, min=1, max=100, step=1, description="Lower Thr:", continuous_update=False
    # )
    # rsi_box = wd.VBox(children=[rsi_label, rsi_periods, rsi_upper, rsi_lower])
    #
    # sec_selector_label = wd.Label("Secondary parameters", layout=wd.Layout(height="45px"))
    # blank_label = wd.Label("", layout=wd.Layout(height="45px"))
    # sec_box_1 = wd.VBox([sec_selector_label, bollinger_box, macd_box])
    # sec_box_2 = wd.VBox([blank_label, rsi_box])
    # secondary_selector_box = wd.HBox([sec_box_1, sec_box_2])
    #
    # controls_dict = {
    #     "asset": stocks_selector,
    #     "indicator": indicator_selector,
    #     "start_date": start_date_selector,
    #     "end_date": end_date_selector,
    #     "bb_k": k_param,
    #     "bb_n": n_param,
    #     "macd_fast": macd_fast,
    #     "macd_slow": macd_slow,
    #     "macd_signal": macd_signal,
    #     "rsi_periods": rsi_periods,
    #     "rsi_upper": rsi_upper,
    #     "rsi_lower": rsi_lower,
    # }
    # ui = wd.HBox([main_selector_box, secondary_selector_box])
    # out = wd.interactive_output(ta_dashboard, controls_dict)
    #
    # display(ui, out)
