import pandas as pd
import backtrader as bt
from icecream import ic


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


src_data = "../data/yf_aapl.pkl"
aapl = pd.read_pickle(src_data)
print("data reading from file...")

aapl_df = aapl["2018-1":"2018-12"]
data = bt.feeds.PandasData(dataname=aapl_df)
ic(aapl_df.head())

cerebro = bt.Cerebro(stdstats=False)
cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.addstrategy(SmaStrategy)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)

print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
cerebro.run()
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
