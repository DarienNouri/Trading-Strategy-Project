
from data_utils.scaling import Preprocessing
from data_utils.technical_indicators import Indications
import ta 

class CustomIndicators(Preprocessing):
    def __init__(self, asset, interval, start_date=None, end_date=None, exchange='Refinitiv', market = None):
        self.engulfing_period = -5
        self.sma = -15
        self.lma = -20
        
        super().__init__(asset, interval, start_date, end_date, exchange=exchange, market=market)
        super(Indications, self).pivot_point()
        super(Indications, self).on_balance_volume()
        super(CustomIndicators, self).engulfing_analysis()
        super(CustomIndicators, self).support_resistance()
        super(CustomIndicators, self).moving_average_analysis()
        super(CustomIndicators, self).macd_analysis()
        super(CustomIndicators, self).stochastic_analysis()
        super(CustomIndicators, self).rsi_divergence_convergence()
        super(CustomIndicators, self).EMA(7)
        super(CustomIndicators, self).EMA(30)

        self.ta_lib_indicators_df = ta.add_all_ta_features(self.df.copy(), open="Open", high="High", low="Low", close="Adj Close", volume="Volume", fillna=True)
        
    def model(self):
        pass