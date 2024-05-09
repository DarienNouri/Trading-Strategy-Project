
from sklearn.preprocessing import scale, StandardScaler
from collections import deque
import numpy as np

from src.technical_indicators import Indications


class Preprocessing(Indications):
    
    def __init__(self, asset, interval, start_date=None, end_date=None, exchange='Refinitiv', market=None):
        super().__init__(asset, interval, start_date, end_date, exchange=exchange, market=market)
        self.engulfing_period = -5
        self.sma = -15
        self.lma = -20
        # super(Preprocessing, self).engulfing_analysis()
        # super(Preprocessing, self).support_resistance()
        # super(Preprocessing, self).moving_average_analysis()
        # super(Preprocessing, self).macd_analysis()
        # super(Preprocessing, self).stochastic_analysis()
        # super(Preprocessing, self).rsi_divergence_convergence()
        # super(Preprocessing, self).price_action()

    def scaling(self, df_values):
        training_window = 60
        df_predictors = df_values
        predictors = df_predictors.iloc[:, :-1].columns
        df_predictors = df_predictors.replace([np.inf, -np.inf], 0)
        
        scaler = StandardScaler()
        df_predictors[predictors] = scale(df_predictors[predictors])
        df_predictors[predictors] = scaler.fit_transform(df_predictors[predictors])

        training_sequence = []
        previous_days = deque(maxlen = training_window)
        for i in df_predictors.values:
            previous_days.append([x for x in i[:-1]])
            if len(previous_days) == training_window:
                training_sequence.append([np.array(previous_days), i[-1:]])
                
        X = []
        y = []
        
        for features, action in training_sequence:
            X.append(features)
            y.append(action)
            
        X = np.array(X)
        y = np.array(y)
                                               
        return X, y
    