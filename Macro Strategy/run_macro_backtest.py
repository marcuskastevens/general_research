from macro_strategy import Strategy
from time import perf_counter
import pandas as pd
import pickle
import os

if __name__ == '__main__':

    # Get macro model probabilities output
    macro_probs = pd.read_csv(r'C:\Users\marcu\Documents\Profession\Windermere Capital\Macro Model\Full Macro Model Quadrant History.csv').set_index('Date')
    macro_probs.index = pd.DatetimeIndex(macro_probs.index)

    # Get macro model asset returns to regress
    asset_returns = pd.read_csv(r'C:\Users\marcu\Documents\Profession\Windermere Capital\Macro Model\Asset Returns\macro_model_asset_returns_filtered.csv').set_index('Date')
    asset_returns.index = pd.DatetimeIndex(asset_returns.index)

    params = [[1, 10]]#, [15, 15], [5, 30], [5, 10], [1, 15], [1, 10], [1, 5]]

    for rebal_freq, reg_lookback_period in params:

        print(f"Backtest: rebal_freq = {rebal_freq} - lookback_period = {reg_lookback_period}")
        t0 = perf_counter()
        backtest = Strategy(macro_probs, asset_returns, rebal_freq = rebal_freq, reg_lookback_period = reg_lookback_period).backtest
        t1 = perf_counter()
        print(f'{t1-t0} seconds')
        sr = backtest['Sharpe Ratio']
        print(f'SR: {sr}')

        print(type(backtest))
        
        # path = fr'C:\Users\marcu\Documents\Quant\Programming\Macro Strategy\Backtests\macro_backtest_rebal_freq_{rebal_freq}_reg_lookback_{reg_lookback_period}.pickle'
        # with open(path, 'wb') as handle:
        #     pickle.dump(backtest, handle, protocol=pickle.HIGHEST_PROTOCOL)





