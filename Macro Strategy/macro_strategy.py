# from statsmodels.regression.linear_model import OLS as ols
from statsmodels.regression.rolling import RollingOLS
from scipy.optimize import minimize as opt
from scipy.optimize import Bounds
from multiprocessing import Pool
import statsmodels.api as sm
import multiprocessing
import pandas as pd
import numpy as np
import pickle

# # Get macro model probabilities output
# macro_probs = pd.read_csv(r'C:\Users\marcu\Documents\Profession\Windermere Capital\Macro Model\Full Macro Model Quadrant History.csv').set_index('Date')
# macro_probs.index = pd.DatetimeIndex(macro_probs.index)

# # Get macro model asset returns to regress
# asset_returns = pd.read_csv(r'C:\Users\marcu\Documents\Profession\Windermere Capital\Macro Model\Asset Returns\macro_model_asset_returns_filtered.csv').set_index('Date')
# asset_returns.index = pd.DatetimeIndex(asset_returns.index)



class Strategy:

    def __init__(self, macro_probs, asset_returns, rebal_freq = 1, 
                                                   reg_lookback_period = 10, 
                                                   target_vol = .01, 
                                                   path = r'C:\Users\marcu\Documents\Quant\Programming\Macro Strategy\Backtests', 
                                                   quadrant = 'Goldilocks'):
        """_summary_

        Args:
            macro_probs (_type_): _description_
            asset_returns (_type_): _description_
            rebal_freq (int, optional): _description_. Defaults to 30.
            reg_lookback_period (int, optional): _description_. Defaults to 10.
            target_vol (float, optional): _description_. Defaults to .1.
            quadrant (str, optional): _description_. Defaults to 'Goldilocks'.
            df1_backtest (bool, optional): _description_. Defaults to False.
        """


        self.macro_probs = macro_probs[quadrant]
        self.asset_returns = asset_returns
        self.rebal_freq = rebal_freq
        self.reg_lookback_period = reg_lookback_period
        self.target_vol = target_vol
        self.quadrant = quadrant
        self.path = path
        
                
        # Use multiprocessing.Manager to use pred_wts as a shared data object across processes when running program in parallel
        manager = multiprocessing.Manager()
        self.regression_summary = manager.dict()
        self.run_regressions_in_parallel()
        
        # Compute expected returns for both returns ~ probabilities & returns ~ pct change in probabilities
        self.pred_returns, self.df1_pred_returns = self.get_expected_returns()

        # Run MVO computation with specified params
        self.mvo_wts = self.run_mvo()

        # Get strategy data via backtest
        self.strategy_returns, self.strategy_cumulative_returns, self.sharpe, self.vol = self.run_backtest()

        # Collate data and store as HashMap / dict
        self.backtest = {   'Sharpe Ratio' : self.sharpe, 
                            'Strategy Returns' : self.strategy_returns, 
                            'Cumulative Returns' : self.strategy_cumulative_returns,
                            'Vol' : self.vol, 
                            'MVO Weights' : self.mvo_wts, 
                            'Predicted Returns' : self.pred_returns,
                        }
        
        # Pickle backtest summary data to specified path
        backtest_path = fr'{self.path}\macro_backtest_rebal_freq_{self.rebal_freq}_reg_lookback_{self.reg_lookback_period}.pickle'
        with open(backtest_path, 'wb') as handle:
            pickle.dump(self.backtest, handle, protocol=pickle.HIGHEST_PROTOCOL)

        

            
    def rolling_reg(self, ticker, ticker_returns, regression_summary):
        """_summary_

        Args:
            ticker (str): security ticker
            ticker_returns (pd.Series): historical returns for specified ticker
            regression_summary (dict): Strategy attribute to store tickers' regression summary data
        """


        # Shift returns back by one step & drop NaNs in returns
        ticker_returns = ticker_returns.dropna().shift(-1).dropna()

        # ------------------------------ Regress probabilities on quadrant's asset returns ------------------------------

        # Ensure overlapping indices for specific ticker & probabilities
        indices = self.macro_probs.index.intersection(ticker_returns.index)
        ticker_returns = ticker_returns.loc[indices]
        tmp_quad_probs = self.macro_probs.loc[indices]

        # Append constant to the independent variable (probabilities)
        model_quad_probs = sm.add_constant(tmp_quad_probs)
        # Initialize OLS model
        returns_model = RollingOLS(ticker_returns, model_quad_probs, window = self.reg_lookback_period)
        # Fit OLS model & print summary
        returns_reg = returns_model.fit()

        # ------------------------------ Regress probabilities' percent change on quadrant's asset returns ------------------------------

        # Compute percent change of probabilities
        quad_probs_pct_change = (np.log(tmp_quad_probs / tmp_quad_probs.shift(1)))
        # Replace infinite values with NaNs
        quad_probs_pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop all NaNs
        quad_probs_pct_change = quad_probs_pct_change.dropna()

        # Ensure overlapping indices
        indices = quad_probs_pct_change.index.intersection(ticker_returns.index)
        ticker_returns = ticker_returns.loc[indices]
        quad_probs_pct_change = quad_probs_pct_change.loc[indices]

        # Append constant to the independent variable (returns)
        model_quad_probs_pct_change = sm.add_constant(quad_probs_pct_change)
        
        # Initialize OLS model
        pct_change_model = RollingOLS(ticker_returns, model_quad_probs_pct_change, window = self.reg_lookback_period)
        # Fit OLS model & print summary
        pct_change_reg = pct_change_model.fit()

        # ------------------------------ Store regression data in summary DataFrame ------------------------------

        # Store t-stats of returns regressed on probabilities
        t_stats = returns_reg.tvalues[self.quadrant]
        t_stats.name = 'T-Stat'
        # Store intercepts and betas of returns regressed on probabilities
        params = returns_reg.params
        params = params.rename(columns = {'const' : 'Intercept', self.quadrant : 'Beta'})
        # Collate t-stats, intercepts, & betas into a single DataFrame for returns regressed on probabilities
        reg_df = pd.concat([params, t_stats], axis=1)

        # Store t-stats of returns regressed on pct change of probabilities
        df1_t_stats = pct_change_reg.tvalues[self.quadrant]
        df1_t_stats.name = 'DF1 T-Stat'
        # Store intercepts and betas of returns regressed on pct change of probabilities
        df1_params = pct_change_reg.params
        df1_params = df1_params.rename(columns = {'const' : 'DF1 Intercept', self.quadrant : 'DF1 Beta'})
        # Collate t-stats, intercepts, & betas into a single DataFrame for returns regressed on pct change of probabilities
        pct_change_reg_df = pd.concat([df1_params, df1_t_stats], axis=1)

        # Consolidate both types of regressions in a single regression summary DataFrame
        ticker_reg_summary = pd.concat([reg_df, pct_change_reg_df], axis=1)

        regression_summary[ticker] = ticker_reg_summary

        return


    # Run rolling n day regressions: Returns ~ Probabilities
    def run_regressions_in_parallel(self):
        """ Executes self.rolling_reg to compute each ticker's regression data in parallel.
            This will utilize multiprocessing.Pool and starmap (which will pass multiple 
            params as arguments to self.rolling_reg) to execute this data aggregation process
            much more efficiently.
        """
        
        # Regress returns on individual quad_probs
        # Run regressions in parallel using 10 workers and pool method
        with Pool(10) as pool:

            # Store an iterable args object to pass through starmap method
            # This iterable object will include each ticker, ticker returns, and must pass the regression proxy dict created by multiprocess.Manager()
            args = [(ticker, ticker_returns, self.regression_summary) for ticker, ticker_returns in self.asset_returns.items()]
            
            # Use starmap to pass multiple arguments to target function
            pool.starmap(self.rolling_reg, args)

        # Pickle regression summary data
        reg_path = fr'{self.path}\rolling_reg_data_lookback_{self.reg_lookback_period}_days.pickle'
        with open(reg_path, 'wb') as handle:
            pickle.dump(self.regression_summary.items(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('UPDATE: Regression Summaries Computed')

        return 

    def get_expected_returns(self):
        """_summary_

        Returns:
            pd.DataFrame : returns both types of predicted returns (probs & pct change in probs)
        """

        # Initialize pd.DataFrames to store both types of predicted returns
        pred_returns = pd.DataFrame()
        df1_pred_returns = pd.DataFrame()

        # Iterate through each ticker's regression data
        for ticker, reg_params in self.regression_summary.items():

            # Drop NaNs for non-recorded dates
            returns = self.asset_returns[ticker].dropna()
            
            # Ensure matching indices
            indices = self.macro_probs.index.intersection(returns.index)
            returns = returns.loc[indices]
            quad_probs = self.macro_probs.loc[indices]
            
            # ------------------------------ Predict rets with quadrant probabilities ------------------------------

            # Get security's regression params
            beta = reg_params['Beta'].shift(1)
            intercept = reg_params['Intercept'].shift(1)
            
            # Compute expected return for next day using previous day's probability
            # This is linear regression equation: Y = X*Beta + Alpha
            pred = quad_probs.shift(1)*beta + intercept
            
            # Store ticker's predicted returns in quadrant's DataFrame
            pred_returns = pd.concat([pred_returns, pd.DataFrame({ticker : pred})], axis=1)

                    
            # ------------------------------ Predict rets with quadrant probabilities' percent change ------------------------------

            
            # Compute percent change of probabilities
            quad_probs_pct_change = np.log(quad_probs / quad_probs.shift(1))
            # Replace infinite values with NaNs
            quad_probs_pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Drop all NaNs
            quad_probs_pct_change = quad_probs_pct_change.dropna()
                    
            # Get security's df1_regression params
            df1_beta = reg_params['DF1 Beta']
            df1_intercept = reg_params['DF1 Intercept']
            
            # Compute expected return for next day using previous day's probability
            df1_pred = quad_probs_pct_change.shift(1)*df1_beta + df1_intercept
            
            # Store ticker's predicted returns in quadrant's DataFrame
            df1_pred_returns = pd.concat([df1_pred_returns, pd.DataFrame({ticker : df1_pred})], axis=1)

        print('UPDATE: Expected Returns Computed')

        return pred_returns, df1_pred_returns

    def sharpe_ratio(self, wts, expected_returns, cov_matrix, neg = True):
        mu = (expected_returns.T * wts).sum()
        sigma = np.sqrt(np.dot(np.dot(wts.T, cov_matrix), wts))
        sharpe_ratio = mu / sigma
        
        if neg == True: 
            return -sharpe_ratio
        
        return sharpe_ratio

    def mvo(self, returns, expected_returns):
        """
        :param returns: historical asset returns
        :type returns: pd.DataFrame
        :param expected_returns: expected return for next period
        :type expected_returns: pd.Series

        """

        # Match tickers across expected returns and historical returns
        expected_returns.dropna(inplace=True)
        returns = returns.loc[:, expected_returns.index]

        vols = returns.std()*252**.5
        cov_matrix = returns.cov()
        
        n = returns.columns.size
        if n > 0:
            
            # Initial guess is naive 1/n portfolio
            initial_guess = np.array([1 / n] * n)
            
            # Set max allocation per security
            bounds = Bounds(-.2, .2)

            constraints = [{"type": "eq", "fun": lambda vols: np.sum(vols) - 1}, 
                           {"type": "eq", "fun": lambda vols: np.sqrt(np.dot(np.dot(vols.T, cov_matrix), vols)) - self.target_vol}]

            wts = pd.Series(opt(self.sharpe_ratio, 
                    initial_guess,
                    args=(expected_returns, cov_matrix), 
                    method='SLSQP', 
                    bounds = bounds,
                    constraints=constraints)['x']
                )

            wts.index = vols.index

            # print('Target Vol: ')
            # print(np.sqrt(np.dot(np.dot(wts.T, cov_matrix), wts)))

            # mvo_sr = sharpe_ratio(wts, expected_returns, cov_matrix, neg=False)
            
            return wts
        return

    def get_mvo_wts(self, date):
        try:
            tmp_pred_rets = self.pred_returns.loc[date]
            tmp_asset_rets = self.asset_returns.loc[:date-pd.DateOffset(days=1), tmp_pred_rets.index]
            wts = self.mvo(tmp_asset_rets, tmp_pred_rets)
            return date, wts
            #pred_wts = pd.concat([pred_wts, pd.DataFrame({date : wts})], axis = 1)
        except:
            return date, None

    def store_wts(self, date, wts, pred_wts):
        pred_wts[date] = wts
        return 

    def run_mvo(self):
        
        # Use multiprocessing.Manager() to use pred_wts as a shared data object across processes when running program in parallel
        wts_manager = multiprocessing.Manager()
        tmp_pred_wts = wts_manager.dict()

        # Acquire each day's target MVO weights in parallel
        with Pool(10) as pool:

            # Use imap_unordered for fastest acquisition of weights for each date
            results = pool.imap_unordered(self.get_mvo_wts, self.pred_returns.index[::self.rebal_freq]) # Iteratable object is every n'th day of predicted returns
            
            # Loop through results to acquire dates and their respective weights
            # These will be used as params for the next parallel proceses which fills in weights dict
            args = [(date, wts, tmp_pred_wts) for date, wts in results]

        # Assign weights to their respective dates in parallel
        with Pool(1) as pool:

            # Use starmap to pass multiple arguments to target function
            pool.starmap(self.store_wts, args)
        
        # Should be able to directly map dict to DataFrame... however, this does not work
        # Fix later - this takes a lot of time for daily rebal... something like 5 minutes to run the loop
        pred_wts = pd.DataFrame()
        for date, wts in tmp_pred_wts.items():
            if wts is not None:
                pred_wts = pd.concat([pred_wts, pd.DataFrame({date : wts})], axis=1)
        
        pred_wts = pred_wts.T

        # with open(fr'C:\Users\marcu\Documents\Quant\Programming\Macro Strategy\cache\pred_wts_lookback_{self.reg_lookback_period}_days_rebal_freq_{self.rebal_freq}.pickle', 'wb') as handle:
        #     pickle.dump(pred_wts, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('UPDATE: MVO Weights Computed')

        return pred_wts

    def run_backtest(self):
        """_summary_
        """

        full_wts = pd.DataFrame()
        full_wts.index = self.asset_returns.index
        full_wts = pd.concat([full_wts, self.mvo_wts], axis=1).ffill()

        strategy_returns = (full_wts * self.asset_returns).sum(1)
        strategy_cumulative_returns = strategy_returns.cumsum()
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * 365 ** .5
        vol = strategy_returns.std() * 365 ** .5

        # strategy_cumulative_returns.plot()
        # plt.savefig('cum_rets.jpg')


        return strategy_returns, strategy_cumulative_returns, sharpe_ratio, vol

# if __name__ == '__main__':

#     params = [[500, 30]]#, [15, 15], [5, 30], [5, 10], [1, 15], [1, 10], [1, 5]]

#     for rebal_freq, reg_lookback_period in params:

#         print(f"Backtest: rebal_freq = {rebal_freq} - lookback_period = {reg_lookback_period}")
#         t0 = perf_counter()
#         backtest = Strategy(macro_probs, asset_returns, rebal_freq = rebal_freq, reg_lookback_period = reg_lookback_period)
#         t1 = perf_counter()
#         print(t1 - t0)
#         print(backtest.sharpe)
        
#         path = fr'C:\Users\marcu\Documents\Quant\Programming\Macro Strategy\Backtests\backtest_rebal_freq_{rebal_freq}_reg_lookback_{reg_lookback_period}.pickle'
#         with open(path, 'wb') as handle:
#             pickle.dump(backtest, handle, protocol=pickle.HIGHEST_PROTOCOL)



