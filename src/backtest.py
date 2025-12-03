import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.optimization import OptimizationEngine

class BacktestEngine:
    """
    Walk-Forward Backtesting Engine.
    Simulates a portfolio that rebalances periodically using the OptimizationEngine.
    """

    def __init__(self, prices: pd.DataFrame, initial_capital: float = 100_000.0):
        """
        prices: DataFrame of daily Adj Close prices (Index=Date, Cols=Assets)
        """
        self.prices = prices.dropna()
        self.returns = self.prices.pct_change().fillna(0.0)
        self.initial_capital = initial_capital
        self.assets = prices.columns

    def run(self, 
            rebalance_freq: str = 'M', 
            lookback_window: int = 252, 
            strategy: str = 'max_sharpe',
            tx_cost_bps: float = 5.0) -> Dict:
        """
        Executes the backtest.
        
        Params:
        - rebalance_freq: 'M' (Month), 'Q' (Quarter), 'W' (Week)
        - lookback_window: Days of history used for optimization (e.g., 252 = 1 year)
        - strategy: 'max_sharpe', 'min_vol', or 'equal_weight'
        - tx_cost_bps: Transaction costs in basis points (e.g., 5 bps = 0.05%)
        """
        
        # 1. Identify Rebalance Dates
        # We resample to the frequency and take the last day
        rebalance_dates = self.prices.resample(rebalance_freq).last().index
        
        # We need enough history for the first lookback
        start_date = self.prices.index[0] + pd.Timedelta(days=lookback_window)
        rebalance_dates = [d for d in rebalance_dates if d >= start_date]

        # 2. Simulation Loop
        portfolio_value = [self.initial_capital]
        current_weights = pd.Series(0.0, index=self.assets)
        cash = self.initial_capital
        shares = pd.Series(0.0, index=self.assets)
        
        # Store history
        history_dates = [self.prices.index[0]] # Start date
        weight_history = []
        
        # Align simulation to daily prices
        # We iterate day by day to track value accurately
        sim_dates = self.prices.index[self.prices.index >= start_date]
        
        # Initialize shares just before the loop starts (at start_date)
        # We assume we are 100% cash until the first rebalance
        
        print(f"[INFO] Starting Backtest: {len(rebalance_dates)} rebalance events...")

        for date in sim_dates:
            price_today = self.prices.loc[date]
            
            # Update Portfolio Value (Mark-to-Market)
            current_value = cash + (shares * price_today).sum()
            
            # Is today a rebalance day?
            if date in rebalance_dates:
                # 1. Slice History (Walk-Forward)
                # We strictly use data FROM THE PAST (up to yesterday)
                history_slice = self.prices.loc[:date].iloc[:-1].tail(lookback_window)
                returns_slice = history_slice.pct_change().dropna()
                
                # 2. Optimize
                opt = OptimizationEngine(returns_slice)
                
                if strategy == 'max_sharpe':
                    target_weights = opt.maximize_sharpe(max_weight=0.5) # Cap at 50%
                elif strategy == 'min_vol':
                    target_weights = opt.minimize_volatility(max_weight=0.5)
                else:
                    target_weights = pd.Series(1/len(self.assets), index=self.assets)
                
                # 3. Calculate Transaction Costs
                # Turnover = sum of absolute difference between current weight and target
                # Current weight is strictly based on value
                if current_value > 0:
                    current_w_actual = (shares * price_today) / current_value
                else:
                    current_w_actual = pd.Series(0.0, index=self.assets)

                turnover = np.abs(target_weights - current_w_actual).sum()
                cost = current_value * turnover * (tx_cost_bps / 10000.0)
                
                # 4. Execute Trade
                # New value after paying costs
                new_value = current_value - cost
                
                # Calculate new shares
                # shares = (Value * Weight) / Price
                shares = (new_value * target_weights) / price_today
                shares = shares.fillna(0.0)
                
                # Update Cash (implicitly 0 if fully invested, but good for logic)
                cash = 0.0 
                
                # Store weights for plotting
                weight_record = target_weights.copy()
                weight_record.name = date
                weight_history.append(weight_record)

            # Store daily value
            portfolio_value.append(cash + (shares * price_today).sum())
            history_dates.append(date)

        # 3. Compile Results
        equity_curve = pd.Series(portfolio_value[1:], index=sim_dates)
        weights_df = pd.DataFrame(weight_history)
        
        return {
            "equity_curve": equity_curve,
            "weights": weights_df,
            "final_value": equity_curve.iloc[-1],
            "total_return": (equity_curve.iloc[-1] / self.initial_capital) - 1
        }