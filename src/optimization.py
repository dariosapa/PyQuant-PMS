import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, Union

class OptimizationEngine:
    """
    Portfolio Optimization Engine.
    Implements Modern Portfolio Theory (Markowitz) using Scipy.
    Supports:
    - Maximize Sharpe Ratio
    - Minimize Volatility
    - Robust Fallback (if solver fails)
    """

    def __init__(self, returns: pd.DataFrame):
        """
        returns: DataFrame of daily returns (Index=Date, Cols=Assets)
        """
        self.returns = returns
        self.assets = returns.columns
        self.n_assets = len(self.assets)
        
        # Annualized estimates
        self.mu = self.returns.mean() * 252
        self.sigma = self.returns.cov() * 252

    def _get_constraints(self, weight_bounds: tuple):
        """Internal method to generate standard constraints."""
        # 1. Sum of weights must be 1 (Fully Invested)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        # 2. Boundaries (e.g., 0% to 100% per asset)
        bounds = tuple(weight_bounds for _ in range(self.n_assets))
        return constraints, bounds

    def maximize_sharpe(self, risk_free_rate: float = 0.0, max_weight: float = 1.0) -> pd.Series:
        """
        Finds the portfolio weights that maximize the Sharpe Ratio.
        """
        num_assets = self.n_assets
        args = (self.mu, self.sigma, risk_free_rate)
        
        # Objective Function: Negative Sharpe (since we minimize)
        def neg_sharpe(weights, mu, sigma, rf):
            ret = np.sum(weights * mu)
            vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            return - (ret - rf) / vol

        constraints, bounds = self._get_constraints((0.0, max_weight))
        
        # Initial Guess (Equal Weights)
        init_guess = num_assets * [1. / num_assets,]

        try:
            result = minimize(neg_sharpe, init_guess, args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
            
            if not result.success:
                print(f"[WARN] Optimizer failed: {result.message}. Using Equal Weights fallback.")
                return self._fallback_equal_weights()
            
            return pd.Series(result.x, index=self.assets, name="MaxSharpe")

        except Exception as e:
            print(f"[ERROR] Optimization error: {e}. Using Equal Weights fallback.")
            return self._fallback_equal_weights()

    def minimize_volatility(self, max_weight: float = 1.0) -> pd.Series:
        """
        Finds the portfolio weights that minimize Volatility (Global Minimum Variance).
        """
        num_assets = self.n_assets
        args = (self.sigma,)

        def port_vol(weights, sigma):
            return np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))

        constraints, bounds = self._get_constraints((0.0, max_weight))
        init_guess = num_assets * [1. / num_assets,]

        try:
            result = minimize(port_vol, init_guess, args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
            
            if not result.success:
                return self._fallback_equal_weights()
            
            return pd.Series(result.x, index=self.assets, name="MinVol")
            
        except Exception:
            return self._fallback_equal_weights()

    def _fallback_equal_weights(self) -> pd.Series:
        """Returns 1/N weights if optimization fails."""
        weights = np.array([1.0 / self.n_assets] * self.n_assets)
        return pd.Series(weights, index=self.assets, name="EqualWeight")