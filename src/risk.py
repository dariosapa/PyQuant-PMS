import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from typing import Dict, Union

class RiskEngine:
    """
    Advanced Risk Management Engine.
    Calculates Value at Risk (VaR) and Expected Shortfall (CVaR)
    using Parametric, Historical, and Cornish-Fisher methods.
    """

    def __init__(self, returns: pd.Series):
        """
        Input: pd.Series of daily returns (clean, no NaNs).
        """
        self.returns = returns.dropna()
        self.mean = self.returns.mean()
        self.std = self.returns.std(ddof=1)
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns, fisher=True)  # Excess kurtosis (Normal = 0)

    def calculate_var_parametric(self, alpha: float = 0.95) -> float:
        """
        Parametric VaR under Normal Distribution assumption.
        """
        z_score = norm.ppf(1 - alpha)
        # VaR is usually a negative number representing loss
        return self.mean + z_score * self.std

    def calculate_var_historical(self, alpha: float = 0.95) -> float:
        """
        Historical VaR (Percentile method).
        No distribution assumptions.
        """
        # If alpha is 0.95 (95% confidence), we look for the 5% quantile
        return np.percentile(self.returns, (1 - alpha) * 100)

    def calculate_var_cornish_fisher(self, alpha: float = 0.95) -> float:
        """
        Cornish-Fisher VaR (Modified VaR).
        Adjusts for Skewness and Kurtosis (Fat Tails).
        Essential for real financial data which is rarely Normal.
        """
        z = norm.ppf(1 - alpha)
        
        # Cornish-Fisher Expansion formula
        # S = Skew, K = Kurtosis
        z_cf = (z 
                + (1/6) * (z**2 - 1) * self.skew 
                + (1/24) * (z**3 - 3*z) * self.kurt 
                - (1/36) * (2*z**3 - 5*z) * (self.skew**2))
        
        return self.mean + z_cf * self.std

    def calculate_cvar(self, alpha: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall).
        Average of losses exceeding the Historical VaR.
        """
        var_hist = self.calculate_var_historical(alpha)
        # Filter returns worse than VaR
        tail_losses = self.returns[self.returns <= var_hist]
        
        if tail_losses.empty:
            return 0.0
            
        return tail_losses.mean()

    def analyze(self, alpha: float = 0.95) -> Dict[str, float]:
        """Returns a summary dictionary of all risk metrics."""
        return {
            "Mean_Daily_Return": self.mean,
            "Volatility_Daily": self.std,
            "Skewness": self.skew,
            "Kurtosis": self.kurt,
            f"VaR_Parametric_{int(alpha*100)}": self.calculate_var_parametric(alpha),
            f"VaR_Historical_{int(alpha*100)}": self.calculate_var_historical(alpha),
            f"VaR_CornishFisher_{int(alpha*100)}": self.calculate_var_cornish_fisher(alpha),
            f"CVaR_{int(alpha*100)}": self.calculate_cvar(alpha)
        }