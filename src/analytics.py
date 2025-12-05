import torch
import numpy as np
from scipy.stats import norm

def vanilla_call_analytics(spot, strike, sigma_bs, rate, ttm):
    """Computes the Black-Scholes price, delta, vega and gamma of a European call option.
    
    Parameters
    ----------
    spot : torch.tensor, float
        Initial stock price.
    strike : float
        Strike price of the option.
    sigma_bs : torch.tensor, float
        Volatility of the stock.
    rate : float
        Risk-free interest rate.
    ttm : float
        Time to maturity.
    
    Returns
    -------
    price, delta, vega, gamma : float, float, float, float
        Black-Scholes price and Greeks of the European call option.
    """

    if isinstance(spot, torch.Tensor):
        spot = spot.item()
    if isinstance(sigma_bs, torch.Tensor):
        sigma_bs = sigma_bs.item()
    
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma_bs**2) * ttm) / (sigma_bs * np.sqrt(ttm))
    d2 = d1 - sigma_bs * np.sqrt(ttm)

    price = spot * norm.cdf(d1) - strike * np.exp(-rate * ttm) * norm.cdf(d2)
    delta = norm.cdf(d1)
    vega = spot * norm.pdf(d1) * np.sqrt(ttm)
    gamma = norm.pdf(d1) / (spot * sigma_bs * np.sqrt(ttm))

    return price, delta, vega, gamma