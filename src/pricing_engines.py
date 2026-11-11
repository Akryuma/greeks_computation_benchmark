import torch
import numpy as np

def generate_normal_vect(batch_size, seed = 0, dtype = torch.float64):
    """Generates a batch of normal random variables.
    
    Parameters
    ----------
    batch_size : int
        Number of independent antithetic pairs (returns 2 * batch_size samples).
    seed : int
        Seed for the random number generator.
    """
    generator = torch.Generator().manual_seed(seed)
    normal_vect = torch.randn(batch_size, dtype=dtype, generator=generator)
    
    # antithetic variates
    return torch.concat([normal_vect, -normal_vect], dim = 0)

def vanilla_call_mc_fd(normal_vect, spot, strike, sigma_bs, rate, ttm, 
                       epsilon = 1e-4):
    """Computes the price, delta, vega and gamma of a European call option using Monte Carlo with finite differences.
    
    Parameters
    ----------
    normal_vect : torch.tensor, np.ndarray
        Normal random variables.
    spot : torch.tensor, float, np.ndarray 
        Initial stock price.
    strike : float
        Strike price of the option.
    sigma_bs : torch.tensor, float
        Volatility of the stock.
    rate : float
        Risk-free interest rate.
    ttm : float
        Time to maturity.
    epsilon : float
        Relative bump: spot step is epsilon * spot, volatility step is epsilon * sigma_bs.
    """
    if isinstance(spot, torch.Tensor):
        spot = spot.detach().numpy()
    if isinstance(sigma_bs, torch.Tensor):
        sigma_bs = sigma_bs.item()
    if isinstance(normal_vect, torch.Tensor):
        normal_vect = normal_vect.detach().numpy()

    discount = np.exp(-rate * ttm)
    brownian_motion = np.sqrt(ttm) * normal_vect.reshape(-1, 1)

    s_t_vect = spot * np.exp((rate - 0.5 * sigma_bs**2) * ttm + sigma_bs * brownian_motion)
    payoff = np.maximum(s_t_vect - strike, 0)
    price = discount * np.mean(payoff, axis=0)

    # delta
    payoff_up = np.maximum(s_t_vect * (1 + epsilon) - strike, 0)
    payoff_down = np.maximum(s_t_vect * (1 - epsilon) - strike, 0)
    price_up = discount * np.mean(payoff_up, axis=0)
    price_down = discount * np.mean(payoff_down, axis=0)
    delta = (price_up - price_down) / (2 * epsilon * spot)

    # gamma
    gamma = (price_up - 2 * price + price_down) / (epsilon * spot)** 2

    # vega
    s_t_vect_up = spot * np.exp((rate - 0.5 * (sigma_bs * (1 + epsilon))**2) * ttm + sigma_bs * (1 + epsilon) * brownian_motion)
    s_t_vect_down = spot * np.exp((rate - 0.5 * (sigma_bs * (1 - epsilon))**2) * ttm + sigma_bs * (1 - epsilon) * brownian_motion)
    payoff_up = np.maximum(s_t_vect_up - strike, 0)
    payoff_down = np.maximum(s_t_vect_down - strike, 0)
    price_up = discount * np.mean(payoff_up, axis=0)
    price_down = discount * np.mean(payoff_down, axis=0)
    vega = (price_up - price_down) / (2 * epsilon * sigma_bs)

    return price, delta, vega, gamma

def vanilla_call_mc_aad(normal_vect, spot, strike, sigma_bs, rate, ttm):
    """Computes the price, delta, vega and gamma of a European call option using Monte Carlo with automatic differentiation.
    
    Parameters
    ----------
    normal_vect : torch.tensor
        Normal random variables.
    spot : torch.tensor
        Initial stock price.
    strike : float
        Strike price of the option.
    sigma_bs : torch.tensor
        Volatility of the stock.
    rate : float
        Risk-free interest rate.
    ttm : float
        Time to maturity.
    """
    # Clone inputs to ensure no in-place modifications
    spot = spot.clone().requires_grad_(True)
    sigma_bs = sigma_bs.clone().requires_grad_(True)

    # Vectorize sigma to match spot dimensions
    sigma_vec = sigma_bs.expand_as(spot).clone().requires_grad_(True)

    discount = np.exp(-rate * ttm)
    brownian_motion = np.sqrt(ttm) * normal_vect.view(-1, 1)
    
    s_t_vect = spot * torch.exp((rate - 0.5 * sigma_vec**2) * ttm + sigma_vec * brownian_motion)
    payoff = torch.relu(s_t_vect - strike)
    price = discount * torch.mean(payoff, dim=0)

    # Gradients
    grads = torch.autograd.grad(price.sum(), inputs=[spot, sigma_vec], create_graph=True)
    delta = grads[0]

    vega = grads[1]

    gamma = torch.autograd.grad(delta.sum(), inputs=spot)[0]


    return price.detach().numpy(), delta.detach().numpy(), vega.detach().numpy(), gamma.detach().numpy()

def vanilla_call_mc_aad_smooth(normal_vect, spot, strike, sigma_bs, rate, ttm, beta = 50):
    """Computes the price, delta, vega and gamma of a European call option using Monte Carlo with automatic differentiation and smooth payoff.
    The payoff is smoothed using the softplus function:
        softplus(x) = log(1 + exp(beta * x)) / beta
    
    Parameters
    ----------
    normal_vect : torch.tensor
        Normal random variables.
    spot : torch.tensor
        Initial stock price.
    strike : float
        Strike price of the option.
    sigma_bs : torch.tensor
        Volatility of the stock.
    rate : float
        Risk-free interest rate.
    ttm : float
        Time to maturity.
    beta : int
        Parameter for the softplus function.
    """
    # Clone inputs to ensure no in-place modifications
    spot = spot.clone().requires_grad_(True)
    sigma_bs = sigma_bs.clone().requires_grad_(True)

    # Vectorize sigma to match spot dimensions
    sigma_vec = sigma_bs.expand_as(spot).clone().requires_grad_(True)
    discount = np.exp(-rate * ttm)
    brownian_motion = np.sqrt(ttm) * normal_vect.view(-1, 1)

    s_t_vect = spot * torch.exp((rate - 0.5 * sigma_vec**2) * ttm + sigma_vec * brownian_motion)
    payoff = torch.nn.functional.softplus(s_t_vect - strike, beta=beta)
    price = discount * torch.mean(payoff, dim=0)

    grads = torch.autograd.grad(price.sum(), inputs=[spot, sigma_vec], create_graph=True)
    delta = grads[0]

    vega = grads[1]

    gamma = torch.autograd.grad(delta.sum(), inputs=spot)[0]


    return price.detach().numpy(), delta.detach().numpy(), vega.detach().numpy(), gamma.detach().numpy()

def vanilla_call_mc_malliavin(normal_vect, spot, strike, sigma_bs, rate, ttm):
    """Computes the price, delta, vega and gamma of a European call option using Monte Carlo with Malliavin calculus.
    
    Parameters
    ----------
    normal_vect : torch.tensor, np.ndarray
        Normal random variables.
    spot : torch.tensor, float, np.ndarray
        Initial stock price.
    strike : float
        Strike price of the option.
    sigma_bs : torch.tensor
        Volatility of the stock.
    rate : float
        Risk-free interest rate.
    ttm : float
        Time to maturity.
    """
    if isinstance(spot, torch.Tensor):
        spot = spot.detach().numpy().astype(np.float64)
    if isinstance(sigma_bs, torch.Tensor):
        sigma_bs = sigma_bs.item()
    if isinstance(normal_vect, torch.Tensor):
        normal_vect = normal_vect.detach().numpy().astype(np.float64)

    
    discount = np.exp(-rate * ttm)
    brownian_motion = np.sqrt(ttm) * normal_vect.reshape(-1, 1)
    spots = spot

    s_t_vect = spots * np.exp((rate - 0.5 * sigma_bs**2) * ttm + sigma_bs * brownian_motion)
    payoff = np.maximum(s_t_vect - strike, 0)
    price = discount * np.mean(payoff, axis=0)

    delta = discount / (spots * sigma_bs * ttm) * np.mean((payoff * brownian_motion), axis=0)
    vega = discount * np.mean((payoff * ((brownian_motion**2 -  ttm)/ (sigma_bs * ttm) - brownian_motion)), axis=0)
    gamma = vega / (spots**2 * sigma_bs * ttm)

    return price, delta, vega, gamma
