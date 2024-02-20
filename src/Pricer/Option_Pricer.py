
import numpy as np
from scipy.stats import norm, sem
from time import time
import matplotlib.pyplot as plt

class OptionPricer:
    """
    This class provides mechanisms to price European options using the Black-Scholes formula. It includes
    functionality for both 'call' and 'put' options, as well as performing Monte Carlo simulations to
    estimate option prices.
    """

    def __init__(self, spot_price, strike_price, maturity, risk_free_rate, volatility, initial_variance, kappa, theta, sigma, rho, mu):
        """
        Initializes the BlackScholesPricer with the necessary parameters for option pricing.

        Parameters:
            spot_price (float): Current price of the underlying asset.
            strike_price (float): Strike price of the option.
            maturity (float): Time to expiration of the option in years.
            risk_free_rate (float): Annualized risk-free interest rate, continuously compounded.
            volatility (float): Volatility of the underlying asset (annualized standard deviation of returns).
            initial_variance (float): Initial variance of the underlying asset.
            kappa (float): The rate at which the variance reverts to the long term mean variance (theta).
            theta (float): The long-term mean variance of the underlying asset's returns.
            sigma (float): The volatility of volatility, or the volatility of the asset's variance.
            rho (float): The correlation coefficient between the asset's returns and the variance.
            mu (float): The expected return of the asset, which is the drift rate in the Heston model.       
        """
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

        self.initial_variance = initial_variance
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.mu = mu

    def _calculate_payoffs(self, terminal_stock_prices, option_type):
        """
        Calculates option payoffs from simulated terminal stock prices.

        Parameters:
            terminal_stock_prices (np.ndarray): Simulated terminal stock prices.
            option_type (str): The type of option ('call' or 'put').

        Returns:
            np.ndarray: An array of payoffs for each simulated terminal stock price.
        """
        if option_type == 'call':
            payoffs = np.maximum(terminal_stock_prices - self.strike_price, 0)
        elif option_type == 'put':
            payoffs = np.maximum(self.strike_price - terminal_stock_prices, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")
 
        return payoffs
        
    def price_option_black_scholes(self, option_type='call'):
        """
        Prices a European option using the Black-Scholes formula.

        Parameters:
            option_type (str): The type of option to price, either 'call' or 'put'.
        
        Returns:
            float: The Black-Scholes price of the option.
        """
        d1, d2 = self._calculate_d1_d2()

        if option_type == 'call':
            price = (self.spot_price * norm.cdf(d1) - 
                     self.strike_price * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(d2))
        elif option_type == 'put':
            price = (self.strike_price * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-d2) - 
                     self.spot_price * norm.cdf(-d1))
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

        return price

    def simulate_option_price_monte_carlo_gbm(self, option_type='call', num_simulations=10000, seed=None):
        """
        Simulates the price of a European option using a Monte Carlo approach.

        Parameters:
            option_type (str): The type of option to simulate, either 'call' or 'put'.
            num_simulations (int): The number of simulation paths.
            seed (int, optional): A seed for the random number generator for reproducible results.
        
        Returns:
            tuple: Estimated option price, standard error of the estimate, and elapsed simulation time.
        """
        np.random.seed(seed)
        start_time = time()
        
        # Simulate end-of-period stock prices
        terminal_stock_prices = self._simulate_terminal_stock_prices_gbm(num_simulations)

        # Calculate payoffs
        payoffs = self._calculate_payoffs(terminal_stock_prices, option_type)

        # Discount payoffs to present value and estimate the option price
        discounted_payoffs = np.exp(-self.risk_free_rate * self.maturity) * payoffs
        option_price_estimate = discounted_payoffs.mean()
        standard_error = discounted_payoffs.std(ddof=1) / np.sqrt(num_simulations)

        elapsed_time = time() - start_time

        return option_price_estimate, standard_error, elapsed_time

    def _calculate_d1_d2(self):
        """
        Calculates the d1 and d2 terms used in the Black-Scholes formula.

        Returns:
            tuple: d1 and d2 values.
        """
        d1 = ((np.log(self.spot_price / self.strike_price) + 
              (self.risk_free_rate + 0.5 * self.volatility**2) * self.maturity) / 
             (self.volatility * np.sqrt(self.maturity)))
        d2 = d1 - self.volatility * np.sqrt(self.maturity)
        return d1, d2

    def _simulate_terminal_stock_prices_gbm(self, num_simulations):
        """
        Simulates terminal stock prices using the geometric Brownian motion model.

        Parameters:
            num_simulations (int): The number of simulation paths.
        
        Returns:
            np.ndarray: Array of simulated terminal stock prices.
        """
        random_shocks = np.random.normal(0, 1, num_simulations)
        drift = (self.risk_free_rate - 0.5 * self.volatility**2) * self.maturity
        diffusion = self.volatility * np.sqrt(self.maturity) * random_shocks
        terminal_prices = self.spot_price * np.exp(drift + diffusion)
        return terminal_prices

    def plot_simulation_paths_monte_carlo_gbm(self, num_paths, num_time_steps, show_plot=True):
        """
        Plots a sample of simulated paths for the underlying asset price.

        Parameters:
            num_paths (int): The number of paths to simulate and plot.
            num_time_steps (int): The number of time steps to use in the simulation.
            show_plot (bool): If True, displays the plot; if False, returns the figure and axes.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot (if show_plot is False).
            matplotlib.axes.Axes: The axes object containing the plot (if show_plot is False).
        """
        time_grid = np.linspace(0, self.maturity, num_time_steps)
        paths = np.zeros((num_time_steps, num_paths))

        # Simulate paths
        for i in range(num_paths):
            paths[:, i] = self._simulate_single_path_monte_carlo(num_time_steps)
        
        # Plotting the paths
        plt.figure(figsize=(10, 6))
        plt.plot(time_grid, paths)
        plt.title('Monte Carlo Simulation Paths for Underlying Asset Price')
        plt.xlabel('Time to Maturity')
        plt.ylabel('Asset Price')
        plt.grid(True)
        
        if show_plot:
            plt.show()
        else:
            return plt.gcf(), plt.gca()

    def _simulate_single_path_monte_carlo(self, num_time_steps):
        """
        Simulates a single path for the underlying asset price.

        Parameters:
            num_time_steps (int): The number of time steps to use in the simulation.

        Returns:
            np.ndarray: An array representing the simulated asset price path.
        """
        dt = self.maturity / (num_time_steps - 1)
        path = np.zeros(num_time_steps)
        path[0] = self.spot_price
        for t in range(1, num_time_steps):
            z = np.random.standard_normal()
            path[t] = path[t-1] * np.exp((self.risk_free_rate - 0.5 * self.volatility**2) * dt +
                                         self.volatility * np.sqrt(dt) * z)
        return path


    def simulate_heston_paths(self, num_time_steps, num_paths):
        """
        Generate simulated paths for the asset price and variance using the Heston model.
    
        The Heston model assumes that the asset price follows a stochastic process with a
        stochastic variance. This method uses the Euler-Maruyama method to discretize both
        stochastic processes and simulates the paths over the time to maturity.
    
        The Feller condition is checked to ensure the variance process does not hit zero or
        become negative.
    
        Parameters:
            num_time_steps (int): The number of time steps in the discretized time horizon.
            num_paths (int): The number of independent paths to simulate.
        
        Returns:
            S_T (np.ndarray): An array of simulated end asset prices for each path.
            v_T (np.ndarray): An array of simulated end variances for each path.
        
        Raises:
        - AssertionError: If the Feller condition (2*kappa*theta > sigma^2) is not satisfied.
        """

        dt = self.maturity / (num_time_steps - 1)
        dt_sq = np.sqrt(dt)
    
        # Feller condition
        assert(2 * self.kappa * self.theta > self.sigma ** 2)
    
        # Initialize arrays to store full paths
        S_paths = np.zeros((num_time_steps, num_paths))
        v_paths = np.zeros((num_time_steps, num_paths))
        
        for path in range(num_paths):
            S = np.zeros(num_time_steps)
            v = np.zeros(num_time_steps)
            S[0] = self.spot_price
            v[0] = self.initial_variance
            
            # Generate correlated Brownian motions
            W_S = np.random.normal(size=num_time_steps-1)
            W_v = self.rho * W_S + np.sqrt(1 - self.rho ** 2) * np.random.normal(size=num_time_steps-1)

            for t in range(1, num_time_steps):
                v[t] = np.fabs(v[t-1] + self.kappa * (self.theta - v[t-1]) * dt + self.sigma * np.sqrt(np.fabs(v[t-1])) * dt_sq * W_v[t-1])
                S[t] = S[t-1] * np.exp((self.mu - 0.5 * v[t-1]) * dt + np.sqrt(np.fabs(v[t-1])) * dt_sq * W_S[t-1])
            
            S_paths[:, path] = S
            v_paths[:, path] = v

        return S_paths, v_paths

    def price_option_heston_monte_carlo(self, num_time_steps, num_paths, option_type='call', Err=False, Time=False):
        """
        Perform Monte Carlo simulations to estimate the option price using the Heston model.
    
        Parameters:
            N (int): The number of time steps in the discretized time horizon.
            paths (int): The number of simulated paths.
            payoff (function): A function that calculates the payoff of the option given the simulated paths.
            Err (bool): If True, the function will also return the standard error of the estimate.
            Time (bool): If True, the function will also return the execution time.
        
        Returns:
            V (float): The estimated option price based on the simulations.
            std_err (float, optional): The standard error of the estimated option price.
            elapsed (float, optional): The execution time of the Monte Carlo simulation.
        """
        t_init = time()

        # Simulate all paths using the Heston model
        S_paths, _ = self.simulate_heston_paths(num_time_steps, num_paths)
        
        # Extract the terminal asset prices for each path
        terminal_prices = S_paths[-1, :]
        
        # Calculate payoffs for the terminal asset prices
        payoffs = self._calculate_payoffs(terminal_prices, option_type)
        
        # Discount the payoffs to present value
        discounted_payoffs = np.exp(-self.risk_free_rate * self.maturity) * payoffs
        
        # Calculate option price estimate and standard error if requested
        V = np.mean(discounted_payoffs)
        std_err = sem(discounted_payoffs) if Err else None
        
        # Calculate elapsed time if requested
        elapsed = time() - t_init if Time else None
        
        # Return the appropriate tuple based on the flags
        return (V, std_err, elapsed) if (Err and Time) else (V, std_err) if Err else (V, elapsed) if Time else V
    
    def plot_heston_paths(self, S_all_paths, v_all_paths, num_paths_to_plot=1000):
        """
        Plot simulated paths for the underlying asset price and variance from the Heston model.

        Parameters:
            S_all_paths (np.ndarray): Matrix of simulated asset price paths.
            v_all_paths (np.ndarray): Matrix of simulated variance paths.
            num_paths_to_plot (int): Number of paths to display on the plot.
        """
        time_steps = S_all_paths.shape[0]
        time_grid = np.linspace(0, self.maturity, time_steps)

        plt.figure(figsize=(14, 6))
        
        # Plot asset price paths
        plt.subplot(1, 2, 1)
        plt.plot(time_grid, S_all_paths[:, :num_paths_to_plot])
        plt.title('Simulated Asset Price Paths')
        plt.xlabel('Time')
        plt.ylabel('Asset Price')
        plt.grid(True)

        # Plot variance paths
        plt.subplot(1, 2, 2)
        plt.plot(time_grid, v_all_paths[:, :num_paths_to_plot])
        plt.title('Simulated Variance Paths')
        plt.xlabel('Time')
        plt.ylabel('Variance')
        plt.grid(True)

        plt.tight_layout()
        plt.show()   