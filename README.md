<p align="center">
    <h1 align="center"> Option Pricer </h1>
</p>
<p align="center">
    <em><code>► Black Scholes(Numerical and theoretical) and Heston method </code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/kimbh43/option-pricer?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/kimbh43/option-pricer?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/kimbh43/option-pricer?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/kimbh43/option-pricer?style=flat&color=0080ff" alt="repo-language-count">
<p></p>
<hr>

---

##  Overview

Various models and simulation techniques have been developed to price derivatives. Among these, the Black-Scholes model stands as the pioneering effort to provide a theoretical estimate for the prices of European-style options. To account for the limitations of the Black-Scholes model and incorporate the stochastic nature of volatility, Monte Carlo simulations can be used. These simulations apply the Black-Scholes framework to a large number of simulated asset paths to calculate an average payoff at option maturity.

Beyond the Black-Scholes assumptions, the Heston model introduces stochastic volatility into the pricing equation, acknowledging that volatility is not constant but varies over time. This model is particularly useful in capturing market behaviors that the Black-Scholes model cannot, such as volatility smiles and skews.

In this analysis, we will:
- Calculate theoretical option prices using the Black-Scholes formula.
- Simulate option prices using Monte Carlo simulations based on the Black-Scholes model.
- Employ the Heston model to simulate option prices, taking into account stochastic volatility.

By comparing these approaches, we aim to gain a deeper understanding of the pricing mechanisms and the potential benefits and limitations of each method.

---

##  Repository Structure

```sh
└── /
    ├── LICENSE
    ├── README.md
    ├── docs
    │   ├── Makefile
    │   ├── conf.py
    │   └── make.bat
    ├── requirements.txt
    ├── setup.py
    ├── src
    │   ├── Pricer
    │   │   ├── Option_Pricer.py
    │   │   └── __init__.py
    │   └── derivative_pricer.ipynb
    └── tests
        ├── __init__.py
        ├── context.py
        └── test_basic.py
```

---

##  Modules

| File                                                                                      | Summary                         |
| ---                                                                                       | ---                             |
| [requirements.txt](https://github.com/kimbh43/option-pricer/blob/master/requirements.txt) | <code>► requirements.txt</code> |
| [setup.py](https://github.com/kimbh43/option-pricer/blob/master/setup.py)                 | <code>► setup.txt</code> |

| File                                                                                                        | Summary                         |
| ---                                                                                                         | ---                             |
| [derivative_pricer.ipynb](https://github.com/kimbh43/option-pricer/blob/master/src/derivative_pricer.ipynb) | <code>► Option pricing report</code> |

| File                                                                                                 | Summary                         |
| ---                                                                                                  | ---                             |
| [Option_Pricer.py](https://github.com/kimbh43/option-pricer/blob/master/src/Pricer/Option_Pricer.py) | <code>► Sourcecode</code> |


---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **JupyterNotebook**: `Python 2.7.18`

###  Installation

1. Clone the  repository:

```sh
git clone https://github.com/kimbh43/option-pricer/
```

2. Change to the project directory:

```sh
cd 
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

---

### Contributing Guidelines

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/kimbh43/option-pricer/
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

---
