# OLG Project Overview

## Purpose

This project implements a small overlapping generations (OLG) model for studying steady states, tax reform, and transition dynamics. It combines an economic notebook with a Python module that solves the model numerically.

The project serves two goals:

1. Explain the economic structure of the model.
2. Simulate how the economy moves from an initial steady state to a new post-reform equilibrium.

## Files

### `OLG.ipynb`

The notebook is the main explanation and demonstration file. It contains:

- the mathematical setup of the OLG model,
- notation tables for variables and parameters,
- an explanation of the main non-plotting methods in `OLG_module.py`,
- an inspector section for testing internal method input-output logic,
- and example code for solving and plotting the model.

### `OLG_module.py`

This file contains the class `OLG_benchmark`, which implements the full computational model.

Its responsibilities include:

- storing model parameters and arrays,
- computing prices,
- solving household decisions,
- aggregating cohort variables into macro variables,
- balancing the government budget through endogenous taxes,
- solving the steady state,
- solving the transition path,
- and plotting model results.

## Installation

This project is lightweight and only requires a standard Python scientific stack.

### Requirements

- Python 3.11 or newer
- `numpy`
- `matplotlib`
- `jupyter` if you want to run the notebook interactively

### Install Dependencies

If you already have a Python environment, install the required packages with:

```bash
pip install numpy matplotlib jupyter
```

If you use a virtual environment, activate it first and then run the same command.

### Run the Project

To work with the notebook:

```bash
jupyter notebook OLG.ipynb
```

To use the model directly from Python:

```python
from OLG_module import OLG_benchmark

OLG = OLG_benchmark(25, 0.2, 0.5, 0.9, 0.3, 0.1, tax_0=1, tax_1=2)
OLG.info()
OLG.get_SteadyState()
OLG.get_Transition()
```

## Economic Structure

### Households

Households live for three periods:

1. young,
2. middle-aged,
3. old.

They work in the first two periods and retire in the third. They choose consumption and saving subject to budget constraints. The household block is driven by:

- cohort growth `n`,
- discount factor `beta`,
- elasticity of intertemporal substitution `gamma`,
- after-tax wage income,
- after-tax asset returns,
- and pension income in retirement.

Optimal decisions are derived from Euler equations and lifetime budget constraints.

### Firms

Firms use capital and labor to produce output with a neoclassical production function.

Key firm-side parameters are:

- capital share `alpha`,
- depreciation rate `delta`.

Given capital and labor, firms determine:

- the real interest rate `r`,
- and the real wage `w`.

### Government

The government has two fiscal components.

1. A financial budget for public spending and public debt.
2. A pay-as-you-go pension system financed by payroll taxes.

The model allows different tax-closure rules, including:

- endogenous consumption tax,
- endogenous wage and interest tax,
- endogenous labor tax,
- endogenous interest tax.

### Equilibrium

The model imposes:

- capital market clearing,
- goods market clearing,
- and labor market clearing.

The code tracks the time paths of aggregate consumption, assets, capital, output, public debt, investment, and tax rates.

## Main Computational Logic

The core computational sequence is:

1. start from a capital guess or capital path,
2. compute prices,
3. solve household decisions,
4. aggregate household outcomes,
5. update government tax rates,
6. iterate until convergence.

This logic is used both for steady-state computation and for transition dynamics.

## Main Methods in `OLG_benchmark`

### `__init__`

Initializes model parameters, fiscal settings, reform timing, solver controls, and all arrays used in the simulation.

### `_prices(it)`

Computes prices at period `it`, including:

- `r[it]`,
- `w[it]`,
- `wn[it]`,
- `Rn[it]`,
- `p[it]`,
- `pen[it]`.

### `_decisions(it)`

Uses current and nearby prices to solve household consumption and saving decisions at period `it`.

### `_aggregations(it)`

Aggregates cohort-level decisions into macro variables such as:

- `CC[it]`,
- `AA[it]`,
- `YY[it]`,
- `BB[it]`,
- `KK[it]`,
- `II[it]`.

### `_government(it)`

Balances the government budget by solving for the endogenous tax rate or tax rates implied by the chosen tax regime.

### `get_SteadyState()`

Iterates on the steady-state block until the goods-market residual is below tolerance.

### `get_Transition()`

Solves the transition path after the reform by iterating over all transition periods.

### `info()`

Prints the model calibration, tax reform setup, and solver settings.

## Typical Workflow

```python
from OLG_module import OLG_benchmark

OLG = OLG_benchmark(25, 0.2, 0.5, 0.9, 0.3, 0.1, tax_0=1, tax_1=2)
OLG.info()
OLG.get_SteadyState()
OLG.get_Transition()
```

After solving the model, you can use the plotting methods to visualize:

- cohort consumption,
- cohort assets,
- aggregate consumption,
- capital stock,
- tax rates,
- and output components.

## Example Output

When you call `OLG.info()`, the model prints a compact calibration summary similar to the following:

```text
========================================
  OLG Model - Economic Setup
========================================

Household Parameters
  Periods (TT)          : 25
  Cohort growth (n)     : 0.200
  EIS (gamma)           : 0.500
  Discount (beta)       : 0.900

Firm Parameters
  Capital share (alpha) : 0.300
  Depreciation (delta)  : 0.100

Government
  Public spending (g)   : [0.12, 0.12, 0.0]
  Initial tax system    : Consumption tax endogenous (1)
  Reform tax system     : Wage & interest tax endogenous (2)

Solver Settings
  Damping               : 0.250
  Tolerance             : 1.00e-05
  Max iterations        : 1000

========================================
```

After `get_SteadyState()` and `get_Transition()`, the notebook can generate figures for cohort consumption, cohort assets, aggregate consumption, capital, tax rates, and output components.

## Notebook Inspector

The notebook includes a simple inspector for the internal non-plotting methods. It works on a deep copy of the model object, so users can modify inputs and observe outputs without changing the original implementation.

This is useful for understanding:

- which internal state variables a method reads,
- which variables it updates,
- and how `_prices`, `_decisions`, `_aggregations`, and `_government` connect to one another.

## What This Project Is Good For

This project is useful for:

- learning the structure of a basic OLG model,
- studying tax reform in a dynamic general equilibrium setting,
- understanding the link between household decisions and fiscal policy,
- experimenting with transition dynamics,
- and teaching how economic equations map into code.

## Summary

This project provides a compact but complete OLG framework that combines:

- economic derivation in a notebook,
- numerical implementation in Python,
- steady-state and transition solvers,
- tax reform analysis,
- and method-level inspection tools.
