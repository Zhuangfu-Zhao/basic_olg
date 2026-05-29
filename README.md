# Basic Overlapping Generations Model

A 3-period OLG model with taxes, pensions, and government debt, implemented in Python.  
Adapted from *Introduction to Computational Economics Using Fortran* (Fehr & Kindermann).

## Structure

- `main_simulate.py` — Three solver classes (`simulate_fortran`, `simulate_kfp`, `simulate_yfp`) that differ in iteration strategy (Gauss-Seidel vs Jacobi) and market-clearing condition (output vs capital market)
- `method_inspector.py` — Debugging utility to inspect internal method inputs/outputs without mutating the model
- `OLG.ipynb` — Full mathematical documentation with worked examples

## Quick Start

```python
from main_simulate import simulate_fortran

model = simulate_fortran(TT=50, n=0.2, gamma=0.5, beta=0.9, alpha=0.3,
                         delta=1.0, tax_0=1, tax_1=1)
model.get_SteadyState()
model.get_Transition()
model.info()
model.plot_output_components()
```

## Solver Variants

| Class | Iteration | Clearing Condition |
|-------|-----------|-------------------|
| `simulate_fortran` | Gauss-Seidel | Output market |
| `simulate_kfp` | Jacobi | Capital market |
| `simulate_yfp` | Jacobi | Output market |

## License

MIT © 2026 Zhuangfu Zhao
