# 🧭 mpc26-team-macropolo

Official repository for **Team Macro Polo**'s entry into the 2026 Macro Place Challenge.

## Team Members
- Ronita Mitra
- Mrinal Mathur

## About The Project
The Macro Place Challenge requires robust Electronic Design Automation (EDA) algorithms capable of efficiently placing macros in complex chip floorplans while optimizing for wirelength, density, and routability. 

Our team's approach focuses on pushing the boundaries of macro placement quality using advanced optimization techniques to navigate the complex VLSI floorplanning space, with strict adherence to the one-hour execution budget constraint.

## Our Approach & Implementations

In this repository, you'll find our team's evolving solutions:

### 1. Baseline Multi-Start (`ms5`)
- Stable coordinate descent and simulated annealing algorithms.
- A multi-start (5 seeds) approach to parallelize local search and establish a strong baseline.

### 2. Hill Climbing & Deep Simmering (`ms5hc`)
- Implemented a "deep-simmering" schedule and exponential cooling for annealed coordinate descent.
- Added multi-start hill climbing specifically targeted at overcoming density-clustering bottlenecks, which we observed heavily in datasets like `ibm01`.
- Tuned memory utilization to prevent swap thrashing during heavy resource loads and parallel sweeps.

### 3. Differentiable Placer (`diff_placer`)
- Experimental framework leveraging differentiable wirelength loss functions to optimize macro placement using gradient descent.

## Evaluation Infrastructure
We have built an automated suite for rapid benchmarking on the IBM challenge datasets:
- **Sanity Checks**: `sanity_sweep_ms5hc.py` for fast validation of algorithm logic.
- **Full Benchmarks**: `full_sweep_ms5hc.py` and `full_sweep_650.py` for full budget utilization tests.
- **Analysis**: `animate_placement.py` for visualization and `ab_incremental.py` for iterative A/B testing of proxy costs.
