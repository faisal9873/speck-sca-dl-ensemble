# speck-sca-dl-ensemble

Deep learning ensemble–based profiling side-channel analysis of the SPECK lightweight cipher.

---

## Overview

This repository presents a **deep learning ensemble–based profiling attack on the SPECK-32/64 cipher**, a lightweight block cipher designed for resource-constrained devices and widely used in IoT systems. 

To the best of our knowledge, this is the **first deep learning–based profiling attack** targeting both unprotected and protected implementations of SPECK.

Link to Paper: https://www.nature.com/articles/s41598-025-08888-1

Authors: Faisal Hameed (New York University, US), Hoda Alkhzaimi (New York University)

---

## Key Contributions

- Profiling side-channel attack on **SPECK-32/64** using deep learning
- **Ensemble learning** to improve robustness and trace efficiency
- Sequential divide-and-conquer key recovery strategy
- Successful key recovery with <250 traces
- Evaluation on both unprotected and masked implementations

---

## Project Focus

This project emphasizes **hyperparameter exploration and ensemble behavior** rather than fixed model configurations. Results are therefore analyzed in terms of **attack efficiency, convergence, and stability** under constrained trace budgets.

---

## Repository Structure

```text
src/        # Datasets, models, ensembles, and SCA metrics
scripts/    # Linux / HPC execution scripts
data/       # SPECK side-channel datasets (HDF5)
results/    # Representative outputs and example metrics
notebooks/  # Analysis and visualization
