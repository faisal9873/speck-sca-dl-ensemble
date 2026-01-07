# speck-sca-dl-ensemble

Deep learning ensemble–based profiling side-channel analysis of the SPECK lightweight cipher.

---

## Overview

Deep learning–based profiling side-channel analysis (SCA) has shown strong performance against classical symmetric ciphers such as AES. However, its application to **lightweight cryptographic primitives** remains largely unexplored.

This repository presents a **deep learning ensemble–based profiling attack on the SPECK-32/64 cipher**, a lightweight block cipher designed for resource-constrained devices and widely used in IoT systems. The proposed approach employs a **sequential divide-and-conquer ensemble of deep learning models** to recover the full **8-byte secret key** from both **unprotected and protected (masked) software implementations** of SPECK, using fewer than **250 traces**.

To the best of our knowledge, this is the **first deep learning–based profiling attack** targeting both unprotected and protected implementations of SPECK.

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
