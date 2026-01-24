# speck-sca-dl-ensemble

Deep learning ensemble–based profiling side-channel analysis of the SPECK lightweight cipher.

---

## Overview

This repository presents a **deep learning ensemble–based profiling attack on the SPECK-32/64 cipher**, a lightweight block cipher designed for resource-constrained devices and widely used in IoT systems. 

To the best of our knowledge, this is the **first deep learning–based profiling attack** targeting both unprotected and protected implementations of SPECK.

Link to Paper: https://www.nature.com/articles/s41598-025-08888-1

Authors: Faisal Hameed (New York University, USA), Hoda Alkhzaimi (New York University, USA)

---

## Datasets

### SPECK Fixed Key

### SPECK Variable Key

### SPECK Protected Key

## Repository Structure

```text
src/        # Datasets, models, ensembles, and SCA metrics
scripts/    # Linux / HPC execution scripts
data/       # SPECK side-channel datasets (HDF5)
results/    # Representative outputs and example metrics
notebooks/  # Analysis and visualization
