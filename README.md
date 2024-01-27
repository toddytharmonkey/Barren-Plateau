# Measurement Induced Landscape Transitions (MILT)

## Overview
This repository contains the source code for "Measurement-induced landscape transitions in hybrid variational quantum circuits," a research paper available at [arXiv:2312.09135](https://arxiv.org/abs/2312.09135). The code is structured to facilitate the reproduction of results and figures presented in the paper. Key computations are performed by four primary Python scripts, labeled `MILT_`, which are utilized by various other scripts stored along with their outputs in the `results` folder.

## Installation

To use this codebase:

1. **Clone the Repository**: 
   ```bash
   git clone [repository URL]
   ```
2. **Install Dependencies**: 
   The required Python libraries are primarily listed at the top of `MILT_Core.py`. These include:
   - SciPy
   - openfermion
   - dask
   - NumPy
   - quspin

3. **Running the Scripts**: 
   Execute the script files as needed to regenerate the data for plots or to utilize the functions in the core files for further analysis or expansion.

## Core Files

The four core Python files (`MILT_` series) encompass the following functionalities:
1. `MILT_mutual_information.py`: Handles mutual information calculations.
2. `MILT_core.py`: Provides fundamental utilities and functions.
3. `MILT_gradient_results.py`: Focuses on gradient analysis in quantum circuits.
4. `MILT_optimization.py`: Implements the VQE algorithm and collects performance data.

## Results

The `results` folder includes scripts used for generating the data and their respective outputs. 
## Authors

- **Sonny Rappaport**: Primary developer and researcher.
- **Gaurav Gyawali**: Priamry developer and researcher.
- **Michael Lawler**: Provided initial code structure and much guidance.

## Citation

If you use this code in your research, please cite the accompanying paper: https://arxiv.org/abs/2312.09135 
