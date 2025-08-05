# Sparse Poisson Gamma Belief Networks (SPGBN)

This repository contains the implementation of **Sparse Poisson Gamma Belief Networks for High-Dimensional Sparse Count Data**, a deep probabilistic model designed for feature extraction from high-dimensional sparse count data.

## Prerequisites

- **Operating System**: Linux (recommended)
- **Python**: ≥ 3.8

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HuangRuiiii/SPGBN.git
cd SPGBN/Demohub
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate spgbn-demo
```

### 3. Alternative: Install with pip
```bash
conda create -n spgbn-demo python=3.8
conda activate spgbn-demo
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preprocessing
Before running the main training script, preprocess the data using the provided Jupyter notebook:

```bash
jupyter notebook data/SUBJ/SUBJ_Read.ipynb
```

Execute all cells in the notebook to generate the required `SUBJ_processed_data.pkl` file.

### 2. Train the Model
Run the main training script with default parameters:

```bash
python main.py
```

### 3. Custom Configuration
You can customize the model architecture and training parameters:

```bash
python main.py \
  --layers_num 3 \
  --initial_nodes_num 200 \
  --burnin 100 \
  --collection 100
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
