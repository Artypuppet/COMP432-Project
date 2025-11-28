# COMP432 Kaggle Competition Project

## Project Overview

This project is for a Kaggle competition focused on classifying images of 50 different household objects using pre-extracted deep learning features. Rather than working with raw images, we use feature vectors, allowing us to focus on machine learning fundamentals: model selection, hyperparameter tuning, and proper evaluation techniques.

## Competition Details

- **Task**: Multi-class classification (50 object classes)
- **Dataset**: Pre-extracted deep learning features
- **Training Set**: ~115,406 samples
- **Test Set**: ~49,460 samples
- **Evaluation Metric**: Classification accuracy
- **Labels**: Integer values [0-49] representing 50 object classes

### Competition Rules

**Allowed:**
- PyTorch (custom `nn.Module`, your own training loop with `torch.optim`, etc.)
- True from-scratch implementations using low-level numeric libraries (e.g., NumPy)
- `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
- Simple `torchvision.transforms` (no pretrained models)
- `numpy`, `pandas`, basic plotting
- Metrics may be computed using imported libraries (such as scikit-learn)

**Not Allowed:**
- scikit-learn (model imports)
- XGBoost/LightGBM/CatBoost
- fastai
- timm pretrained models
- Keras high-level `.fit()` workflows
- Any AutoML tools (AutoGluon, AutoKeras, H2O Driverless AI, TPOT, etc.)
- Pretrained weights and transfer learning

## Project Structure

```
├── README.md
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/          <- Original Kaggle CSVs (train.csv, test.csv)
│   ├── interim/      <- Intermediate data transformations
│   └── processed/    <- Final features ready for modeling
├── models/           <- Trained model checkpoints
├── notebooks/        <- Jupyter notebooks for exploration and python modules for reusable code
├── configs/          <- Hyperparameter configuration files (YAML/JSON)
├── submissions/      <- Kaggle submission CSV files
├── reports/          <- Generated outputs and analysis
│   └── figures/      <- Plots and visualizations
```

## Notebooks Directory

The `notebooks/` directory contains both Jupyter notebooks for exploration and Python modules for reusable code. Below is a brief description of each file:

### Python Modules

- **`data_loader.py`**: Utility module for loading training and test data from CSV files. Provides functions to automatically detect the project root and load data with proper handling of ID columns. Includes `load_train_data()` and `load_test_data()` functions.

- **`mlp_model.py`**: Main MLP model implementation using PyTorch. Contains the `MLPConfig` dataclass for hyperparameter configuration and the `MLPTrainer` class that handles model training, validation, and evaluation. Supports various activation functions (ReLU, Leaky ReLU, GELU), normalization layers (BatchNorm, LayerNorm), and regularization techniques.

- **`ensemble_ml_model.py`**: Ensemble MLP implementation using stratified k-fold cross-validation. Builds multiple models on different data folds and combines their predictions through averaging. Contains the `EnsembleMLPConfig` class and `EnsembleMLPTrainer` for managing ensemble training.

### Jupyter Notebooks

- **`data_analysis.ipynb`**: Comprehensive exploratory data analysis (EDA) notebook. Analyzes label distribution, feature statistics, feature sparsity, and feature correlations. Generates visualizations including label distribution plots, feature distribution histograms, and correlation heatmaps saved to `reports/figures/`.

- **`MLP_pytorch.ipynb`**: Main PyTorch training notebook for model development and experimentation. Contains the complete training pipeline, hyperparameter tuning experiments, and model evaluation. Includes code for generating confusion matrices, loss plots, and other performance visualizations.

- **`MLP.ipynb`**: Simple NumPy-based MLP implementation from scratch (without PyTorch). Implements forward propagation with ReLU activation and softmax output layer. Useful for understanding the basic mechanics of multilayer perceptrons.

- **`csv2matrix.ipynb`**: Simple utility notebook for converting CSV data files into NumPy matrices. Demonstrates basic data loading and conversion operations for quick data inspection.

### Documentation

- **`best_results.md`**: Documentation of best-performing hyperparameter configurations and their results. Contains JSON configuration examples for various model architectures and their validation accuracies, useful for reproducing best results.

## Setup Instructions

### 1. Create Conda Environment

Create the conda environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will install all base dependencies (numpy, pandas, scikit-learn, matplotlib, jupyter, etc.) but **not** PyTorch.

### 2. Activate Environment

```bash
conda activate comp432-project
```

### 3. Install PyTorch with GPU Support

PyTorch must be installed manually to ensure compatibility with your system's CUDA version.

#### Check Your CUDA Version

First, check your NVIDIA driver version and supported CUDA version:

```bash
nvidia-smi
```

Look for the "CUDA Version" field in the output. This indicates the maximum CUDA version your driver supports.

**Important Note on CUDA Compatibility:**
CUDA drivers are backward compatible. If your driver supports CUDA 13.0, it can also run PyTorch built with CUDA 12.4, 12.1, or 11.8. You should install the latest stable `pytorch-cuda` version available, not necessarily the exact CUDA version shown by `nvidia-smi`.

#### Install PyTorch

Choose the installation command based on available stable PyTorch CUDA versions:

**For CUDA 12.4 (Latest Stable - Recommended):**
```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

**For CUDA 12.1:**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**For CUDA 11.8:**
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

**For CPU-only (no GPU):**
```bash
conda install pytorch torchvision -c pytorch
```

#### Verify GPU Setup

After installation, verify that PyTorch can detect your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

If GPU is available, you should see `CUDA available: True` and the CUDA version.

#### Troubleshooting: AMD CPU Issues

If you encounter an error like `ImportError: undefined symbol: iJIT_NotifyEvent` when importing PyTorch, this is typically caused by Intel MKL libraries conflicting with AMD CPUs. To fix this:

1. Install OpenBLAS to replace Intel MKL dependencies:
   ```bash
   conda install "libblas=*=*openblas" "liblapack=*=*openblas" -c conda-forge
   ```

2. Reinstall PyTorch with CUDA support:
   ```bash
   conda remove pytorch torchvision pytorch-mutex --force-remove
   conda install pytorch torchvision pytorch-cuda=12.4 pytorch-mutex=1.0=cuda -c pytorch -c nvidia
   ```

3. Verify the fix:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

This should resolve the import error and allow PyTorch to work correctly on AMD CPU systems.

## Dataset Setup

### Download Dataset

1. Go to the Kaggle competition page
2. Navigate to the **Data** tab
3. Download the following files:
   - `train.csv` - Training data with labels
   - `test.csv` - Test data (features only, no labels)
   - `sample_submission.csv` - Example submission format (optional)

### Place Files

Place the downloaded CSV files in the `data/raw/` directory:

```
data/raw/
├── train.csv
└── test.csv
```

**Note**: The `data/` directory is gitignored, so you'll need to download the files manually.

## Usage

### Running Jupyter Notebooks

```bash
conda activate comp432-project
jupyter notebook
```

Or use JupyterLab:

```bash
jupyter lab
```

### Training Models

To train the models, open and run all cells in the `notebooks/MLP_pytorch.ipynb` notebook. The notebook contains detailed comments and markdown cells that explain what each code cell does, including:

- Loading the config file
- Training one individual model using the `configs/mlp_config.json` and training the ensemble model using `configs/ensemble_config.json` file.
- Model architecture configuration
- Training loop implementation
- Validation and evaluation procedures
- Visualization of training metrics and results

Simply execute all cells sequentially to train the model from scratch, or modify the configuration cells to experiment with different hyperparameters. But beware, it took around 45 minutes to run all the cells on a laptop with RTX 3050 so it may take longer.


## Notes

- This project uses a lightweight structure based on cookiecutter-data-science principles
- The `src/` directory is initially empty and will be populated as code is refactored from notebooks
- Configuration files can be added to `configs/` for hyperparameter management
- Model checkpoints are saved in `models/` (gitignored)
- All plots and visualizations should be saved in `reports/figures/`

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/your-competition-name)
- [Competition Evaluation Guide](https://www.kaggle.com/competitions/your-competition-name/overview/evaluation)

