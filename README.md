# JLSAE

Official implementation of **JLSAE: Joint Low-Rank and Sparse Autoencoder for Hyperspectral Anomaly Detection**.

JLSAE jointly models the low-rank structure of the background and the sparse characteristics of anomalies within a unified autoencoder framework.

This repository provides a simple implementation for reproducing and comparing hyperspectral anomaly detection methods.

## Framework

<p align="center">
  <img src="figures/JLSAE_framework.png" width="900"><br>
  <em>Overall framework of JLSAE.</em>
</p>

## Requirements

A standard PyTorch environment is sufficient to run the code.

The main required packages include:

```text
Python
PyTorch
NumPy
SciPy
Matplotlib
```

A CUDA-enabled PyTorch environment is recommended for GPU acceleration.

## Data Preparation

Place the test hyperspectral data in the `data` folder.

Each data file should be stored in MATLAB `.mat` format and contain the following variables:

- `data`: hyperspectral image data
- `map`: ground-truth anomaly map

An example directory structure is:

```text
JLSAE/
├── data/
│   └── example.mat
├── models/
├── utils/
├── result/
└── train.py
```

Before running the code, modify the data file name in `train.py`:

```python
path = 'data/'
file_name = 'example.mat'
```

## Running the Code

Run the following command:

```bash
python train.py
```

## Results

The detection results are automatically saved in the `result` folder.

The saved `.mat` file contains:

- `detection`: anomaly detection map
- `auc`: area under the ROC curve

For example:

```text
result/example_result.mat
```

## Parameter Settings

The following parameters can be adjusted in `train.py`:

### Number of Iterations

```python
num_iter = 150
```

`num_iter` controls the number of optimization iterations. The default value for a new data set is `150`. It can be adjusted according to the convergence behavior and detection performance.

Recommended candidate values include:

```python
[100, 150, 200, 250, 300]
```

### Penalty Parameter

```python
pho = 1
```

`pho` is the penalty parameter used in the optimization process. If the detection result is unsatisfactory, it can be tuned using values such as:

```python
[1, 1.25, 1.5, 1.75]
```

### Trade-Off Parameter

```python
lam = 0.5
```

`lam` is the trade-off parameter associated with the sparse anomaly component. It can be adjusted for different data sets to obtain better detection performance.

## Reproducibility

The network parameters and the input noise tensor `Z` are randomly initialized for each run. Therefore, the detection results and AUC values may vary slightly across different runs.

For a fair comparison, it is recommended to run the method multiple times or fix the random seed before training.

## Detection Results

<table>
  <tr>
    <td align="center">
      <img src="figures/Original_HSI.png" width="180"><br>
      <b>Original HSI</b>
    </td>
    <td align="center">
      <img src="figures/map.png" width="180"><br>
      <b>Ground Truth</b>
    </td>
    <td align="center">
      <img src="figures/Reconstructed_background.png" width="180"><br>
      <b>Reconstructed Background</b>
    </td>
    <td align="center">
      <img src="figures/Anomaly_E.png" width="180"><br>
      <b>Anomaly Component E</b>
    </td>
    <td align="center">
      <img src="figures/detection.png" width="180"><br>
      <b>Detection Map</b>
    </td>
  </tr>
</table>
