# A lightweight All-MLP time–frequency anomaly detection for IIoT time series
This repository provides a PyTorch implementation of LTFAD ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608025002795?via%3Dihub)).
## Framework
<img src="https://github.com/infogroup502/LTFAD/blob/main/img/workflow.png" width="850px">

## Main Result
<img src="https://github.com/infogroup502/LTFAD/blob/main/img/result2.png" width="850px">
<img src="https://github.com/infogroup502/LTFAD/blob/main/img/result1.png" width="850px">

## Requirements
The recommended requirements for LTFAD are specified as follows:
- torch==1.13.0
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1
- matplotlib==3.9.2
- statsmodels==0.14.2
- tsfresh==0.20.3
- hurst==0.0.5
- arch==7.0.0

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data
The datasets can be obtained and put into datasets/ folder in the following way:
- Our model supports anomaly detection for univariate and multivariate time series datasets.
- We provide some dataset. If you want to use your own dataset, please place your datasetfiles in the `/dataset/<dataset>/` folder, following the format `<dataset>_train.npy`, `<dataset>_test.npy`, `<dataset>_test_label.npy`.

## Code Description
There are six files/folders in the source
- data_factory: The preprocessing folder/file. All datasets preprocessing codes are here.
- main.py: The main python file. You can adjustment all parameters in there.
- metrics: There is the evaluation metrics code folder.
- model: LTFAD model folder
- solver.py: Another python file. The training, validation, and testing processing are all in there
- requirements.txt: Python packages needed to run this repo

- ## Usage
1. Install Python 3.9, PyTorch >= 1.4.0
2. Download the datasets
3. To train and evaluate LTFAD on a dataset, run the following command:
```bash
python main.py 
```
## BibTex Citation
```bash
@article{CHEN2025107400,
title = {A lightweight All-MLP time–frequency anomaly detection for IIoT time series},
journal = {Neural Networks},
volume = {187},
pages = {107400},
year = {2025},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2025.107400},
url = {https://www.sciencedirect.com/science/article/pii/S0893608025002795},
author = {Lei Chen and Xinzhe Cao and Tingqin He and Yepeng Xu and Xuxin Liu and Bowen hu},
keywords = {Industrial Internet of Things (IIoT), Time series, Anomaly detection, Time–frequency joint learning, All-MLP architecture, Lightweight network}
}

```
