# LTFAD


## Requirements
The recommended requirements for LFTSAD are specified as follows:
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
- For univariate datasets : You can download at (https://github.com/TheDatumOrg/TSB-UAD) and split them  60% into training set (_<datasaet>_train.npy) and 40% into test set (_<datasaet>_test.npy), and save the labels out as (< datasaet>_test_label.npy)
- For multivariate datasets : - [MSL](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/MSL/MSL_train.npy...`.
                              - [SMD](https://github.com/NetManAIOps/OmniAnomaly) should be placed at `datasets/SMD/SMD.csv`.
                              - [SMAP](https://en.wikipedia.org/wiki/Soil_Moisture_Active_Passive) should be placed at `datasets/SMAP/SMAP.csv`.
                              - [SwaT](https://drive.google.com/drive/folders/1ABZKdclka3e2NXBSxS9z2YF59p7g2Y5I) should be placed at `datasets/SwaT/SwaT.csv`.
