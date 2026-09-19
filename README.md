# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## 1. Project Overview

Mobile network operators need short-term traffic forecasts to plan capacity and balance load. This project is an empirical study of one-step-ahead Internet-traffic forecasting on the Telecom Italia "Milan CDR" dataset (telecommunications activity in 10,000 grid squares of Milan, recorded every 10 minutes) [1].

**Research question:** How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

Three model families are implemented, tuned and compared on the three busiest squares: a statistical baseline (SARIMA, implemented as a regression with Fourier seasonal terms and ARIMA errors), a recurrent network (LSTM) and a Temporal Convolutional Network (TCN) written from scratch in PyTorch.

The full write-up (related work, methodology, results and discussion) is in `report/report.pdf` (source: `report/report.md`).

## 2. Data Window

The raw archive in `data/raw/` covers 2013-11-01 to 2014-01-01 (62 daily files). All exploratory analysis and the ranking of the squares use this full period. Forecasting is evaluated on the assignment week, 2013-12-16 to 2013-12-22 (1,008 ten-minute steps), with models trained on 2013-11-01 to 2013-12-15. These dates are set as constants at the top of `notebooks/02_modeling_and_results.ipynb` (`TRAIN_START`, `TRAIN_END`, `TEST_START`, `TEST_END`).

## 3. Dataset

Each daily raw file (`data/raw/sms-call-internet-mi-YYYY-MM-DD.txt`) is tab-separated with no header and one row per (square, time interval, country code):

```
square_id  timestamp_ms  country_code  sms_in  sms_out  call_in  call_out  internet
```

A square and interval can appear in several rows (one per country code), and many fields are empty on any given row. A single day has about 4.8 to 5.3 million rows (4,842,625 on Nov 1 and 5,320,759 on Dec 16) and the files are 277 to 376 MB each, 20.8 GB in total. The forecasting target is the `internet` column, summed over the country codes of each (square, interval).

## 4. Data Handling & Memory Management

The whole archive does not fit in the 7.7 GB of RAM of the machine used, so `notebooks/01_data_and_eda.ipynb` reads it in two chunked passes (2,000,000 rows per chunk), reading only the columns it needs with smaller dtypes:

- **Pass 1 (ranking):** reads 2 columns (`square_id`, `internet`) of every file and keeps a running total for each of the 10,000 squares. This gives the ranking of all squares by total traffic.
- **Pass 2 (extraction):** reads 3 columns (`square_id`, `timestamp_ms`, `internet`) of every file and keeps only the top-3 squares and squares 4159 and 4556, which are rebuilt as complete 10-minute series (missing intervals filled with 0).

Measured evidence (`results/memory_usage.csv` and the size cell of notebook 1, process memory measured with `psutil`):

- One day's DataFrame (Nov 1, 4,842,625 rows) is 296 MB with all 8 columns and default dtypes, 37 MB with the Pass 1 columns and dtypes (87% smaller) and 74 MB with the Pass 2 columns (75% smaller).
- Loading that one day in full raised the process memory (RSS) from 204.0 MB to 500.4 MB (+296.4 MB, 3.2 s).
- Scanning all 62 files in chunks ended at only +13.6 MB (Pass 1, 136 s) and +28.2 MB (Pass 2, 155 s) above the baseline.

Trade-offs: the data is read twice (about twice the I/O), the chunk size is fixed and RSS is measured after each stage, not at its peak. These memory and timing numbers change from run to run. The details are in the report (Dataset and Data Preparation).

## 5. Models Implemented and Justification

| # | Model | Notebook section | Justification |
|---|-------|------------------|---------------|
| 1 | SARIMA (Fourier terms + ARIMA errors) | `02_modeling_and_results.ipynb`, Model 1 | Classical statistical baseline. The exploratory analysis shows a strong daily and weekly cycle (ACF 0.878 at lag 144 and 0.838 at lag 1008 for square 5161). The seasonality is modeled with sin/cos regressors instead of seasonal orders, because `SARIMAX` with seasonal period 144 was too slow to tune. |
| 2 | LSTM | `02_modeling_and_results.ipynb`, Model 2 | Recurrent network on a one-day window with time-of-day and day-of-week features; a standard model in the cellular traffic literature. |
| 3 | TCN | `02_modeling_and_results.ipynb`, Model 3 | Dilated causal convolutions with residual connections, structurally different from both SARIMA and the LSTM (non-recurrent, receptive field of 253 steps). |

The full justification, tied to the exploratory analysis and the literature, is in the report (Related Work and Methodology).

## 6. Installation & Setup

### Prerequisites

- Python 3.11 (used for development)
- About 21 GB of free disk for the raw archive; 8 GB of RAM recommended (the run used 7.7 GB)
- No GPU needed: everything runs on the CPU

### Setup

```bash
git clone https://github.com/amaliza-shal/Time-Series-Forecasting1
cd Time-Series-Forecasting1
py -3.11 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ts-forecasting --display-name "ts-forecasting"
```

Place the raw daily files (`sms-call-internet-mi-YYYY-MM-DD.txt`) in `data/raw/`. The dataset is available from the Harvard Dataverse (references [2] and [3] below); a short download form is required. `data/` is not tracked by git.

## 7. How to Run the Pipeline

The pipeline is 2 notebooks, run in order. Notebook 2 reads only the files written by notebook 1, never the raw files. There is no separate script: each notebook defines its own helper functions. Run them from Jupyter or headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=ts-forecasting notebooks/01_data_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=ts-forecasting notebooks/02_modeling_and_results.ipynb
```

| Stage | Notebook | Produces |
|-------|----------|----------|
| 1. Data and EDA | `01_data_and_eda.ipynb` | `data/processed/timeseries_selected_squares.parquet`, `results/{square_ranking, memory_usage, stl_strength_top1, adf_test_top1}.csv`, `figures/eda_*.png` (6 figures) |
| 2. Modeling and results | `02_modeling_and_results.ipynb` | `results/{sarima, lstm, tcn}_grid_search.csv`, `results/metrics_square_{5161, 5059, 5259}.csv`, `results/timing.csv`, `figures/results_*.png` (9 actual-vs-predicted plots, a 3x3 grid and the worst-case plot) |

Notebook 1 takes about 10 minutes (it reads the 20.8 GB twice). Notebook 2 took about an hour on the CPU, mostly the LSTM and TCN grid searches and training.

Do not run a notebook while another program (for example VS Code) has it open with unsaved changes, and do not run both notebooks at the same time: the notebooks overwrite the files in `results/` and `figures/`, and saving an older copy of a notebook from an editor would overwrite a finished run. The numbers quoted in the report (memory, timings and everything derived from them) belong to one particular run.

## 8. Results Summary

See `results/` for the tables and `report/report.pdf` for the write-up. One-step-ahead errors over Dec 16-22, 2013 (walk-forward, one model per square):

| Square | Model | MAE | MAPE (%) | RMSE |
|--------|-------|-----|----------|------|
| 5161 | SARIMA | 82.7 | 9.4 | 122.8 |
| 5161 | LSTM | 98.9 | 17.2 | 129.2 |
| 5161 | TCN | 87.9 | 10.0 | 130.9 |
| 5059 | SARIMA | 69.5 | 7.7 | 97.8 |
| 5059 | LSTM | 73.6 | 7.9 | 101.4 |
| 5059 | TCN | 79.0 | 9.0 | 106.1 |
| 5259 | SARIMA | 68.7 | 9.1 | 92.9 |
| 5259 | LSTM | 71.8 | 9.8 | 96.9 |
| 5259 | TCN | 72.3 | 10.4 | 96.1 |

Headline findings: SARIMA is the best model on all three squares by MAE, MAPE and RMSE, but not by a large margin (the MAE of the best neural model is 4.6% to 6.3% higher). The LSTM is the weakest model on square 5161 (MAPE 17.2%, against 9.4% for SARIMA), with predictions above the actual values in the overnight troughs (judged from the plots). The worst case overall is the TCN on square 5161 on Tue Dec 17, where it under-predicts the midday peak. The daily seasonal strength (0.858, 0.891 and 0.497 for the three squares) does not explain the differences between squares. The explanations are given, with their limits, in the report (Results and Discussion, Conclusion).

## 9. Repository Structure

```
├── data/
│   ├── raw/               # daily CDR files (gitignored, see section 3)
│   └── processed/         # timeseries_selected_squares.parquet (gitignored, regenerate with notebook 1)
├── notebooks/             # the 2-notebook pipeline (run in order 01 -> 02)
│   ├── 01_data_and_eda.ipynb
│   └── 02_modeling_and_results.ipynb
├── figures/               # all plots produced by the notebooks
├── results/               # all tables produced by the notebooks
├── report/                # report.md (source), report.pdf and report.html (rendered from it)
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## 10. Reproducibility

The random seed is fixed (`SEED = 42`) for NumPy and PyTorch, and the PyTorch seed is set again before each model is built. Training runs on the CPU with fixed data windows. In practice the LSTM results were identical (to all digits) in repeated runs on the same machine, and SARIMA is deterministic given the data. Timings and memory numbers vary from run to run and are single measurements. TensorFlow and XGBoost are not used.

## 11. References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Scientific Data*, vol. 2, 150055, 2015. https://doi.org/10.1038/sdata.2015.55

[2] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

[3] Telecom Italia, "Milano Grid," Harvard Dataverse. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

## 12. Author and Date

Author: Amaliza Shalom
Date: September 2026
Course/Assignment: Machine Learning Techniques — Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting
