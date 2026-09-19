# Mobile Network Traffic Forecasting — Milan CDR Grid

One-step-ahead forecasting of Internet traffic on the Milan Telecom Italia CDR grid
(Nov 1, 2013 – Jan 1, 2014, 10-minute intervals, 10,000 squares). The evaluation week is
Dec 16–22, 2013, on the three busiest squares.

The work is split into two notebooks that follow the report sections:

- `notebooks/01_data_and_eda.ipynb` — dataset and data preparation, exploratory analysis.
  Loads the raw files in two chunked passes, ranks all squares by total traffic, builds the
  10-minute series for the top-3 squares plus squares 4159 and 4556, and runs the EDA
  (distribution, STL decomposition, ACF/PACF, ADF test).
- `notebooks/02_modeling_and_results.ipynb` — methodology, results and discussion. Trains
  SARIMA, an LSTM and a hand-written TCN per square (with a two-stage hyperparameter search) and
  evaluates them walk-forward over the test week.

Each notebook defines its own helper functions; there is no separate scripts folder.

## Project structure

```
data/raw/         raw daily .txt files
data/processed/   parquet time series written by notebook 1, read by notebook 2
notebooks/        the two notebooks
figures/          figures from both notebooks
results/          CSV tables (ranking, memory usage, STL/ADF, grid searches, metrics, timing)
report/           report.md (source), report.pdf and report.html (rendered from it)
requirements.txt
```

## Data

The raw data is the Telecom Italia Big Data Challenge Milan CDR set [1 in the report], one tab-separated file per day named `sms-call-internet-mi-YYYY-MM-DD.txt`. Place the 62 files (2013-11-01 to 2014-01-01, about 20 GB) in `data/raw/`; the folder is not tracked by git.

## Setup

Python 3.11 was used.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name ts-forecasting --display-name "ts-forecasting"
```

## Running

Run the notebooks in order with the `ts-forecasting` kernel. Notebook 2 only reads what
notebook 1 saved, never the raw files.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling_and_results.ipynb
```

Notebook 1 takes about 10 minutes because it reads the ~20 GB of raw files twice. Notebook 2
runs on the CPU and took about an hour, mostly the LSTM and TCN grid searches and training.
Timing and memory numbers change from run to run, so the numbers in the report are those of
one particular run.
