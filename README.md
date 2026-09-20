# Mobile Network Traffic Forecasting Analysis.

## 1. Project Overview

I compared three forecasting models on mobile Internet traffic from the Telecom Italia Milan dataset (10,000 grid squares, one value every 10 minutes) [1]. The models predict the next 10 minutes of traffic, using only the real past values.

**Research question:** How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

The models are SARIMA (a statistical model), an LSTM (a recurrent neural network) and a TCN (a convolutional network I wrote myself in PyTorch). I tested them on the three busiest squares. The full write-up is in `report/report.pdf`.

## 2. Data Window

The raw data covers 2013-11-01 to 2014-01-01 (62 daily files). I trained on 2013-11-01 to 2013-12-15 and tested on the week 2013-12-16 to 2013-12-22 (1,008 steps).

## 3. Dataset

Each daily file is tab-separated, has no header and has one row per square, time and country code:

```
square_id  timestamp_ms  country_code  sms_in  sms_out  call_in  call_out  internet
```

I forecast the `internet` column, added up over the country codes. A day has about 5 million rows, and the 62 files are 20.8 GB in total.

## 4. Data Handling & Memory Management

The data is much bigger than the 7.7 GB of RAM on my computer, so I read it in chunks, twice, and only the columns I need:

- **Pass 1:** finds the total traffic of every square, so I can pick the busiest ones.
- **Pass 2:** keeps only the 5 squares I use (the top 3, plus 4159 and 4556).

What I measured (`results/memory_usage.csv`):

- One day loaded normally takes 296 MB. With only the columns I need and smaller data types it takes 74 MB (75% less).
- Loading one day normally added 296 MB of memory. Reading all 62 files in chunks added only 14 MB and 28 MB.

## 5. Models Implemented and Justification

| Model | Why I chose it |
|-------|----------------|
| SARIMA | A simple statistical baseline. The traffic has a very strong daily and weekly cycle, which it can model directly. |
| LSTM | A standard neural network for time series, also used in earlier work on mobile traffic (see the report). |
| TCN | A different kind of network (convolutions instead of recurrence), so the comparison covers three model types. |

The reasons in detail, with the literature, are in the report.

## 6. Installation & Setup

You need Python 3.11, about 21 GB of free disk for the raw data and 8 GB of RAM. No GPU is needed.

```bash
git clone https://github.com/amaliza-shal/Time-Series-Forecasting1
cd Time-Series-Forecasting1
py -3.11 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ts-forecasting --display-name "ts-forecasting"
```

Put the raw files (`sms-call-internet-mi-YYYY-MM-DD.txt`) in `data/raw/`. Download them from the Harvard Dataverse (references [2] and [3]); a short form is required. The `data/` folder is not on GitHub.

## 7. How to Run the Pipeline

Run the two notebooks in order:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=ts-forecasting notebooks/01_data_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=ts-forecasting notebooks/02_modeling_and_results.ipynb
```

| Notebook | What it does | Time |
|----------|--------------|------|
| `01_data_and_eda.ipynb` | Loads the data, ranks the squares, makes the exploratory plots | about 10 minutes |
| `02_modeling_and_results.ipynb` | Tunes and tests the 3 models, makes the results tables and plots | about 1 hour |

Notebook 2 only needs the files that notebook 1 saves. Do not run both at the same time, because they write to the same `results/` and `figures/` folders.

## 8. Results Summary

Errors on the test week (Dec 16-22, 2013), one model per square. Lower is better:

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

- SARIMA was the best model on all three squares, but only by a small margin (the best neural model had a 4.6% to 6.3% higher MAE).
- The LSTM was the weakest on square 5161 (MAPE 17.2%, against 9.4% for SARIMA).
- The biggest single failure was the TCN on square 5161 on Tuesday Dec 17: it predicted the midday peak too low.

**What I ran into and learned**

- **Getting the data was hard.** The dataset is 20.8 GB, so I had to find flash disks to copy it to my computer before I could start.
- **Notebook 2 takes a very long time.** It runs for about an hour, and every change to the models meant running it again.
- **The SARIMA I planned was too slow.** SARIMA with a daily cycle of 144 steps did not finish after one hour. I replaced the seasonal part with sine and cosine terms, and it then took only a few seconds per fit.
- **My first tuning was too quick to trust.** I first tuned the neural networks with only 5 training rounds, and this made a different model the best on one square. I repeated it with the same 20 rounds as the final training.
- **Timings are noisy.** The same LSTM took about 61 s on two squares and 312 s on the third, so I only trust large differences.
- **I expected the neural networks to win, but they did not.** The simpler model was best, and I could not explain the differences between squares by the strength of the daily cycle. The report says which explanations I tested and which I did not.

## 9. Repository Structure

```
├── data/
│   ├── raw/          # daily CDR files (not on GitHub)
│   └── processed/    # series written by notebook 1 (not on GitHub)
├── notebooks/        # 01_data_and_eda.ipynb, 02_modeling_and_results.ipynb
├── figures/          # all plots
├── results/          # all tables
├── report/           # report.pdf (final), report.md, report.html
├── requirements.txt
└── README.md
```

## 10. Reproducibility

I fixed the random seed (42) for NumPy and PyTorch. Running the notebooks again gave the same errors for the LSTM and SARIMA. Timings and memory numbers change every run, so the ones in the report come from one run.

## 11. References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Scientific Data*, vol. 2, 150055, 2015. https://doi.org/10.1038/sdata.2015.55

[2] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

[3] Telecom Italia, "Milano Grid," Harvard Dataverse. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

## 12. Author and Date

Author: Amaliza Shalom
Date: September 2026
Course/Assignment: Machine Learning Techniques — Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting
