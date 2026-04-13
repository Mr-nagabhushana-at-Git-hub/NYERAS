# NYERAS

Retail Store Sales Insights and Prediction Model

A local retail analysis workspace for cleaning data, training a sales model, chatting with the model, and exploring results in a browser dashboard.

## One Paste To Run Everything

If you just want to run the full setup and see the project working, do this first:

```bat
START_APP.bat
```

Keep internet on while it runs.

What happens behind the scenes:

1. If Conda/Anaconda is not available, the launcher installs Miniconda automatically.
2. It creates the project virtual environment.
3. It installs the required Python packages from `requirements.txt`.
4. It starts the dashboard.
5. From there, the auto-scraping / auto-analysis flow can run from the dashboard or notebook workflow.

If you prefer the notebook path for auto scraping and model build, open `retail_auto_scrape_clean_model.ipynb` after the launcher finishes and run the cells in order.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Mr-nagabhushana-at-Git-hub/NYERAS.git
cd NYERAS
```

### 2. Run the one-click launcher

```bat
START_APP.bat
```

This will:

- install Miniconda if Conda/Anaconda is missing
- create a dedicated project environment
- install the Python dependencies from `requirements.txt`
- start the dashboard
- open `http://127.0.0.1:7860/` in your browser
- keep the console open so you can watch logs and errors

### 3. Run the dashboard directly

If you already have Python and dependencies installed:

```bash
python retail_ui_server.py
```

### 4. Open the notebook workflow

```bash
jupyter notebook retail_auto_scrape_clean_model.ipynb
```

## What Each File Does

| File | Purpose |
| --- | --- |
| `phase1_retail_insights.py` | Main analysis script that cleans the data, builds plots, trains the model, and writes the report |
| `retail_auto_scrape_clean_model.ipynb` | Notebook version for Colab or local Jupyter |
| `retail_ui_server.py` | Browser dashboard with workflow status, chat, and interactive plots |
| `START_APP.ps1` | Bootstrap script that checks for Conda, installs Miniconda if needed, creates the env, installs dependencies, starts the dashboard, and opens the browser |
| `START_APP.bat` | Windows double-click entry point that keeps the console visible |
| `launch.bat` | Compatibility wrapper that forwards to `START_APP.bat` |
| `requirements.txt` | Python dependencies used by the script, notebook, and dashboard |
| `outputs/` | Generated CSVs, plots, models, and reports |

## Dashboard Buttons

- `Run Full Flow`: load data, clean it, run EDA, and train the model in one pass
- `Use Demo Data`: use the built-in synthetic retail dataset if no file is uploaded
- `Refresh Status`: pull the latest workflow state from the server
- `Upload Data`: choose a custom dataset in manual mode

## Launcher Behavior

`START_APP.ps1` does the real work. It:

1. Checks whether Conda is already installed.
2. Installs Miniconda silently if Conda is missing.
3. Creates or reuses the project environment in `%LOCALAPPDATA%\NYERAS\retail-ui`.
4. Installs or refreshes the packages from `requirements.txt`.
5. Starts `retail_ui_server.py` in the same console so the logs stay visible.
6. Opens `http://127.0.0.1:7860/` in the browser.

`START_APP.bat` is the file you double-click on Windows.
`launch.bat` is just a compatibility alias for `START_APP.bat`.

## Copy-Paste Commands

### Run the analysis script on your own CSV

```bash
python phase1_retail_insights.py --input path/to/your_dataset.csv --output-dir outputs
```

### Run the demo analysis

```bash
python phase1_retail_insights.py --output-dir outputs --demo-rows 1200 --seed 42
```

### Open the dashboard with one double-click

```bat
START_APP.bat
```

### Run the notebook in Jupyter

```bash
jupyter notebook retail_auto_scrape_clean_model.ipynb
```

## Outputs

After a run, look in `outputs/` for:

- `cleaned_retail_data.csv`
- `phase1_report.md`
- `plots/`
- `models/`

Note: `outputs/` is ignored by git, but you do not need to create it by hand. The scripts and dashboard create it automatically when you run the project.

## Notes

- The script auto-detects the relevant business columns by name and can handle common variations like `revenue`, `amount`, `qty`, `unit_price`, `order_date`, `product_name`, and `customer_id`.
- If the dataset does not contain a clear customer identifier or date field, the report will automatically fall back to the strongest available transaction-level analysis.
- The dashboard is designed for non-technical users.
- The cleaned outputs and report are meant to be easy to export or share.
