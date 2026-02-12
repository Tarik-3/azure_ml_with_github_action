# ML Pipeline with GitHub Actions

A simple machine learning pipeline that runs on GitHub Actions without requiring Azure infrastructure.

## What it does

The pipeline automates three ML steps:

1. **Prep Step** (`prep.py`): Loads CSV data, handles missing values, splits into train/test sets
2. **Train Step** (`train.py`): Trains a linear regression model on training data
3. **Test Step** (`test.py`): Evaluates the model on test set and computes metrics (MSE, RMSE, MAE, R²)

## Local Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python pipeline.py
```

Outputs will be saved to `outputs/`:
- `prep_output/`: Train/test CSV files
- `train_output/`: Trained model (`model.pkl`) and metadata
- `metrics.json`: Evaluation metrics

## GitHub Actions (Automated with Parallel Jobs)

The pipeline automatically runs whenever you push to the `main` branch. It uses **4 parallel jobs** for efficient execution:

1. **Prepare Data** - Loads and splits the CSV data
2. **Train Model** - Trains the ML model (waits for Prepare Data)
3. **Test Model** - Evaluates the model (waits for Train + Prepare)
4. **Summarize Results** - Uploads all artifacts (waits for all steps)

### 1. Push Your Code
```bash
git add .
git commit -m "My changes"
git push origin main
```

### 2. Monitor Jobs
- Go to repository → **Actions** tab
- Click the latest "ML Pipeline" workflow run
- You'll see 4 jobs running:
  - `prepare-data` (runs first)
  - `train-model` (starts after prepare-data completes)
  - `test-model` (starts after train-model completes)
  - `summarize` (runs after all jobs complete)

**Workflow Dependency Graph:**
```
prepare-data ──→ train-model ──→ test-model ──┐
    ↓                                          │
    └──────────────────────────────────────→ summarize
```

### 3. View Results
- Check the **Job Summary** that appears at the top of the workflow run
- Each job logs its output
- Scroll down to **Artifacts** section to download results:
  - `prep-output/` - Training/test data
  - `train-output/` - Model and metadata
  - `metrics/` - Evaluation metrics

## Troubleshooting

**Pipeline fails locally**
- Make sure `data/sample_data.csv` exists

**GitHub Actions not running**
- Verify `.github/workflows/azureml.yml` exists
- Check you're pushing to `main` branch
- View error logs in Actions tab

## Dependencies

- Python 3.9+
- scikit-learn
- pandas
- numpy
- joblib
