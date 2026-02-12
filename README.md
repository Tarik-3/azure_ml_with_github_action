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

## GitHub Actions (Automated)

The pipeline automatically runs whenever you push to the `main` branch.

### 1. Push Your Code
```bash
git add .
git commit -m "My changes"
git push origin main
```

### 2. View Results
- Go to repository → **Actions** tab
- Click the latest "ML Pipeline" workflow run
- Scroll down to **Artifacts** section
- Download `pipeline-outputs` (contains outputs/)

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
