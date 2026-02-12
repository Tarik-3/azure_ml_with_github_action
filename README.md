# ML Pipeline with GitHub Actions

A complete machine learning pipeline with **two workflows**:
1. **Training Pipeline** - Trains a new model from scratch
2. **Inference Pipeline** - Uses existing model to make predictions on new data

## Workflows

### 1️⃣ Training Pipeline (Automatic)
Trains a new model whenever you push to `main` branch.

**Jobs:**
- `prepare-data` - Loads CSV, splits train/test
- `train-model` - Trains LinearRegression model
- `test-model` - Evaluates model performance
- `summarize` - Creates report with metrics

**Trigger:** Automatic on push to `main`

### 2️⃣ Inference Pipeline (Manual)
Makes predictions using the latest trained model without retraining.

**Jobs:**
- `predict` - Loads latest model, generates predictions on new data

**Trigger:** Manual via GitHub Actions UI

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

### Training Pipeline

The training pipeline automatically runs whenever you push to the `main` branch. It uses **4 parallel jobs** for efficient execution:

1. **Prepare Data** - Loads and splits the CSV data
2. **Train Model** - Trains the ML model (waits for Prepare Data)
3. **Test Model** - Evaluates the model (waits for Train + Prepare)
4. **Summarize Results** - Uploads all artifacts (waits for all steps)

**How to trigger:**
```bash
git add .
git commit -m "My changes"
git push origin main
```

### Inference Pipeline

The inference pipeline is **triggered manually** when you want to make predictions with the existing trained model.

**How to trigger:**
1. Go to repository → **Actions** tab
2. Click "Inference Pipeline" on the left
3. Click "Run workflow" button
4. (Optional) Specify custom data file path
5. Click green "Run workflow" button

**Default behavior:**
- Uses the latest trained model from the Training Pipeline
- Makes predictions on `data/new_data.csv`
- Saves predictions as artifact `predictions-{run-number}`

**Custom data file:**
- You can specify any CSV file in your repo (e.g., `data/my_data.csv`)
- File must have the same columns as training data (f1, f2, f3, f4)
- No target column needed

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

**Training Pipeline results:**
- Check the **Job Summary** at the top of the workflow run
- Download artifacts:
  - `prep-output/` - Training/test data
  - `train-output/` - Model and metadata (kept for 90 days)
  - `metrics/` - Evaluation metrics

**Inference Pipeline results:**
- Check the **Job Summary** for prediction statistics
- Download artifact `predictions-{run-number}`:
  - `predictions.csv` - Original data + predictions column
  - `prediction_summary.json` - Mean, min, max, std

## Local Testing

### Run Training Locally
```bash
pip install -r requirements.txt
python pipeline.py
```

### Run Predictions Locally
```bash
# First, train a model (or use existing one)
python pipeline.py

# Then make predictions on new data
python predict.py \
  --model outputs/train_output \
  --input data/new_data.csv \
  --output outputs/predictions
```

Output includes:
- `predictions.csv` - Data with predictions
- `prediction_summary.json` - Statistics

## Project Structure

```
.
├── pipeline.py          # Training orchestrator
├── prep.py             # Data preparation
├── train.py            # Model training
├── test.py             # Model evaluation
├── predict.py          # Inference/predictions (NEW)
├── requirements.txt    # Dependencies
├── data/
│   ├── sample_data.csv     # Training data
│   └── new_data.csv        # New data for predictions (NEW)
├── outputs/            # All outputs
│   ├── prep_output/
│   ├── train_output/
│   ├── predictions/    # Prediction outputs (NEW)
│   └── metrics.json
├── .github/
│   └── workflows/
│       ├── azureml.yml     # Training pipeline
│       └── inference.yml   # Inference pipeline (NEW)
└── README.md
```

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
