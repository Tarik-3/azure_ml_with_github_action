# ML Pipeline with GitHub Actions

A complete machine learning pipeline with **two workflows**:
1. **Training Pipeline** - Manually retrain model from scratch (on-demand)
2. **Inference Pipeline** - Automatically runs predictions on new data (on push)

## Workflows

### 1️⃣ Training Pipeline (Manual)
Retrains the model when you need to update it with new training data.

**Jobs:**
- `prepare-data` - Loads CSV, splits train/test
- `train-model` - Trains LinearRegression model
- `test-model` - Evaluates model performance
- `summarize` - Creates report with metrics

**Trigger:** Manual via GitHub Actions UI

### 2️⃣ Inference Pipeline (Automatic)
Runs predictions using the existing trained model - **No retraining needed!**

**Jobs:**
- `prepare-inference-data` - Cleans and validates raw data
- `get-model` - Downloads latest trained model
- `predict` - Generates predictions on new data
- `results` - Summarizes prediction results

**Trigger:** Automatic on push to `main` (when data files change) OR manual

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

## GitHub Actions Workflows

### Training Pipeline (Manual Retrain)

**When to use:** When you want to retrain the model with new training data or different parameters.

**How to trigger:**
1. Go to repository → **Actions** tab
2. Click "**Training Pipeline**" on the left
3. Click "**Run workflow**" button
4. (Optional) Enter reason for retraining
5. Click green "**Run workflow**" button

**What happens:**
- Runs full training pipeline with 4 jobs
- Model saved for 90 days as artifact `train-output`
- Metrics and test results available as artifacts

### Inference Pipeline (Automatic Predictions)

**When to use:** When you have new data to predict on (no retraining needed).

**How it triggers automatically:**
```bash
# Add new data to predict on
echo "5.2,3.1,2.4,1.8" >> data/new_data.csv

# Push to main branch
git add data/new_data.csv
git commit -m "Add new data for predictions"
git push origin main
```

The pipeline automatically runs when you push changes to:
- `data/**` (any data files)
- `predict.py` (prediction script)
- `.github/workflows/inference.yml` (workflow itself)

**Manual trigger (optional):**
1. Go to repository → **Actions** tab
2. Click "**Inference Pipeline**" on the left
3. Click "**Run workflow**" button
4. (Optional) Specify custom data file path
5. Click green "**Run workflow**" button

**What happens:**
- **Job 1**: Prepares and cleans raw data (handles missing values)
- **Job 2**: Downloads latest trained model from Training Pipeline
- **Job 3**: Generates predictions on cleaned data
- **Job 4**: Creates summary report with statistics

**Workflow Dependency Graph:**
```
prepare-inference-data ──→ get-model ──→ predict ──→ results
```

### View Results

**Training Pipeline artifacts:**
- `prep-output/` - Training/test data
- `train-output/` - Model and metadata (kept for 90 days)
- `metrics/` - Evaluation metrics

**Inference Pipeline artifacts:**
- `inference-data/` - Cleaned input data
- `model/` - Trained model used for predictions
- `predictions/` - Results:
  - `predictions.csv` - Original data + predictions column
  - `prediction_summary.json` - Mean, min, max, std

Check the **Job Summary** at the top of each workflow run for quick statistics!

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

**Inference Pipeline not running automatically**
- Make sure you're pushing data file changes to `main` branch
- Check that files changed are in `data/` directory
- Verify `.github/workflows/inference.yml` exists

**"Model not found" error**
- You need to run Training Pipeline at least once to create a model
- Check that `train-output` artifact exists from a previous training run
- Model artifacts are kept for 90 days

**Prediction fails with column mismatch**
- New data must have same columns as training data (f1, f2, f3, f4)
- Do NOT include target column in prediction data
- Check that column names match exactly (case-sensitive)

**Pipeline fails locally**
- Make sure `data/sample_data.csv` exists for training
- Ensure you have trained a model before running predictions
- Check all dependencies are installed: `pip install -r requirements.txt`

## Dependencies

- Python 3.9+
- scikit-learn
- pandas
- numpy
- joblib
