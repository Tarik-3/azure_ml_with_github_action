"""
Simple ML Pipeline Runner - No Azure ML Required
Chains together prep.py -> train.py -> test.py
and stores outputs in outputs/ directory
"""

import subprocess
from pathlib import Path


def run_step(step_name: str, command: list) -> None:
    """Run a pipeline step and exit if it fails."""
    print("\n" + "=" * 60)
    print(f"Running: {step_name}")
    print("=" * 60)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"[FAILED] {step_name} failed with exit code {result.returncode}")
        exit(1)
    print(f"[SUCCESS] {step_name} completed")


def main():
    # Create outputs directory for pipeline artifacts
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Ensure input data exists
    input_data = Path("data/sample_data.csv")
    if not input_data.exists():
        print(f"❌ Error: Input data not found at {input_data}")
        exit(1)
    
    # Step 1: Data Preparation
    prep_output = output_dir / "prep_output"
    prep_output.mkdir(exist_ok=True)
    run_step(
        "Prep Step",
        ["python", "prep.py", 
         "--input", str(input_data),
         "--output", str(prep_output)]
    )
    
    # Step 2: Model Training
    train_output = output_dir / "train_output"
    train_output.mkdir(exist_ok=True)
    run_step(
        "Train Step",
        ["python", "train.py",
         "--input", str(prep_output),
         "--output", str(train_output)]
    )
    
    # Step 3: Model Testing
    run_step(
        "Test Step",
        ["python", "test.py",
         "--input", str(prep_output),
         "--model", str(train_output)]
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"Outputs saved to: {output_dir.absolute()}")
    print(f"  - prep_output/: Prepared data files")
    print(f"  - train_output/: Trained model files")
    print(f"  - Metrics: Check test output above")


if __name__ == "__main__":
    main()