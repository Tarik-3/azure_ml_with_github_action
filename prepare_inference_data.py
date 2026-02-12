"""
Prepare and validate inference data for prediction pipeline.
Loads raw data, checks for missing values, and saves cleaned version.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd


def prepare_inference_data(input_file: str, output_dir: str) -> None:
    """
    Validate and prepare inference data.
    
    Args:
        input_file: Path to raw input CSV file
        output_dir: Directory to save cleaned data
    """
    try:
        # Load raw data
        print(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n⚠️  Missing values found:")
            print(missing[missing > 0])
            
            # Drop rows with missing values
            original_count = len(df)
            df = df.dropna()
            dropped = original_count - len(df)
            print(f"Dropped {dropped} rows with missing values")
            print(f"After cleaning: {len(df)} rows remaining")
        else:
            print("✅ No missing values found")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        output_file = output_path / 'cleaned_data.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ SUCCESS: Cleaned data saved to {output_file}")
        print(f"Final dataset: {len(df)} rows × {len(df.columns)} columns")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found: {input_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: Input file is empty: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to prepare data: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare and validate inference data for predictions'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directory to save cleaned data'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INFERENCE DATA PREPARATION")
    print("=" * 60)
    
    prepare_inference_data(args.input, args.output)
    
    print("=" * 60)


if __name__ == '__main__':
    main()
