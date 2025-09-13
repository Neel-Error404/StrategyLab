#!/usr/bin/env python3
"""
Parquet to CSV Converter Script
Converts parquet files to CSV format with proper handling of financial data
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, List


def convert_parquet_to_csv(
    input_path: str,
    output_path: Optional[str] = None,
    preserve_index: bool = True
) -> bool:
    """
    Convert a single parquet file to CSV format
    
    Args:
        input_path: Path to input parquet file
        output_path: Path for output CSV file (optional)
        preserve_index: Whether to preserve the index in CSV
        
    Returns:
        bool: Success status
    """
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"Error: Input file {input_path} does not exist")
            return False
            
        if not input_file.suffix.lower() == '.parquet':
            print(f"Error: Input file must be a parquet file, got {input_file.suffix}")
            return False
            
        # Determine output path
        if output_path is None:
            output_file = input_file.with_suffix('.csv')
        else:
            output_file = Path(output_path)
            
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting {input_file} to {output_file}")
        
        # Read parquet file
        df = pd.read_parquet(input_file)
        
        # Convert to CSV
        df.to_csv(output_file, index=preserve_index)
        
        print(f"Successfully converted {input_file.name}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Output: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


def convert_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    preserve_index: bool = True
) -> int:
    """
    Convert all parquet files in a directory to CSV
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path (optional)
        recursive: Whether to search recursively
        preserve_index: Whether to preserve index in CSV
        
    Returns:
        int: Number of files successfully converted
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return 0
        
    # Find all parquet files
    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = list(input_path.glob(pattern))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return 0
        
    print(f"Found {len(parquet_files)} parquet file(s)")
    
    success_count = 0
    for parquet_file in parquet_files:
        # Determine output path
        if output_dir:
            output_path = Path(output_dir)
            if recursive:
                # Preserve relative directory structure
                rel_path = parquet_file.relative_to(input_path)
                output_file = output_path / rel_path.with_suffix('.csv')
            else:
                output_file = output_path / parquet_file.with_suffix('.csv').name
        else:
            output_file = parquet_file.with_suffix('.csv')
            
        if convert_parquet_to_csv(str(parquet_file), str(output_file), preserve_index):
            success_count += 1
            
    print(f"\nConversion complete: {success_count}/{len(parquet_files)} files converted")
    return success_count


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Convert parquet files to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python parquet_to_csv.py data/RELIANCE.parquet
  
  # Convert single file with custom output
  python parquet_to_csv.py data/RELIANCE.parquet -o output/RELIANCE.csv
  
  # Convert all parquet files in directory
  python parquet_to_csv.py data/ -d
  
  # Convert directory recursively with output directory
  python parquet_to_csv.py data/ -d -r -o csv_output/
  
  # Convert without preserving index
  python parquet_to_csv.py data/RELIANCE.parquet --no-index
        """
    )
    
    parser.add_argument(
        "input",
        help="Input parquet file or directory path"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file or directory path (optional)"
    )
    
    parser.add_argument(
        "-d", "--directory",
        action="store_true",
        help="Process directory instead of single file"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively (only with -d)"
    )
    
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Don't preserve index in CSV output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.recursive and not args.directory:
        print("Error: --recursive can only be used with --directory")
        sys.exit(1)
        
    preserve_index = not args.no_index
    
    try:
        if args.directory:
            success_count = convert_directory(
                args.input,
                args.output,
                args.recursive,
                preserve_index
            )
            sys.exit(0 if success_count > 0 else 1)
        else:
            success = convert_parquet_to_csv(
                args.input,
                args.output,
                preserve_index
            )
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()