import pandas as pd
import os
from pathlib import Path

def process_hq_countries():
    # Define input and output paths
    current_dir = Path(__file__).parent
    input_dir = current_dir / "input"
    output_dir = current_dir / "output"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Read the base dataset
    base_file = next(input_dir.glob("export_data_table*.xlsx"))
    base_df = pd.read_excel(base_file)
    
    # Read the institution data
    institution_file = input_dir / "result_institution.xlsx"
    institution_df = pd.read_excel(institution_file)
    
    # Get unique HQ countries per result_id
    unique_countries = institution_df.groupby('result_id')['HQ_country'].unique()
    
    # Convert the unique countries to comma-separated strings
    country_mapping = unique_countries.apply(lambda x: ', '.join(map(str, x)))
    
    # Add the new HQ_country column to the base dataset
    base_df['HQ_country'] = base_df['Result code'].map(country_mapping)
    
    # Save the enhanced dataset
    output_file = output_dir / f"enhanced_{base_file.name}"
    base_df.to_excel(output_file, index=False)
    
    print(f"Enhanced dataset saved to: {output_file}")

if __name__ == "__main__":
    process_hq_countries()

