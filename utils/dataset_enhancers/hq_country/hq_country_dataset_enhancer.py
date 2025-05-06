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
    base_file = next(input_dir.glob("export_data_table_results_20250605_154958CET.xlsx"))
    base_df = pd.read_excel(base_file)
    
    # Read the institution data
    institution_file = input_dir / "result_institution_20250605.xlsx"
    institution_df = pd.read_excel(institution_file)
    
    # Read the mapping file and get highest result_id per result_code
    mapping_file = input_dir / "result_code_id_mapping_20250605.csv"
    mapping_df = pd.read_csv(mapping_file)
    highest_ids = mapping_df.groupby('result_code')['result_id'].max()
    
    # Get all HQ countries per result_id (including duplicates)
    unique_countries = institution_df.groupby('result_id')['HQ_country'].apply(list)
    
    # Convert the countries to semicolon-separated strings
    country_mapping = unique_countries.apply(lambda x: '; '.join(map(str, x)))
    
    # First map result_codes to their highest result_ids, then map to countries
    result_id_mapping = base_df['Result code'].map(highest_ids)
    base_df['HQ_country'] = result_id_mapping.map(country_mapping)
    
    # Save the enhanced dataset
    output_file = output_dir / f"enhanced_{base_file.name}"
    base_df.to_excel(output_file, index=False)
    
    print(f"Enhanced dataset saved to: {output_file}")

if __name__ == "__main__":
    process_hq_countries()