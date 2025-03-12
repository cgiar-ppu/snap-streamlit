import pandas as pd
from collections import Counter

def get_regional_initiative(submitter, initiative_to_ri):
    """
    Get Regional Initiative mapping from a submitter code
    """
    if pd.isna(submitter):
        return None
        
    # Clean submitter code
    submitter = str(submitter).strip()
    
    # Return Regional Initiative mapping if submitter exists in mapping
    return initiative_to_ri.get(submitter)

def main():
    # Define direct mapping from initiatives to Regional Integrated Initiatives
    initiative_to_ri = {
        'INIT-01': 'GI',  # Accelerated Breeding
        'INIT-03': 'GI',  # Genebanks
        'INIT-04': 'GI',  # Breeding Resources
        'INIT-05': 'GI',  # Market Intelligence
        'INIT-06': 'GI',  # Seed Equal
        'INIT-07': 'RAFS',  # One Health
        'INIT-10': 'Regional Integrated Initiative',  # Fragility to Resilience
        'INIT-11': 'RAFS',  # Excellence in Agronomy
        'INIT-12': 'RAFS',  # Nature-Positive Solutions
        'INIT-13': 'RAFS',  # Plant Health
        'INIT-14': 'Regional Integrated Initiative',  # AgriLAC Resiliente
        'INIT-15': 'RAFS',  # Aquatic Foods
        'INIT-16': 'RAFS',  # Resilient Cities
        'INIT-17': 'RAFS',  # Sustainable Animal Productivity
        'INIT-18': 'Regional Integrated Initiative',  # Asian Mega-Deltas
        'INIT-19': 'RAFS',  # Mixed Farming Systems
        'INIT-20': 'Regional Integrated Initiative',  # Transforming Agrifood Systems in South Asia
        'INIT-21': 'Regional Integrated Initiative',  # Diversification in E & S Africa
        'INIT-22': 'Regional Integrated Initiative',  # W & C African Food Systems Transformation
        'INIT-23': 'ST',   # Climate Resilience
        'INIT-24': 'ST',   # Foresight
        'INIT-25': 'ST',   # Digital Innovation
        'INIT-26': 'ST',   # Gender Equality
        'INIT-27': 'ST',   # National Policies and Strategies
        'INIT-28': 'ST',   # NEXUS Gains
        'INIT-29': 'ST',   # Rethinking Food Markets
        'INIT-30': 'ST',   # Sustainable Healthy Diets
        'INIT-31': 'ST',   # Agroecology
        'INIT-32': 'ST',   # Low-Emission Food Systems
        'INIT-33': 'ST',   # Fruits and Vegetables
        'INIT-34': 'RAFS', # Livestock and Climate
        'INIT-35': 'ST',   # Fragility, Conflict and Migration
        'PLAT-01': 'ST',   # Gender
        'PLAT-02': 'ST',   # Climate
        'PLAT-03': 'ST',   # Environment
        'PLAT-04': 'ST',   # Nutrition
        'PLAT-05': 'ST',   # Poverty
        'SGP-01': 'GI',    # RTB
        'SGP-02': 'GI',    # AVISA
        'SGP-03': 'GI',    # Genome Editing
        'SGP-04': 'GI',    # AGGRi2
        'SGP-05': 'ST'     # Adaptation Insights
    }
    
    input_file = 'input/export_data_table_results_20251203_101413CET.xlsx'
    
    # Read the Excel file
    print("Reading input file...")
    data_df = pd.read_excel(input_file)
    
    # Add Action Area column based on Submitter
    print("Adding Action Area column...")
    data_df['Action Area'] = data_df['Submitter'].apply(
        lambda x: get_regional_initiative(x, initiative_to_ri)
    )
    
    # Save back to the original file
    print(f"Saving Action Area column back to {input_file}...")
    data_df.to_excel(input_file, index=False)
    print("Done!")
    
    # Print statistics
    total_rows = len(data_df)
    mapped_rows = data_df['Action Area'].notna().sum()
    print(f"\nMapping Statistics:")
    print(f"Total rows: {total_rows}")
    print(f"Successfully mapped rows: {mapped_rows}")
    print(f"Mapping success rate: {(mapped_rows/total_rows)*100:.2f}%")
    
    # Print Action Area distribution
    print("\nAction Area Distribution:")
    print(data_df['Action Area'].value_counts(dropna=False))
    
    # Print sample of unmapped submitters to help debugging
    unmapped = data_df[data_df['Action Area'].isna()]['Submitter'].unique()
    print("\nSample of unmapped submitters (first 10):")
    print(sorted(unmapped)[:10])

if __name__ == "__main__":
    main() 