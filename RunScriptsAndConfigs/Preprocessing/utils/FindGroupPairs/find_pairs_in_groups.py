import os
import pandas as pd

def read_all_csv(folder_path):
    all_files = os.listdir(folder_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    combined_df = pd.DataFrame()
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def simplify_section_names(name):
    if '(' in name:
        return name.split(' ')[0]
    return name

def process_sections(df):
    # Simplify section names
    df['Group A'] = df['Group A'].apply(simplify_section_names)
    df['Group B'] = df['Group B'].apply(simplify_section_names)

    # Filter out DTW values of 0
    df = df[df['FID'] > 0].copy()
    df = df[df['Mean Distance'] > 0].copy()
    df_filtered = df[df['Feature Distance'] > 0].copy()

    # Filter out the top 50% values for each pairing
    df_filtered['FeatureDistanceRank'] = df_filtered.groupby(['Group A', 'Group B'])['Feature Distance'].rank(pct=True, method='first')
    df_better_half = df_filtered[df_filtered['FeatureDistanceRank'] <= 0.5]

    # Calculate average for each pair and count occurrences
    avg_values = df_better_half.groupby(['Group A', 'Group B']).agg(
        Avg_FD=('Feature Distance', 'mean'),
        Avg_FID=('FID', 'mean'),
        Avg_MD=('Mean Distance', 'mean'),
        Pair_Count=('Feature Distance', 'size')
    ).reset_index()

    # Find unique best and second best matching sections for each section in Group A
    unique_pairs = pd.DataFrame()
    for section in avg_values['Group A'].unique():
        section_pairs = avg_values[avg_values['Group A'] == section]
        best_pair = section_pairs.sort_values(by='Avg_FD').head(1)
        best_pair['Rank'] = 'Best'
        unique_pairs = pd.concat([unique_pairs, best_pair], ignore_index=True)

        # Exclude the best pair and find the second best
        section_pairs = section_pairs[~section_pairs['Group B'].isin(best_pair['Group B'])]
        second_best_pair = section_pairs.sort_values(by='Avg_FD').head(1)
        second_best_pair['Rank'] = 'Second Best'
        unique_pairs = pd.concat([unique_pairs, second_best_pair], ignore_index=True)

    return unique_pairs

# Define the folder containing your CSV files
csv_folder = "../../../Data/Input/Light/AbstractedData/PASv02/Results_Pair_Search/RF_to_NC"

# Process
combined_df = read_all_csv(csv_folder)
best_pairs_df = process_sections(combined_df)

# Save Results to CSV
output_csv_path = 'best_and_second_best_section_pairings.csv'
best_pairs_df.to_csv(output_csv_path, index=False)

print(f"Best and second-best section pairings saved to {output_csv_path}")
