import pandas as pd
import scipy.stats as stats
from glob import glob
import os
import statsmodels.stats.multicomp as mc

# Define directory containing the CSV files
data_dir = 'finalised_data'

# Get list of all CSV file paths
file_paths = glob(os.path.join(data_dir, '*.csv'))

# Conditions corresponding to the files
conditions = (['Negative'] * 8) + (['Neutral'] * 8) + (['Positive'] * 8)

# Generate IDs based on the number of files
ids = range(len(file_paths))

# Function to calculate points_diff
def calculate_points_diff(df):
    df['points_diff'] = df['points'].diff()
    df['points_diff'].fillna(0, inplace=True)
    return df

# Function to calculate big spike threshold
def calculate_threshold(df, threshold_fraction=0.02):
    max_points = df['points'].max()
    return max_points * threshold_fraction

# Function to classify big spikes
def classify_and_count_spikes(df, threshold_fraction=0.02):
    threshold = calculate_threshold(df, threshold_fraction)
    df['big_spike'] = df['points_diff'].apply(lambda x: 1 if x >= threshold else 0)
    big_spike_count = df['big_spike'].sum()
    return df, big_spike_count

# List to hold processed dataframes
processed_dfs = []
big_spike_counts = []
both_conditions_bsc = []

# Process each file
for file_path, condition, id_ in zip(file_paths, conditions, ids):
    df = pd.read_csv(file_path)
    df = calculate_points_diff(df)
    df, big_spike_count = classify_and_count_spikes(df)
    df['ID'] = id_
    df['condition'] = condition
    df['category'] = 'Non-Neutral' if condition != 'Neutral' else 'Neutral'
    processed_dfs.append(df)
    big_spike_counts.append({'ID': id_, 'category': df['category'].iloc[0], 'big_spike_count': big_spike_count})
    both_conditions_bsc.append({'ID': id_, 'condition': df['condition'].iloc[0], 'big_spike_count': big_spike_count})

# Combine all processed dataframes into a single dataframe
combined_df = pd.concat(processed_dfs, ignore_index=True)
df = pd.DataFrame(big_spike_counts)
both_conditions_df = pd.DataFrame(both_conditions_bsc)

# Separate the data into two groups
neutral_spikes = df[df['category'] == 'Neutral']['big_spike_count']
negative_spikes = both_conditions_df[both_conditions_df['condition'] == 'Negative']['big_spike_count']
positive_spikes = both_conditions_df[both_conditions_df['condition'] == 'Positive']['big_spike_count']
non_neutral_spikes = df[df['category'] == 'Non-Neutral']['big_spike_count']

# Perform Mann-Whitney U Test
u_stat, p_value = stats.mannwhitneyu(non_neutral_spikes, neutral_spikes, alternative='two-sided')

# Print the results
print(f"Mann-Whitney U Statistic: {u_stat}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the big spike counts between Neutral and Non-Neutral conditions.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the big spike counts between Neutral and Non-Neutral conditions.")

# Perform Independent Samples t-test
t_stat, p_value = stats.ttest_ind(non_neutral_spikes, neutral_spikes, equal_var=False)  # Use equal_var=False if variances are not equal

# Print the results
print(f"\nT-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the mean big spike counts between Neutral and Non-Neutral conditions.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the mean big spike counts between Neutral and Non-Neutral conditions.")

# Perform One-Way ANOVA
f_stat, p_value = stats.f_oneway(negative_spikes, neutral_spikes, positive_spikes)

# Print the results of ANOVA
print(f"\nF-statistic: {f_stat}")
print(f"P-value: {p_value}")

# If ANOVA is significant, perform post-hoc test
if p_value < 0.05:
    # Perform Tukey's HSD test
    comp = mc.MultiComparison(df['big_spike_count'], df['category'])
    tukey_result = comp.tukeyhsd()

    # Print Tukey's HSD test result
    print(tukey_result)
    print(tukey_result.summary())

