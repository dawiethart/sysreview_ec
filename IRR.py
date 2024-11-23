
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load the datasets
coder1_df = pd.read_csv("coding_review_1stCoder.csv")
coder2_df = pd.read_csv("coding_review_2ndCoder.csv", delimiter=";")

# Align datasets to only keep matching titles
merged_df = pd.merge(
    coder1_df, coder2_df,
    on="Titel",
    suffixes=("_coder1", "_coder2")
)

# Define variables excluding "effect:trust loss"
variables_coder1 = [
    "pos. findings ec hypothesis_coder1",
    "neg. findings ec hypothesis_coder1",
    "effect: polarization_coder1",
    "effect: misinformation_coder1",
    "effect: no effect_coder1",
    "antecedent: recommender system_coder1",
    "antecedent and property: user behaviour_coder1",
    "antecedent: offline fragmentation _coder1"
]

variables_coder2 = [
    "pos. findings ec hypothesis_coder2",
    "neg. findings ec hypothesis_coder2",
    "effect: polarization_coder2",
    "effect: misinformation_coder2",
    "effect: no effect_coder2",
    "antecedent: recommender system_coder2",
    "antecedent and property: user behaviour_coder2",
    "antecedent: offline fragmentation _coder2"
]

# Extract and standardize the data for the selected variables
filtered_data = merged_df[variables_coder1 + variables_coder2]
filtered_data_standardized = filtered_data.copy()
filtered_data_standardized[variables_coder1] = filtered_data[variables_coder1].astype(int)
filtered_data_standardized[variables_coder2] = filtered_data[variables_coder2].astype(int)

# Flatten the data for Cohen's kappa calculation
coder1_data = filtered_data_standardized[variables_coder1].values.flatten()
coder2_data = filtered_data_standardized[variables_coder2].values.flatten()

# Calculate Cohen's kappa
kappa_score = cohen_kappa_score(coder1_data, coder2_data)

print(f"Cohen's kappa (excluding 'effect:trust loss'): {kappa_score}")
