import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os

def descriptive_stats(csv_file_path):
    df = pd.read_csv(csv_file_path)

    report = {}

    for column in df.columns:
        data = df[column]
        stats = {}
        
        stats['type'] = data.dtype
        stats['missing'] = data.isnull().sum()
        stats['unique'] = data.nunique()
        
        if pd.api.types.is_numeric_dtype(data):
            stats['count'] = data.count()
            stats['mean'] = data.mean()
            stats['std'] = data.std()
            stats['min'] = data.min()
            stats['25%'] = data.quantile(0.25)
            stats['50%'] = data.median()
            stats['75%'] = data.quantile(0.75)
            stats['max'] = data.max()
            stats['skewness'] = skew(data.dropna())
            stats['kurtosis'] = kurtosis(data.dropna())
        else:
            stats['top'] = data.mode().iloc[0] if not data.mode().empty else None
            stats['freq'] = data.value_counts().iloc[0] if not data.value_counts().empty else None
            stats['value_counts'] = data.value_counts().to_dict()
        
        report[column] = stats

    return pd.DataFrame(report).T  # Transpose for better readability


input_dir = 'movie data\dataset'
input_file = 'acting_credits.csv'
input = os.path.join(input_dir, input_file)

output = descriptive_stats(input)
print(output)

# exporting the summary to a CSV file and put it into ./descriptive_stats
output_dir = 'movie data\descriptive\descriptive_stats'
output_name = 'acting_credits.csv'
output_path = os.path.join(output_dir, output_name)

output.to_csv(output_path, index=True)
