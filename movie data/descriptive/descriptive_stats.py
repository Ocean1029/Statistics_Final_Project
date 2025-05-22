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


csv_path = 'movie data\dataset\movie_summary.csv'
# 'Final Project\movie_summary.csv'

summary = descriptive_stats(csv_path)
print(summary)

# exporting the summary to a CSV file and put it into ./descriptive_stats
if not os.path.exists('descriptive_stats'):
    os.makedirs('descriptive_stats')
summary.to_csv('descriptive_stats/summary.csv', index=True)
