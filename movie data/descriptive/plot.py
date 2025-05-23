import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os
import matplotlib.pyplot as plt
import seaborn as sns

'''
執行程式後輸入要畫圖的檔案名稱，程式會自動在同一資料夾下建立一個新的資料夾，並將所有圖表存入該資料夾中。
'''

def visuals(csv_file_path, show_plots=True, plot_dir=None):
    df = pd.read_csv(csv_file_path)
    report = {}

    for column in df.columns:
        data = df[column]
        stats = {}
        
        if pd.api.types.is_numeric_dtype(data):
            clean_data = data.dropna()
            
            unique_vals = set(clean_data.unique())
            if unique_vals == {0, 1} or unique_vals == {0} or unique_vals == {1}:
                # binary 畫 bar chart
                plt.figure(figsize=(4, 4))
                ax = clean_data.value_counts().sort_index().plot(kind='bar', color='mediumseagreen')
                clean_data.value_counts().sort_index().plot(kind='bar', color='mediumseagreen')
                plt.title(f'Bar Chart (Binary): {column}')
                plt.xlabel(column)
                plt.ylabel('Count')
                plt.tight_layout() # 在每根柱子上加上數字
                for p in ax.patches:
                    ax.annotate(str(int(p.get_height())), 
                                (p.get_x() + p.get_width() / 2, p.get_height()), 
                                ha='center', va='bottom')
                plt.savefig(os.path.join(plot_dir, f'{column}_binary_bar_chart.png'))
                continue  # 畫完 bar chart 就跳過 histogram
            
            if column == 'movie_odid':
                continue  # 跳過 odid 的 histogram
            
            # Plot histogram
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(clean_data, kde=True, bins=20, color='skyblue')
            sns.histplot(clean_data, kde=True, bins=20, color='skyblue')
            plt.title(f'Histogram: {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            # plt.tight_layout()
            # 在每根柱子上加上數字
            for p in ax.patches:
                height = int(p.get_height())
                if height > 0:
                    ax.annotate(str(height),
                                (p.get_x() + p.get_width() / 2, height),
                                ha='center', va='bottom', fontsize=6.5)
            plt.savefig(os.path.join(plot_dir, f'{column}_histogram.png'))
            # plt.show()
                
        else:
            top = data.mode().iloc[0] if not data.mode().empty else None
            freq = data.value_counts().iloc[0] if not data.value_counts().empty else None

            # Plot bar chart
            plt.figure(figsize=(6, 4))
            data.value_counts().head(10).plot(kind='bar', color='coral')
            ax = data.value_counts().head(10).plot(kind='bar', color='coral')
            plt.title(f'Bar Chart: {column}')
            plt.ylabel('Frequency')
            plt.tight_layout()
            # 在每根柱子上加上數字
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())),
                            (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='bottom')
            plt.savefig(os.path.join(plot_dir, f'{column}_bar_chart.png'))
            # plt.show()

    # Correlation Matrix (numeric only)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'correlation_matrix.png'))
        plt.show()

    return df.describe(include='all')

def visuals_intervals(csv_file_path, plot_dir=None, upper_quantile=0.99, bins_main=30):
    df = pd.read_csv(csv_file_path)
    if plot_dir is None:
        plot_dir = './interval_plots'
    os.makedirs(plot_dir, exist_ok=True)

    for column in df.select_dtypes(include=[np.number]).columns:
        clean_data = df[column].dropna()
        # 去除極大值（只保留前99%數據）
        threshold = clean_data.quantile(upper_quantile)
        filtered_data = clean_data[clean_data <= threshold]

        # 畫主要分布區間的 histogram（分更多 bins）
        plt.figure(figsize=(6, 4))
        ax = sns.histplot(filtered_data, bins=bins_main, color='dodgerblue', kde=True)
        plt.title(f'細分主區間 Histogram: {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        # 在每根柱子上加數字
        for p in ax.patches:
            height = int(p.get_height())
            if height > 0:
                ax.annotate(str(height),
                            (p.get_x() + p.get_width() / 2, height * 0.85),
                            ha='center', va='bottom', fontsize=7)
        plt.savefig(os.path.join(plot_dir, f'{column}_interval_histogram.png'))
        plt.close()

input_dir = 'movie data\dataset'
input_ = input() # 用輸的；要加 .csv
input_file = input_ # 或直接換成檔名
input = os.path.join(input_dir, input_file)

output_dir_suf = 'movie data\descriptive\plot'
output_dir = os.path.join(output_dir_suf, input_file.replace('.csv', ''))
os.makedirs(output_dir, exist_ok=True)

# output = visuals(input, show_plots=False, plot_dir=output_dir)
output = visuals_intervals(input, plot_dir=output_dir)
# print(output)
plt.close()