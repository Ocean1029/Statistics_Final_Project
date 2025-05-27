import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the data
df = pd.read_csv('movie_summary.csv')


# Filter the dataset to only include movies from the 21st century
# df = df[df['production_year'] >= 2000]
# # Filter the dataset to only include movies with a total budget within 20,000,000
# df = df[df['production_budget'] <= 20000000]

# Create target variable (total revenue)
df['total_revenue'] = df['domestic_box_office'] + df['international_box_office']

# Create violin plot for total revenue distribution
plt.figure(figsize=(15, 8))
sns.violinplot(x='genre', y='total_revenue', data=df, palette='pastel')
plt.title('Distribution of Total Revenue by Genre')
plt.xlabel('Genre')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('revenue_distribution_by_genre.png')
plt.close()

# Define features
numeric_features = ['production_budget', 'opening_weekend_revenue']
categorical_features = ['genre', 'source', 'sequel']

# Print information about missing values
print("\nMissing values in each column:")
print(df[numeric_features + categorical_features].isnull().sum())

# Prepare the data
X = df[numeric_features + categorical_features]
y = df['total_revenue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare the data for statsmodels
X_train_processed = preprocessor.fit_transform(X_train).toarray()
X_test_processed = preprocessor.transform(X_test).toarray()

# Add a constant for the intercept
X_train_processed = sm.add_constant(X_train_processed)
X_test_processed = sm.add_constant(X_test_processed)

# Convert to DataFrame for easier manipulation
feature_names = ['const'] + numeric_features + list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Reset index for alignment
y_train = y_train.reset_index(drop=True)
X_train_processed_df = X_train_processed_df.reset_index(drop=True)

# Fit the model using statsmodels
model = sm.OLS(y_train, X_train_processed_df).fit()

# Print initial model summary
print("\nInitial Model Summary:")
print(model.summary())

# Perform backward regression
p_value_threshold = 0.05
step = 1
while True:
    # Get p-values
    p_values = model.pvalues
    # Find the feature with the highest p-value
    max_p_value = p_values.max()
    if max_p_value > p_value_threshold:
        # Remove the feature with the highest p-value
        feature_to_remove = p_values.idxmax()
        if feature_to_remove != 'const':  # Ensure the constant is not removed
            X_train_processed_df = X_train_processed_df.drop(columns=[feature_to_remove])
            X_test_processed_df = X_test_processed_df.drop(columns=[feature_to_remove])
            # Refit the model
            model = sm.OLS(y_train, X_train_processed_df).fit()
            print(f"\nStep {step}: Removed {feature_to_remove} due to high p-value: {max_p_value:.4f}")
            step += 1
        else:
            break
    else:
        break

# Print final model summary
print("\nFinal Model Summary:")
print(model.summary())

# Make predictions
y_pred = model.predict(X_test_processed_df)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nModel Performance:')
print(f'Mean Squared Error: {mse:,.2f}')
print(f'R-squared Score: {r2:.4f}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Total Revenue')
plt.ylabel('Predicted Total Revenue')
plt.title('Actual vs Predicted Total Revenue (Including Production Budget)')
plt.tight_layout()
plt.savefig('actual_vs_predicted_with_budget.png')
plt.close()

# Plot coefficient values
plt.figure(figsize=(12, 6))
coef_df = pd.DataFrame({
    'Feature': model.params.index,
    'Coefficient': model.params.values
})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.title('Feature Coefficients (Including Production Budget)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('coefficients_with_budget.png')
plt.close() 