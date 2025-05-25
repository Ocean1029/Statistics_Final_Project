import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold

# Load the data
# Note: Update the file path to match your data file location
df = pd.read_csv('movie_summary.csv')

# Filter the dataset to only include movies with a total budget within 20,000,000
df = df[df['production_budget'] <= 20000000]

# Create target variable (total revenue)
df['total_revenue'] = df['domestic_box_office'] + df['international_box_office']

# Define features
numeric_features = [
    'production_year', 'running_time', 
    'opening_weekend_theaters', 'maximum_theaters', 'theatrical_engagements',
    'domestic_dvd_units', 'domestic_dvd_spending', 'domestic_bluray_units',
    'domestic_bluray_spending', 'production_budget'
]

categorical_features = [
    'sequel', 'distributor', 'creative_type', 'source',
    'production_method', 'genre'
]

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
X_train_processed_df = pd.DataFrame(X_train_processed, columns=['const'] + numeric_features + list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)))
X_test_processed_df = pd.DataFrame(X_test_processed, columns=X_train_processed_df.columns)

# Reset index for alignment
y_train = y_train.reset_index(drop=True)
X_train_processed_df = X_train_processed_df.reset_index(drop=True)

# Remove low-variance features before VIF calculation
selector = VarianceThreshold(threshold=0.005)
X_train_reduced = selector.fit_transform(X_train_processed_df)
selected_columns = X_train_processed_df.columns[selector.get_support()]
X_train_processed_df = pd.DataFrame(X_train_reduced, columns=selected_columns)
X_test_processed_df = X_test_processed_df[selected_columns]

# Ensure the constant is included
if 'const' not in X_train_processed_df.columns:
    X_train_processed_df['const'] = 1
    X_test_processed_df['const'] = 1

# Function to calculate VIF
def calculate_vif(X):
    print("Calculating VIF...")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    # Handle division by zero
    vif_data["VIF"] = vif_data["VIF"].replace([np.inf, -np.inf], np.nan)
    print("VIF calculated")
    return vif_data

# Check for multicollinearity
vif_data = calculate_vif(X_train_processed_df)
print("\nVIF Data:")
print(vif_data)

# Remove features with VIF > 5
while vif_data["VIF"].max() > 5:
    remove_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
    if remove_feature != 'const':  # Ensure the constant is not removed
        X_train_processed_df = X_train_processed_df.drop(columns=[remove_feature])
        X_test_processed_df = X_test_processed_df.drop(columns=[remove_feature])
        vif_data = calculate_vif(X_train_processed_df)
        print(f"\nRemoved {remove_feature} due to high VIF.")
        print(vif_data)
    else:
        break

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
plt.title('Actual vs Predicted Total Revenue')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close() 