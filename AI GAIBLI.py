# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import datetime
import gdown  # Library for downloading files from Google Drive

# Load Data from Google Drive
file_id = '1hkebti9cgRECbze77SKfm8b_eGBJkTnL'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'ai_ghibli_trend_dataset.csv'

try:
    gdown.download(url, output, quiet=False)
    AISG = pd.read_csv(output)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error occurred while loading data: {e}")
    # Use sample data as fallback
    print("Using sample data instead...")
    data = {
        'image_id': range(1, 101),
        'likes': np.random.randint(100, 10000, 100),
        'shares': np.random.randint(10, 1000, 100),
        'platform': np.random.choice(['Instagram', 'TikTok', 'Twitter'], 100)
    }
    AISG = pd.DataFrame(data)

# Data Exploration
print("\nData sample:")
print(AISG.head(10))

print("\nData information:")
AISG.info()

print("\nDescriptive statistics:")
print(AISG.describe())

# Data Cleaning and Processing
# (Added column existence checks before processing to avoid errors)

# List of expected columns
expected_columns = ['image_id', 'user_id', 'prompt', 'likes', 'shares', 'comments',
                  'platform', 'generation_time', 'gpu_usage', 'file_size_kb',
                  'resolution', 'style_accuracy_score', 'is_hand_edited',
                  'ethical_concerns_flag', 'creation_date', 'top_comment']

# Check for essential columns
if 'platform' not in AISG.columns:
    AISG['platform'] = 'Unknown'  # Default value if column doesn't exist

if 'likes' not in AISG.columns:
    AISG['likes'] = np.random.randint(100, 10000, len(AISG))  # Random values if column doesn't exist

# 1. Text Column Processing
# Remove extra spaces from platform column
AISG["platform"] = AISG["platform"].str.strip().str.replace(" ", "_")

# 2. Date Column Processing (if exists)
if 'creation_date' in AISG.columns:
    try:
        AISG['creation_date'] = pd.to_datetime(AISG['creation_date'])
        AISG['year'] = AISG['creation_date'].dt.year
        AISG['month'] = AISG['creation_date'].dt.month
        AISG['day'] = AISG['creation_date'].dt.day
        AISG['weekday'] = AISG['creation_date'].dt.weekday
    except:
        print("Failed to convert date column, it will be ignored")
else:
    print("Date column not found in data")

# 3. Categorical Column Processing
if 'is_hand_edited' in AISG.columns:
    AISG['is_hand_edited'] = AISG['is_hand_edited'].map({'Yes': 1, 'No': 0})
else:
    AISG['is_hand_edited'] = 0  # Default: Not hand-edited

if 'ethical_concerns_flag' in AISG.columns:
    AISG['ethical_concerns_flag'] = AISG['ethical_concerns_flag'].map({'Yes': 1, 'No': 0})
else:
    AISG['ethical_concerns_flag'] = 0  # Default: No ethical concerns

# Encode image resolution (if column exists)
if 'resolution' in AISG.columns:
    resolution_map = {'512x512': 0, '1024x1024': 1, '2048x2048': 2}
    AISG['resolution'] = AISG['resolution'].map(resolution_map)
else:
    AISG['resolution'] = 0  # Default: 512x512 resolution

# Platform encoding
le = LabelEncoder()
AISG['platform_encoded'] = le.fit_transform(AISG['platform'])

# 4. Remove unnecessary columns (with existence check first)
columns_to_drop = []
for col in ['image_id', 'user_id', 'prompt', 'top_comment', 'creation_date', 'platform']:
    if col in AISG.columns:
        columns_to_drop.append(col)

AISG_cleaned = AISG.drop(columns_to_drop, axis=1)

# 5. Handle Missing Values
# Select only numeric columns for median imputation
numeric_cols = AISG_cleaned.select_dtypes(include=np.number).columns
AISG_cleaned_numeric = AISG_cleaned[numeric_cols]

imputer = SimpleImputer(strategy='median')
AISG_cleaned[numeric_cols] = imputer.fit_transform(AISG_cleaned_numeric)

# Data Analysis
print("\nPlatform distribution:")
print(AISG['platform'].value_counts())

# Data Visualization
plt.figure(figsize=(12, 6))
sns.boxplot(x='platform', y='likes', data=AISG)
plt.title('Likes Distribution by Platform')
plt.xticks(rotation=45)
plt.show()

# Model Building
if len(AISG_cleaned.columns) > 1:  # Ensure there are enough features
    # Split data
    target_cols = ['likes', 'shares', 'comments']
    features_to_drop = [col for col in target_cols if col in AISG_cleaned.columns]

    X = AISG_cleaned.drop(features_to_drop, axis=1, errors='ignore')  # Features
    if 'likes' in AISG_cleaned.columns and pd.api.types.is_numeric_dtype(AISG_cleaned['likes']):
        y = AISG_cleaned['likes']  # Target (predicting number of likes)
    else:
        print("Column 'likes' not found or not numeric in AISG_cleaned. Cannot build model.")
        X = pd.DataFrame()
        y = pd.Series(dtype='float64')

    if not X.empty:
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Train Linear Regression model
        print("\nTraining Linear Regression model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)

        # Evaluate model
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        print(f"Linear Regression - MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}")

        # Train Random Forest model
        print("\nTraining Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        # Evaluate model
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        print(f"Random Forest - MSE: {mse_rf:.2f}, R2: {r2_rf:.2f}")

        # Compare models
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred_lr, alpha=0.5, label='Linear Regression')
        plt.scatter(y_test, y_pred_rf, alpha=0.5, label='Random Forest')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.title('Model Performance Comparison')
        plt.legend()
        plt.show()

        # Feature importance analysis for Random Forest
        feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances)
        plt.title('Feature Importance in Random Forest Model')
        plt.show()
    else:
        print("No valid features available for model building after data processing.")
else:
    print("Not enough features to build model. Check your data.")
