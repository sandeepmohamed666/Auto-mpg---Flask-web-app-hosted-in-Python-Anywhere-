import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# from flask import Flask
import pickle
# from flask import render_template, request
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# import joblib

# Load dataset
file_path = "auto-mpg.csv"
df = pd.read_csv(file_path)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert horsepower to numeric
if 'horsepower' in df.columns:
    df['horsepower'] = pd.to_numeric(df['horsepower'])

# Drop car name if present
if 'car name' in df.columns:
    df.drop('car name', axis=1, inplace=True)

# Separate features and target
X = df.drop('mpg', axis=1)
y = df['mpg']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)
print(f"Model R2 Score: {score:.2f}")

# Save model using pickle
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

# python train_model.py



# # Load dataset
# df = pd.read_csv("auto-mpg.csv")
# df.replace("?", np.nan, inplace=True)
# # Clean dataset (important)
# # df = df.dropna()

# # Clean missing values
# df.replace("?", np.nan, inplace=True)

# # Convert horsepower to numeric
# df["horsepower"] = pd.to_numeric(df["horsepower"])

# # Drop missing rows
# df.dropna(inplace=True)

# # Features and target
# X = df.drop("mpg", axis=1)
# y = df["mpg"]

# # Encode categorical columns
# X = pd.get_dummies(X, drop_first=True)


# # Target
# y = df["mpg"]
# X = df.drop("mpg", axis=1)

# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42
# )

# # Model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Save model
# print(pickle.dump(model, open("model.pkl", "wb")))

# # Save scaler
# print(pickle.dump(scaler, open("scaler.pkl", "wb")))

# # pip install flask
# # flask run --port 8000