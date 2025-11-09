#!/usr/bin/env python3
"""
House Price Prediction using ML
Author: Ashish Jha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess house price data"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Feature engineering
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = 2024 - df['YearBuilt']
    
    if 'TotalSF' not in df.columns and all(col in df.columns for col in ['1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def train_linear_models(X_train, y_train, X_test, y_test):
    """Train linear regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        
        results[name] = {'model': model, 'r2': r2, 'rmse': rmse}
    
    return results

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with GridSearchCV"""
    print("\nTraining Gradient Boosting with GridSearchCV...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    grid_search = GridSearchCV(
        gb, param_grid, cv=3, 
        scoring='r2', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nGradient Boosting Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    
    return best_model, r2, rmse

def plot_predictions(y_test, y_pred, model_name):
    """Plot actual vs predicted prices"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'{model_name}: Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig(f'results/{model_name.replace(" ", "_")}_predictions.png')
    plt.close()

if __name__ == "__main__":
    # Load data (Boston Housing or Ames Housing Dataset)
    df = load_and_preprocess_data('data/house_prices.csv')
    
    # Separate features and target
    X = df.drop('SalePrice', axis=1)  # Adjust column name as needed
    y = df['SalePrice']
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target variable range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear models
    linear_results = train_linear_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Train Gradient Boosting with hyperparameter tuning
    gb_model, gb_r2, gb_rmse = train_gradient_boosting(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Find best model
    all_models = {**linear_results, 'Gradient Boosting': {'model': gb_model, 'r2': gb_r2, 'rmse': gb_rmse}}
    best_model_name = max(all_models, key=lambda x: all_models[x]['r2'])
    best_model = all_models[best_model_name]['model']
    best_r2 = all_models[best_model_name]['r2']
    
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"R² Score: {best_r2:.4f}")
    print(f"{'='*50}")
    
    # Save models
    joblib.dump(best_model, 'models/best_house_price_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Plot predictions
    y_pred = best_model.predict(X_test_scaled)
    plot_predictions(y_test, y_pred, best_model_name)
    
    print("\nModels saved successfully!")
    print(f"R² score of {best_r2:.2f} achieved!")
    
    # Feature importance (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print("\nTop 10 Important Features:")
        print(importance_df.to_string(index=False))
