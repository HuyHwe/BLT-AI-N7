import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Import your custom linear regression
from linearRegression import LinearRegression

def main():
    # Load your data
    print("Loading data...")
    df = pd.read_csv('car_detail_cleaned.csv')
    
    # Basic preprocessing
    df = df.dropna(subset=['Price'])
    
    # Define categorical columns
    categorical_cols = [
        'origin', 'condition', 'car_model', 'exterior_color', 'interior_color',
        'num_of_doors', 'seating_capacity', 'engine', 'transmission',
        'drive_type', 'fuel_consumption', 'grade', 'car_name'
    ]
    
    # Filter to existing columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Fill missing values
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Separate features and target
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply log transformation to target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Apply target encoding to categorical columns
    print("Applying target encoding...")
    target_encoder = TargetEncoder(target_type='continuous', smooth='auto', cv=5)
    
    # Fit and transform categorical columns
    X_train_cat_encoded = target_encoder.fit_transform(X_train[categorical_cols], y_train_log)
    X_test_cat_encoded = target_encoder.transform(X_test[categorical_cols])
    
    # Get numerical columns
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Combine encoded categorical and numerical features
    if numerical_cols:
        X_train_processed = np.hstack([X_train_cat_encoded, X_train[numerical_cols].values])
        X_test_processed = np.hstack([X_test_cat_encoded, X_test[numerical_cols].values])
    else:
        X_train_processed = X_train_cat_encoded
        X_test_processed = X_test_cat_encoded
    
    # Scale all features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    # Convert to float64
    X_train_scaled = X_train_scaled.astype(np.float64)
    X_test_scaled = X_test_scaled.astype(np.float64)
    y_train_log = y_train_log.astype(np.float64)
    
    # Train the model
    print("Training model...")
    model = LinearRegression(learning_rate=0.01, max_iters=1000)
    model.fit(X_train_scaled, y_train_log)
    
    # Make predictions
    y_train_pred_log = model.predict(X_train_scaled)
    y_test_pred_log = model.predict(X_test_scaled)
    
    # Convert back to original scale
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Testing RMSE:  ${test_rmse:,.2f}")
    print(f"Training R²:   {train_r2:.4f}")
    print(f"Testing R²:    {test_r2:.4f}")
    
    # Show sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_test_pred[i]
        error = abs(predicted - actual) / actual * 100
        print(f"Actual: ${actual:,.0f}, Predicted: ${predicted:,.0f}, Error: {error:.1f}%")

if __name__ == "__main__":
    main()