import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def preprocess_data(df):
    required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']
    df = df.copy()
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[required_columns] = df[required_columns].fillna(df[required_columns].mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[required_columns])

    return X_scaled, df['Final_Performance'], scaler

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f" {model_name} Performance:")
    print(f" Mean Absolute Error (MAE): {mae:.4f}")
    print(f" Mean Squared Error (MSE): {mse:.4f}")
    print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" R² Score: {r2:.4f}")
    print("-" * 50)
    
    return mae, mse, rmse, r2

def train_and_compare_models():
    data = pd.read_csv("Data/student_performance_data.csv")

    X, y, scaler = preprocess_data(data)
    y = pd.to_numeric(y, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)

    
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest Regressor")
    gb_results = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting Regressor")

   
    if gb_results[3] > rf_results[3]:  
        best_model = gb_model
        best_model_name = "Gradient Boosting Regressor"
    else:
        best_model = rf_model
        best_model_name = "Random Forest Regressor"

    joblib.dump((best_model, scaler), "student_performance_model.pkl")
    
    print(f" Model Training Completed! Best Model : {best_model_name}")

if __name__ == "__main__":
    train_and_compare_models()
