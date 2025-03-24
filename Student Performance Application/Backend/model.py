import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Define required columns
required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']

def preprocess_data(df):
    df = df.copy()
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[required_columns] = df[required_columns].fillna(df[required_columns].mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[required_columns])

    return X_scaled, df['Final_Performance'], scaler

def train_model():
    data = pd.read_csv("Data/student_performance_data.csv")

    X, y, scaler = preprocess_data(data)
    y = pd.to_numeric(y, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Save model, scaler, and required columns
    joblib.dump((model, scaler, required_columns), "student_performance_model.pkl")

    print("✅ Model Training Complete! Gradient Boosting Regressor Saved.")

if __name__ == "__main__":
    train_model()

