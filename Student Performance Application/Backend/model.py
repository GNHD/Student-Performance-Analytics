import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Define required columns
REQUIRED_COLUMNS = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']

def preprocess_data(df):
    """
    Preprocesses the dataset by converting columns to numeric, handling missing values, and scaling features.
    """
    df = df.copy()

    # Convert required columns to numeric, replacing errors with NaN
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values with column mean
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].fillna(df[REQUIRED_COLUMNS].mean())

    # Ensure the target variable is numeric
    df['Final_Performance'] = pd.to_numeric(df['Final_Performance'], errors='coerce').fillna(0)

    # Scale the feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[REQUIRED_COLUMNS])

    return X_scaled, df['Final_Performance'], scaler

def train_and_save_rf_model():
    """
    Trains a Random Forest model and saves it along with the scaler.
    """
    # Load dataset
    data = pd.read_csv("Data/student_performance_data.csv")

    # Preprocess data
    X, y, scaler = preprocess_data(data)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(rf_model, "student_performance_rf.pkl")
    print("✅ Random Forest model saved as 'student_performance_rf.pkl'")

    # Save the scaler (specific to Random Forest)
    joblib.dump(scaler, "scaler_rf.pkl")
    print("✅ Scaler saved as 'scaler_rf.pkl'")

if __name__ == "__main__":
    train_and_save_rf_model()

