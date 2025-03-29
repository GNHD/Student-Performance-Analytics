import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

def train_and_save_gb_model():
    """
    Trains a Gradient Boosting model and saves it along with the scaler.
    """
    # Load dataset
    data = pd.read_csv("Data/student_performance_data.csv")

    # Preprocess data
    X, y, scaler = preprocess_data(data)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting model
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(gb_model, "student_performance_gb.pkl")
    print("✅ Gradient Boosting model saved as 'student_performance_gb.pkl'")

    # Save the scaler (specific to Gradient Boosting)
    joblib.dump(scaler, "scaler_gb.pkl")
    print("✅ Scaler saved as 'scaler_gb.pkl'")

if __name__ == "__main__":
    train_and_save_gb_model()


