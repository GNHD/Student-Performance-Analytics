import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']
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
    joblib.dump((model, scaler), "student_performance_model.pkl")
    print("Model Training Complete with Gradient Boosting!")

def predict_performance(input_df):
    required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']
    input_df = input_df.copy()
    if 'Student_ID' in input_df.columns:
        input_df = input_df.drop(columns=['Student_ID'])
    for col in required_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    input_df.fillna(input_df.mean(), inplace=True)
    model, scaler = joblib.load("Backend/student_performance_model.pkl")
    X_scaled = scaler.transform(input_df[required_columns])
    predictions = model.predict(X_scaled)
    return predictions

if __name__ == "__main__":
    train_model()
