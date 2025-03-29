from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import pandas as pd
import joblib
from io import StringIO
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define base directory for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# Define available models
AVAILABLE_MODELS = {
    "random_forest": {
        "model_path": os.path.join(BASE_DIR, "student_performance_rf.pkl"),
        "scaler_path": os.path.join(BASE_DIR, "scaler_rf.pkl"),
    },
    "gradient_boosting": {
        "model_path": os.path.join(BASE_DIR, "student_performance_gb.pkl"),
        "scaler_path": os.path.join(BASE_DIR, "scaler_gb.pkl"),
    }
}

# Define required input features
required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']

def load_model_and_scaler(model_name: str):
    """Loads the selected model and scaler based on user input."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model selection! Choose from: {list(AVAILABLE_MODELS.keys())}")

    try:
        model = joblib.load(AVAILABLE_MODELS[model_name]["model_path"])
        scaler = joblib.load(AVAILABLE_MODELS[model_name]["scaler_path"])
        print(f"✅ {model_name} Model & Scaler Loaded Successfully.")
        return model, scaler
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"❌ Model or Scaler not found for '{model_name}' in {BASE_DIR}!")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    """Predict student performance from a CSV file using the selected model."""
    try:
        # Validate model selection
        model, scaler = load_model_and_scaler(model_name)

        # Read uploaded CSV file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty!")

        df = pd.read_csv(StringIO(contents.decode("utf-8")), encoding_errors="ignore")
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        # Ensure all required columns are present
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        # Handle missing Student_IDs
        if "Student_ID" not in df.columns:
            df["Student_ID"] = [f"Student_{i+1}" for i in range(len(df))]

        # Convert to numeric and fill missing values
        X = df[required_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        df["Predicted_Performance"] = predictions

        return {"predictions": df[["Student_ID", "Predicted_Performance"]].to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-single/")
async def predict_single(
    model_name: str = Form(...),
    attendance: float = Form(...),
    assignment: float = Form(...),
    quiz: float = Form(...),
    final_exam: float = Form(...),
    study_hours: float = Form(...)
):
    """Predicts a single student's performance using the selected model."""
    try:
        # Validate model selection
        model, scaler = load_model_and_scaler(model_name)

        # Prepare input data
        input_data = pd.DataFrame(
            [[attendance, assignment, quiz, final_exam, study_hours]], 
            columns=required_columns
        )

        # Convert to numeric and fill missing values
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        return {"Predicted_Performance": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")



