from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import pandas as pd
import joblib
from io import StringIO
import os

app = FastAPI()

# Load trained model, scaler, and required columns
model_path = os.path.join(os.path.dirname(__file__), "student_performance_model.pkl")

try:
    model, scaler, required_columns = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError("❌ Model file not found! Make sure 'student_performance_model.pkl' is in the Backend folder.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        extra_columns = set(df.columns) - set(required_columns) - {"Student_ID"}
        if extra_columns:
            df.drop(columns=extra_columns, inplace=True)

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        X = df[required_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        df["Predicted_Performance"] = predictions

        return {"predictions": df[["Student_ID", "Predicted_Performance"]].to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-single")
async def predict_single(
    attendance: float = Form(...),
    assignment_score: float = Form(...),
    quiz_score: float = Form(...),
    final_exam_score: float = Form(...),
    study_hours: float = Form(...),
):
    try:
        input_data = pd.DataFrame([[attendance, assignment_score, quiz_score, final_exam_score, study_hours]], 
                                  columns=required_columns)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        return {"Predicted_Performance": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")
