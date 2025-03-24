# Student Performance Analytics

This project predicts student performance based on academic factors such as **attendance, assignment scores, quiz scores, final exam scores, and study hours**. It includes a **FastAPI backend** for data processing and machine learning predictions, and a **frontend built with HTML, CSS, and JavaScript**.

---

## Features
✅ **Batch Predictions** - Upload a CSV file to predict performance for multiple students.  
✅ **Real-time Predictions** - Enter student details manually for an instant performance prediction.  
✅ **Data Visualization** - Graphical representation of predicted student performance.  
✅ **FastAPI Backend** - Handles data processing, model inference, and API requests.  
✅ **User-friendly Frontend** - Simple web interface to interact with the model.  

---

## Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/GNHD/Student-Performance-Analytics.git
cd Student-Performance-Analytics
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Backend**
```bash
cd Backend
uvicorn api:app --reload
```

### **4️⃣ Run the Frontend**
Open `index.html` in a browser.

---

## API Endpoints

### **1️⃣ Predict from CSV**
- **Endpoint:** `/predict/`
- **Method:** `POST`
- **Request:** Upload a CSV file containing student data.
- **Response:** JSON with predicted performance for each student.

### **2️⃣ Predict from User Input**
- **Endpoint:** `/predict-single`
- **Method:** `POST`
- **Request:** Form data containing student details.
- **Response:** JSON with the predicted performance.

---

## Project Structure
```
📁 Student Performance Predictor
│── 📁 Backend
│   │── api.py  # FastAPI backend for handling requests
│   │── model.py  # Machine learning model training & predictions
│   │── student_performance_model.pkl  # Trained machine learning model
│   │── requirements.txt  # Required Python packages
│
│── 📁 Frontend
│   │── index.html  # Web interface
│   │── styles.css  # Styling for the UI
│   │── script.js  # Handles API requests and visualization
│
└── README.md  # Project documentation
```

---

## Notes
- Ensure **`student_performance_model.pkl`** is in the `Backend/` folder.
- The backend must be running before using the frontend.
- Modify `API_URL` in `script.js` if hosting the backend elsewhere.

---

## Technologies Used
- **Python (FastAPI)**
- **Scikit-Learn (Machine Learning)**
- **HTML, CSS, JavaScript (Frontend)**
- **Matplotlib (Graph Visualization)**
- **Joblib (Model Saving & Loading)**

---

## Contact
For issues or suggestions, raise an issue on GitHub.

---

🚀 **Happy Coding!** 🎯
```

---

