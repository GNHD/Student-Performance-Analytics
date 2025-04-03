document.addEventListener("DOMContentLoaded", function () {
    const API_URL_CSV = "http://127.0.0.1:8000/predict/";
    const API_URL_MANUAL = "http://127.0.0.1:8000/predict-single/";
    let predictionChart = null;
    let manualChart = null; // Chart for manual predictions

    function uploadCSV() {
        const fileInput = document.getElementById("fileInput");
        const modelSelect = document.getElementById("modelSelect");

        if (!fileInput || !fileInput.files.length) {
            alert("⚠️ Please select a CSV file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        formData.append("model_name", modelSelect.value);

        fetch(API_URL_CSV, {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("Server Response:", data);
            if (data.predictions && data.predictions.length > 0) {
                displayPredictions(data.predictions);
            } else {
                alert("⚠️ No predictions received.");
            }
        })
        .catch(error => {
            alert(`❌ Error: ${error.message}`);
            console.error("Error:", error);
        });
    }

    function predictSingle() {
        const modelSelect = document.getElementById("manualModelSelect");
        const attendance = document.getElementById("attendance").value;
        const assignment = document.getElementById("assignment").value;
        const quiz = document.getElementById("quiz").value;
        const finalExam = document.getElementById("final_exam").value;
        const studyHours = document.getElementById("study_hours").value;
        const resultContainer = document.getElementById("manualResult");

        if (!modelSelect || !attendance || !assignment || !quiz || !finalExam || !studyHours) {
            alert("⚠️ Missing input fields in HTML.");
            return;
        }

        const formData = new FormData();
        formData.append("model_name", modelSelect.value);
        formData.append("attendance", attendance);
        formData.append("assignment", assignment);
        formData.append("quiz", quiz);
        formData.append("final_exam", finalExam);
        formData.append("study_hours", studyHours);

        console.log("📤 Sending FormData:", [...formData.entries()]);

        fetch(API_URL_MANUAL, {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            console.log("📥 Full API Response:", JSON.stringify(data, null, 2));

            if (data.Predicted_Performance !== undefined) {
                resultContainer.innerHTML = `<p><strong>Predicted Performance:</strong> ${data.Predicted_Performance}</p>`;
                resultContainer.style.display = "block";

                // Update Graph with new data
                // updateManualChart(
                //     ["Attendance", "Assignment", "Quiz", "Final Exam", "Study Hours"], 
                //     [attendance, assignment, quiz, finalExam, studyHours]
                // );

            } else {
                alert("⚠️ Prediction failed. Check API response.");
            }
        })
        .catch(error => {
            alert(`❌ API Error: ${error.message}`);
            console.error("Error:", error);
        });
    }

    function displayPredictions(predictions) {
        const resultContainer = document.getElementById("resultContainer");
        const resultTable = document.getElementById("resultTable").getElementsByTagName("tbody")[0];
        resultTable.innerHTML = "";

        let studentIDs = [];
        let performanceScores = [];

        predictions.forEach(pred => {
            let row = resultTable.insertRow();
            row.insertCell(0).innerText = pred.Student_ID;
            row.insertCell(1).innerText = pred.Predicted_Performance;

            studentIDs.push(pred.Student_ID);
            performanceScores.push(pred.Predicted_Performance);
        });

        resultContainer.style.display = "block";
        updateChart(studentIDs, performanceScores);
    }

    function updateChart(labels, data) {
        const ctx = document.getElementById("barChart").getContext("2d");

        if (predictionChart) {
            predictionChart.destroy();
        }

        predictionChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Predicted Performance",
                    data: data,
                    backgroundColor: ["#257180", "#F2E5BF", "#FD8B51", "#CB6040"],
                    borderColor: ["#1F5E63", "#D9D0A3", "#E47E4D", "#A04E34"],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    // function updateManualChart(labels, data) {
    //     const ctx = document.getElementById("manualChart").getContext("2d");

    //     if (manualChart) {
    //         manualChart.destroy();
    //     }

    //     manualChart = new Chart(ctx, {
    //         type: "bar",
    //         data: {
    //             labels: labels,
    //             datasets: [{
    //                 label: "Manual Entry Data",
    //                 data: data,
    //                 backgroundColor: ["#257180", "#F2E5BF", "#FD8B51", "#CB6040", "#75B1A9"],
    //                 borderColor: ["#1F5E63", "#D9D0A3", "#E47E4D", "#A04E34", "#5E9A8A"],
    //                 borderWidth: 1
    //             }]
    //         },
    //         options: {
    //             responsive: true,
    //             scales: {
    //                 y: { beginAtZero: true }
    //             }
    //         }
    //     });
    // }

    function downloadTemplate() {
        const csvContent = "data:text/csv;charset=utf-8,Student_ID,Attendance_Percentage,Assignment_Scores,Quiz_Scores,Final_Exam_Score,Study_Hours\n";
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "template.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    window.uploadCSV = uploadCSV;
    window.predictSingle = predictSingle;
    window.downloadTemplate = downloadTemplate;
});

    
